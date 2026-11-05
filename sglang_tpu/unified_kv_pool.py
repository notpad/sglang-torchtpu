# SPDX-License-Identifier: Apache-2.0
"""Logical-to-Physical Unified KV cache pool for hybrid Full + SWA models on TPU.

Shares ``G = full.num_layers`` physical 5D Pallas KV buffers across Full-Attention
(1 physical page per logical page) and Sliding-Window Attention
(``M_swa = ceil(swa.num_layers / G)`` groups) with zero external fragmentation for
models where ``H_swa == H_full`` and ``D_swa == D_full``. Physical page 0 is
reserved as the dummy-write sink for padded tokens.
"""

from __future__ import annotations

import inspect
import logging
import math
import weakref
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.pool_configurator import (
    MemoryPoolConfig,
    MemoryPoolConfigurator,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_server_args,
)
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_free_fence import HeldPages
from sglang_tpu.kv_mxint8 import GROUP_SIZE as MXINT8_DEFAULT_GROUP_SIZE
from sglang_tpu.kv_mxint8 import cache_bytes_per_token as mxint8_cache_bytes_per_token
from sglang_tpu.kv_pool import (
    GB,
    TpuFusedMHATokenToKVPool,
    _cdiv,
)
from sglang_tpu.tpu_dma_hicache import is_tpu_dma_hicache_backend

logger = logging.getLogger(__name__)


def num_swa_groups(num_full_layers: int, num_swa_layers: int) -> int:
    """Return number of SWA groups ``M_swa = ceil(num_swa_layers / num_full_layers)``."""
    if num_full_layers <= 0 or num_swa_layers <= 0:
        raise ValueError(
            f"Unified KV pool requires positive full ({num_full_layers}) and "
            f"SWA ({num_swa_layers}) layer counts."
        )
    return _cdiv(num_swa_layers, num_full_layers)


def _align_pages_up(num_pages: int, align: int = 8) -> int:
    if num_pages <= 0:
        return 0
    return _cdiv(num_pages, align) * align


def draft_case2b_num_pages(
    target_pages: int,
    target_swa_groups: int,
    draft_swa_groups: int,
    *,
    align_pages: int = 8,
) -> int:
    """Return physical page count ``N_D`` for a unified mixed FA+SWA draft pool."""
    if draft_swa_groups <= target_swa_groups:
        return target_pages
    max_live_swa = max(1, _cdiv(target_pages, 1 + target_swa_groups))
    overflow_pages = (draft_swa_groups - target_swa_groups) * max_live_swa
    return _align_pages_up(target_pages + overflow_pages, align_pages)


def draft_case2c_swa_num_pages(
    target_pages: int,
    target_swa_groups: int,
    *,
    align_pages: int = 8,
) -> int:
    """Return compact physical SWA page count ``N_D`` for an SWA-only (or split) draft pool."""
    max_live_swa = max(align_pages, _cdiv(target_pages, 1 + target_swa_groups))
    return _align_pages_up(max_live_swa, align_pages)


def _window_tokens(sliding_window_size: Optional[int], page_size: int) -> int:
    if sliding_window_size is None:
        return 0
    return _cdiv(sliding_window_size + page_size, page_size) * page_size


def _max_request_tokens(
    shared_tokens: int,
    page_size: int,
    sliding_window_size: Optional[int],
    num_swa_groups: int,
) -> int:
    if sliding_window_size is None:
        unit = (1 + num_swa_groups) * page_size
        return (max(0, shared_tokens) // unit) * page_size
    return max(
        0,
        shared_tokens - num_swa_groups * _window_tokens(sliding_window_size, page_size),
    )


@dataclass(frozen=True)
class AttnLayerSpec:
    """Per-attention-type geometry for either Full or SWA layers.

    Attributes:
        head_num: KV head count on this TP rank (post-TP partitioning).
        head_dim: Unpadded per-head K/V dimension (must satisfy K == V).
        layer_ids: Global model layer indices belonging to this attention type,
            in ascending model execution order.
    """

    head_num: int
    head_dim: int
    layer_ids: Sequence[int]

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)


@dataclass(frozen=True)
class UnifiedPageLayout:
    """Resolved physical page layout for a unified Full + SWA KV pool.

    Attributes:
        page_head_num: Physical KV heads per page (requires ``swa.head_num == full.head_num``).
        page_head_dim: Physical head dimension per page (requires ``swa.head_dim == full.head_dim``).
        num_shared_buffers: Number of shared 5D physical KV tensors ``G = full.num_layers``.
        num_swa_groups: Number of physical SWA page groups per logical SWA token
            ``M_swa = ceil(swa.num_layers / G)``.
    """

    page_head_num: int
    page_head_dim: int
    num_shared_buffers: int
    num_swa_groups: int


def resolve_unified_page_layout(
    full: AttnLayerSpec,
    swa: AttnLayerSpec,
) -> UnifiedPageLayout:
    """Validate matching Full/SWA KV geometry and compute SWA group counts."""
    if (
        full.head_num <= 0
        or full.head_dim <= 0
        or swa.head_num <= 0
        or swa.head_dim <= 0
    ):
        raise ValueError(
            f"Invalid head geometry: full=({full.head_num}, {full.head_dim}), "
            f"swa=({swa.head_num}, {swa.head_dim})."
        )
    if swa.head_num != full.head_num or swa.head_dim != full.head_dim:
        raise ValueError(
            "TpuUnifiedSWAKVPool requires matching Full and SWA KV head geometry "
            f"(H_swa == H_full and D_swa == D_full), got full=({full.head_num}, "
            f"{full.head_dim}), swa=({swa.head_num}, {swa.head_dim})."
        )
    return UnifiedPageLayout(
        page_head_num=full.head_num,
        page_head_dim=full.head_dim,
        num_shared_buffers=full.num_layers,
        num_swa_groups=num_swa_groups(full.num_layers, swa.num_layers),
    )


def _kvc_unified_layout(kvc) -> UnifiedPageLayout:
    tp = get_parallel().attn_tp_size
    mc = kvc.model_config
    return resolve_unified_page_layout(
        full=AttnLayerSpec(
            head_num=mc.get_num_kv_heads(tp),
            head_dim=mc.head_dim,
            layer_ids=kvc.layer_info.full_attention_layer_ids,
        ),
        swa=AttnLayerSpec(
            head_num=mc.get_swa_num_kv_heads(tp),
            head_dim=mc.swa_head_dim or mc.head_dim,
            layer_ids=kvc.layer_info.swa_attention_layer_ids,
        ),
    )


def _mxint8_unified_bytes_per_token(kvc, layout: UnifiedPageLayout) -> int:
    return mxint8_cache_bytes_per_token(
        layout.page_head_num,
        layout.page_head_dim,
        kvc.pool_page_size,
        group_size=kvc.kv_group_size or MXINT8_DEFAULT_GROUP_SIZE,
        layout="blocked",
    )


def _is_supported_spec_algorithm(spec_algorithm) -> bool:
    if spec_algorithm.is_none():
        return True
    if getattr(spec_algorithm, "is_ngram", lambda: False)():
        return True
    if spec_algorithm.is_eagle() and not getattr(
        spec_algorithm, "is_frozen_kv_mtp", lambda: False
    )():
        return True
    if spec_algorithm.is_standalone() or spec_algorithm.is_dflash_family():
        return True
    return False


def _has_draft_kv_pool(spec_algorithm) -> bool:
    return not spec_algorithm.is_none() and not getattr(
        spec_algorithm, "is_ngram", lambda: False
    )()


def _draft_unified_layout_or_none(kvc) -> Optional[UnifiedPageLayout]:
    if not getattr(kvc, "is_hybrid_swa", False) or getattr(
        kvc, "draft_swa_full_capacity", False
    ):
        return None
    full_ids = getattr(kvc.layer_info, "full_attention_layer_ids", None) or ()
    swa_ids = getattr(kvc.layer_info, "swa_attention_layer_ids", None) or ()
    if len(full_ids) <= 0 or len(swa_ids) <= 0:
        return None
    pool_cls = kvc._tpu_mha_pool_class()
    if not (
        isinstance(pool_cls, type) and issubclass(pool_cls, TpuFusedMHATokenToKVPool)
    ):
        return None
    if kvc.use_mla_backend or kvc.model_config.has_asymmetric_kv:
        return None
    try:
        layout = _kvc_unified_layout(kvc)
        if kvc.kv_cache_dtype_str == "mxint8":
            _mxint8_unified_bytes_per_token(kvc, layout)
        return layout
    except (ValueError, NotImplementedError):
        return None


def _unsupported_reason(kvc) -> Optional[str]:
    """Return a human-readable reason why ``kvc`` cannot use the TPU unified KV pool,
    or ``None`` when supported."""
    parallel = get_parallel()
    memory = get_memory()
    disagg = get_disagg()
    disagg_mode = disagg.disaggregation_mode
    disagg_backend = getattr(disagg, "disaggregation_transfer_backend", None)
    pool_cls = kvc._tpu_mha_pool_class()
    spec_algo = kvc.spec_algorithm
    checks = (
        (not kvc.is_hybrid_swa, "requires a hybrid SWA model."),
        (kvc.use_mla_backend, "does not support MLA models."),
        (
            kvc.model_config.has_asymmetric_kv,
            "does not support asymmetric K/V head dimensions.",
        ),
        (
            kvc.mambaish_config is not None
            or kvc.hybrid_gdn_config is not None
            or kvc.hybrid_kda_config is not None,
            "does not support hybrid linear-attention or Mamba models.",
        ),
        (
            kvc.is_draft_worker,
            "target unified pool configurator cannot be used directly on a draft worker.",
        ),
        (
            not _is_supported_spec_algorithm(spec_algo),
            f"does not support speculative_algorithm={getattr(spec_algo, 'name', spec_algo)!r}.",
        ),
        (
            parallel.pp_size > 1 or kvc.pp_size > 1,
            "does not support pipeline parallelism.",
        ),
        (
            parallel.attn_cp_size > 1,
            "does not support prefill context parallelism.",
        ),
        (
            parallel.dcp_enabled or parallel.attn_dcp_size > 1,
            "does not support decode context parallelism.",
        ),
        (
            disagg_mode not in ("null", "prefill", "decode"),
            f"does not support disaggregation_mode={disagg_mode!r}.",
        ),
        (
            disagg_mode != "null" and disagg_backend != "raiden",
            "with PD disaggregation requires --disaggregation-transfer-backend raiden, "
            f"got {disagg_backend!r}.",
        ),
        (
            getattr(disagg, "enable_pd_role_switch", False),
            "does not support --enable-pd-role-switch.",
        ),
        (
            getattr(disagg, "disaggregation_decode_enable_offload_kvcache", False),
            "does not support --disaggregation-decode-enable-offload-kvcache.",
        ),
        (
            getattr(disagg, "disaggregation_decode_retraction_backup", None)
            == "host_pool",
            "does not support --disaggregation-decode-retraction-backup host_pool.",
        ),
        (
            getattr(disagg, "disaggregation_decode_host_receive_threshold", 0.0) > 0,
            "does not support --disaggregation-decode-host-receive-threshold > 0.",
        ),
        (
            disagg_mode != "null" and tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.get(),
            "does not support SGLANG_RAIDEN_PIPELINE_TRANSFER.",
        ),
        (
            memory.enable_hierarchical_cache
            and not is_tpu_dma_hicache_backend(
                getattr(memory, "hicache_io_backend", None)
            ),
            "with --enable-hierarchical-cache requires a TPU DMA "
            f"--hicache-io-backend (e.g. 'raiden'), got "
            f"{getattr(memory, 'hicache_io_backend', None)!r}.",
        ),
        (
            memory.enable_hierarchical_cache
            and getattr(memory, "hicache_storage_backend", None) is not None,
            "with --enable-hierarchical-cache does not support L3 storage "
            f"(--hicache-storage-backend={getattr(memory, 'hicache_storage_backend', None)!r}).",
        ),
        (
            memory.enable_hierarchical_cache
            and getattr(memory, "hicache_host_memory_mode", "cache") == "buffer_only",
            "with --enable-hierarchical-cache does not support "
            "--hicache-host-memory-mode buffer_only.",
        ),
        (
            memory.enable_unified_cache_external_linker,
            "does not support --enable-unified-cache-external-linker.",
        ),
        (
            tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get(),
            "does not support SGLANG_TPU_KV_PACK_HEADS.",
        ),
        (
            tpu_envs.SGLANG_TPU_USE_BLOCK_MAJOR_KV.get(),
            "does not support SGLANG_TPU_USE_BLOCK_MAJOR_KV.",
        ),
        (
            not (
                isinstance(pool_cls, type)
                and issubclass(pool_cls, TpuFusedMHATokenToKVPool)
            ),
            f"requires a TpuFusedMHATokenToKVPool subclass, got {pool_cls}.",
        ),
        (
            kvc.pool_page_size <= 1,
            f"requires page_size > 1, got {kvc.pool_page_size}.",
        ),
    )
    for bad, msg in checks:
        if bad:
            return f"--enable-unified-memory on TPU {msg}"
    try:
        layout = _kvc_unified_layout(kvc)
        if kvc.kv_cache_dtype_str == "mxint8":
            _mxint8_unified_bytes_per_token(kvc, layout)
    except (ValueError, NotImplementedError) as exc:
        return str(exc)
    return None


def _raise_if_unsupported(kvc) -> None:
    reason = _unsupported_reason(kvc)
    if reason is not None:
        raise ValueError(reason)


UNIFIED_KV_POOL_RESOLUTION_SOURCE = "tpu.unified_kv_pool"


def unified_kv_pool_requested(server_args=None) -> bool:
    """Return True if ``--enable-unified-memory`` (or ``SGLANG_TPU_UNIFIED_KV_POOL=1``)
    requested the TPU unified Full+SWA KV pool."""
    if tpu_envs.SGLANG_TPU_UNIFIED_KV_POOL.get():
        return True
    if server_args is None:
        try:
            server_args = get_server_args()
        except ValueError:
            return False
    overrides = getattr(server_args, "_resolved_overrides", None) or ()
    return any(source == UNIFIED_KV_POOL_RESOLUTION_SOURCE for source, _ in overrides)


def unified_kv_pool_enabled(kvc) -> bool:
    """Return True if ``--enable-unified-memory`` is active and validate ``kvc``."""
    if getattr(kvc, "is_draft_worker", False):
        alloc = getattr(kvc, "token_to_kv_pool_allocator", None)
        if alloc is not None:
            if not isinstance(alloc, TpuUnifiedSWATokenToKVPoolAllocator):
                return False
            return _draft_unified_layout_or_none(kvc) is not None
        if not unified_kv_pool_requested(getattr(kvc, "server_args", None)):
            return False
        return _draft_unified_layout_or_none(kvc) is not None
    if not unified_kv_pool_requested(getattr(kvc, "server_args", None)):
        return False
    _raise_if_unsupported(kvc)
    return True


class TpuUnifiedSWAKVPool(BaseSWAKVPool):
    """Unified Hybrid Full/SWA KV pool backed by ``G = full.num_layers`` shared 5D buffers."""

    AttnSpec = AttnLayerSpec

    def __init__(
        self,
        size: int,
        *,
        page_size: int,
        dtype: torch.dtype,
        full: AttnLayerSpec,
        swa: AttnLayerSpec,
        device: str,
        token_to_kv_pool_class: type = TpuFusedMHATokenToKVPool,
        pool_kwargs: Optional[dict] = None,
        start_layer: int = 0,
        enable_memory_saver: bool = False,
    ):
        if tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get():
            raise ValueError(
                "TpuUnifiedSWAKVPool does not support SGLANG_TPU_KV_PACK_HEADS."
            )
        if not (
            isinstance(token_to_kv_pool_class, type)
            and issubclass(token_to_kv_pool_class, TpuFusedMHATokenToKVPool)
        ):
            raise ValueError(
                "TpuUnifiedSWAKVPool requires a TpuFusedMHATokenToKVPool subclass, "
                f"got {token_to_kv_pool_class}."
            )
        self.full = full
        self.swa = swa
        self.layout = resolve_unified_page_layout(full, swa)
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            layer_num=full.num_layers + swa.num_layers,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
        )
        self.head_num = self.layout.page_head_num
        self.head_dim = self.layout.page_head_dim

        G = self.layout.num_shared_buffers
        num_blocks = size // page_size + 1
        self.swa_group_pages = torch.zeros(
            (max(0, self.layout.num_swa_groups - 1), num_blocks), dtype=torch.int32
        )

        extra_kwargs = dict(pool_kwargs) if pool_kwargs else {}
        if extra_kwargs.get("kv_cache_dtype_str") == "mxint8":
            extra_kwargs["kv_layout"] = "blocked"
        self.kv_pool: TpuFusedMHATokenToKVPool = token_to_kv_pool_class(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=self.layout.page_head_num,
            head_dim=self.layout.page_head_dim,
            layer_num=G,
            device=device,
            enable_memory_saver=enable_memory_saver,
            allocation_label="Unified",
            **extra_kwargs,
        )
        if "kv_layout" in extra_kwargs:
            self.kv_layout = extra_kwargs["kv_layout"]
        self._swa_copy_kwargs = (
            {"allow_raiden_backup": False}
            if "allow_raiden_backup"
            in inspect.signature(self.kv_pool.get_cpu_copy).parameters
            else {}
        )

        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        self._swa_layer_group: Dict[int, int] = {}
        for i, gid in enumerate(full.layer_ids):
            self.layers_mapping[gid] = (i, False)
        for j, gid in enumerate(swa.layer_ids):
            self.layers_mapping[gid] = (j, True)
            self._swa_layer_group[gid] = j // G

        self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
        self._allocator_ref: Optional[
            weakref.ReferenceType[TpuUnifiedSWATokenToKVPoolAllocator]
        ] = None
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB
        logger.info(
            "TpuUnifiedSWAKVPool mem usage: %.2f GB, unified size: %d tokens "
            "(shared buffers=%d, swa_groups=%d)",
            self.mem_usage,
            self.size,
            G,
            self.layout.num_swa_groups,
        )

    def _buffer_idx(self, layer_id: int) -> int:
        idx, is_swa = self.layers_mapping[layer_id]
        return idx % self.layout.num_shared_buffers if is_swa else idx

    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_fused_kv_buffer(self._buffer_idx(layer_id))

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_key_buffer(self._buffer_idx(layer_id))

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_pool.get_value_buffer(self._buffer_idx(layer_id))

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.kv_pool.get_kv_buffer(self._buffer_idx(layer_id))

    def set_kv_buffer(self, *args, **kwargs) -> None:
        self.kv_pool.set_kv_buffer(*args, **kwargs)

    def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
        self.full_to_swa_index_mapping = full_to_swa_index_mapping

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        mapping = self.full_to_swa_index_mapping
        return (
            mapping[kv_indices.to(mapping.device)].to(torch.int32).to(kv_indices.device)
        )

    def swa_page_group(self, layer_id: int) -> int:
        return self._swa_layer_group.get(layer_id, 0)

    def get_state(self) -> list[torch.Tensor]:
        return self.kv_pool.get_state()

    def set_state(self, state: list[torch.Tensor]) -> None:
        self.kv_pool.set_state(state)

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return self.kv_pool.get_kv_size_bytes()

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return self.kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return [], [], []

    def expand_swa_pages(self, page_ids: Sequence[int]) -> List[int]:
        """Expand logical SWA page IDs into ``[token_page, group_slot]`` physical page IDs."""
        if not page_ids:
            return []
        if self.swa_group_pages.shape[0] == 0:
            return [int(x) for x in page_ids]
        base = torch.as_tensor(page_ids, dtype=torch.long)
        companions = self.swa_group_pages[:, base].to(torch.long)
        expanded = torch.cat([base.unsqueeze(0), companions], dim=0)
        return expanded.T.reshape(-1).tolist()

    def translate_swa_pages_for_draft(self, page_ids: Sequence[int]) -> List[int]:
        """Translate target base SWA page IDs to Case 2c compact draft SWA page IDs."""
        if not page_ids:
            return []
        alloc = self._allocator_ref() if self._allocator_ref is not None else None
        if alloc is None or alloc.target_swa_to_draft_swa is None:
            return [int(x) for x in page_ids]
        pages = torch.as_tensor(page_ids, dtype=torch.int64)
        return alloc.translate_target_swa_pages_to_draft(pages).tolist()

    def _map_swa_group_indices(
        self, group: int, swa_indices_cpu: torch.Tensor
    ) -> torch.Tensor:
        if group == 0:
            return swa_indices_cpu
        phys_pages = self.swa_group_pages[
            group - 1, swa_indices_cpu // self.page_size
        ].to(torch.long)
        return phys_pages * self.page_size + (swa_indices_cpu % self.page_size)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        full_kv_cpu = self.kv_pool.get_cpu_copy(
            indices, mamba_indices=mamba_indices, req_pool_index=req_pool_index
        )
        swa_kv_cpu = None
        swa_mask = None
        if self.full_to_swa_index_mapping is not None:
            mapping = self.full_to_swa_index_mapping
            swa_indices = mapping[indices.to(mapping.device)]
            swa_mask = swa_indices > 0
            if torch.any(swa_mask):
                valid_swa = swa_indices[swa_mask].to(device="cpu", dtype=torch.long)
                swa_kv_cpu = [
                    self.kv_pool.get_cpu_copy(
                        self._map_swa_group_indices(g, valid_swa),
                        **self._swa_copy_kwargs,
                    )
                    for g in range(self.layout.num_swa_groups)
                ]
                swa_mask = swa_mask.cpu()
        return {"full": full_kv_cpu, "swa": swa_kv_cpu, "swa_mask": swa_mask}

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        self.kv_pool.load_cpu_copy(
            kv_cache_cpu["full"],
            indices,
            mamba_indices=mamba_indices,
            req_pool_index=req_pool_index,
        )
        swa_kv_cpu = kv_cache_cpu["swa"]
        if swa_kv_cpu is None or self.full_to_swa_index_mapping is None:
            return
        mapping = self.full_to_swa_index_mapping
        swa_indices = mapping[indices.to(mapping.device)]
        new_swa_mask = swa_indices > 0
        old_swa_mask = kv_cache_cpu.get("swa_mask")
        if old_swa_mask is not None:
            old_swa_mask = old_swa_mask.to(indices.device)
            row_mask = new_swa_mask[old_swa_mask].cpu()
            swa_indices = swa_indices[old_swa_mask][row_mask.to(indices.device)]
        else:
            row_mask = new_swa_mask.cpu()
            swa_indices = swa_indices[new_swa_mask]
        if swa_indices.numel() == 0:
            return
        valid_swa = swa_indices.to(device="cpu", dtype=torch.long)
        need_filter = not bool(torch.all(row_mask).item())
        for g, grp_cpu in enumerate(swa_kv_cpu):
            if need_filter:
                grp_cpu = self.kv_pool.filter_cpu_copy(grp_cpu, row_mask)
            self.kv_pool.load_cpu_copy(
                grp_cpu, self._map_swa_group_indices(g, valid_swa)
            )


class _SharedPageStore:
    """Shared CPU physical page pool (1 .. num_pages) drawn from by Full and SWA."""

    def __init__(self):
        self.free_pages: Optional[torch.Tensor] = None
        self.page_is_free: Optional[torch.Tensor] = None
        self.staged_pages: list[torch.Tensor] = []
        self.num_staged_pages = 0
        self.held_pages = HeldPages()

    def unheld_free_pages(self) -> int:
        return (
            0 if self.free_pages is None else len(self.free_pages)
        ) + self.num_staged_pages

    def num_free_pages(self) -> int:
        return self.unheld_free_pages() + self.held_pages.num_pages


def _store_attr(name: str):
    return property(
        lambda self: getattr(self._page_store, name),
        lambda self, val: setattr(self._page_store, name, val),
    )


class _SharedPagesAllocator(TpuPagedTokenToKVPoolAllocator):
    """One side (Full or SWA) of the zero-fragmentation Logical-to-Physical unified allocator."""

    free_pages = _store_attr("free_pages")
    staged_pages = _store_attr("staged_pages")
    num_staged_pages = _store_attr("num_staged_pages")
    held_pages = _store_attr("held_pages")
    page_is_free = _store_attr("page_is_free")

    def __init__(
        self,
        *,
        page_store: _SharedPageStore,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        group_pages: Optional[torch.Tensor] = None,
    ):
        self._page_store = page_store
        self.allocated_pages = 0
        self._group_pages = group_pages
        self._pages_per_id = 1 if group_pages is None else group_pages.shape[0] + 1
        self._free_pages_before_take: Optional[torch.Tensor] = None
        self._on_ensure_pages: Optional[Callable[[int], bool]] = None
        self._on_take_pages: Optional[Callable[[torch.Tensor], None]] = None
        self._on_release_pages: Optional[Callable[[torch.Tensor], None]] = None
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
        )

    def available_size(self) -> int:
        return (
            self._page_store.num_free_pages() // self._pages_per_id
        ) * self.page_size

    def allocated_count(self) -> int:
        return (self.allocated_pages // self._pages_per_id) * self.page_size

    def alloc(self, need_size: int):
        return self._count_taken_pages(super().alloc, need_size)

    def alloc_extend(self, *args, **kwargs):
        return self._count_taken_pages(super().alloc_extend, *args, **kwargs)

    def alloc_decode(self, *args, **kwargs):
        return self._count_taken_pages(super().alloc_decode, *args, **kwargs)

    def _count_taken_pages(self, take, *args, **kwargs):
        free_before = self._page_store.num_free_pages()
        out = take(*args, **kwargs)
        if self._group_pages is not None or self._on_take_pages is not None:
            self._take_group_pages()
        self.allocated_pages += free_before - self._page_store.num_free_pages()
        return out

    def _ensure_free_pages(self, num_pages: int) -> bool:
        if self._on_ensure_pages is not None and not self._on_ensure_pages(num_pages):
            self._free_pages_before_take = None
            return False
        ok = super()._ensure_free_pages(num_pages * self._pages_per_id)
        if self._group_pages is not None or self._on_take_pages is not None:
            self._free_pages_before_take = self.free_pages if ok else None
        return ok

    def _take_group_pages(self) -> None:
        before = self._free_pages_before_take
        self._free_pages_before_take = None
        if before is None:
            return
        num_taken = len(before) - len(self.free_pages)
        if num_taken == 0:
            return
        base_pages = before[:num_taken]
        if self._group_pages is not None:
            num_extra = self._group_pages.shape[0] * num_taken
            extra = self.free_pages[:num_extra]
            self._mark_handed_out(extra)
            self._group_pages[:, base_pages.long()] = extra.view(-1, num_taken)
            self.free_pages = self.free_pages[num_extra:]
        if self._on_take_pages is not None:
            self._on_take_pages(base_pages)

    def _release_page_ids(self, *page_ids: torch.Tensor) -> None:
        base_page_ids = tuple(
            ids[ids > 0]
            for ids in (
                p.reshape(-1).to(device="cpu", dtype=torch.int32) for p in page_ids
            )
        )
        flat_base = (
            torch.cat([p.to(torch.int64) for p in base_page_ids])
            if base_page_ids
            else None
        )
        if (
            flat_base is not None
            and flat_base.numel() > 0
            and self._on_release_pages is not None
        ):
            self._on_release_pages(flat_base)
        if (
            self._group_pages is not None
            and flat_base is not None
            and flat_base.numel() > 0
        ):
            all_page_ids = (
                *base_page_ids,
                self._group_pages[:, flat_base].reshape(-1),
            )
        else:
            all_page_ids = base_page_ids
        total_released = sum(ids.numel() for ids in all_page_ids)
        if total_released == 0:
            return
        self.allocated_pages -= total_released
        super()._release_page_ids(*all_page_ids)

    def clear(self) -> None:
        super().clear()
        self.allocated_pages = 0


class TpuUnifiedSWATokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Logical-to-Physical Unified KV allocator for Full + SWA attention on TPU."""

    def __init__(
        self,
        size: int,
        *,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TpuUnifiedSWAKVPool,
        need_sort: bool,
        req_to_token_pool=None,
    ):
        if page_size <= 1:
            raise ValueError(
                f"TpuUnifiedSWATokenToKVPoolAllocator requires page_size > 1, got {page_size}."
            )
        assert isinstance(kvcache, TpuUnifiedSWAKVPool)
        self._page_store = _SharedPageStore()
        self._swa_groups = kvcache.layout.num_swa_groups
        self.need_sort = need_sort
        self._kvcache = kvcache
        kvcache._allocator_ref = weakref.ref(self)
        self._req_to_token_pool = req_to_token_pool
        self.draft_kv_pool: Optional[KVCache] = None
        self.draft_kv_pools: list[KVCache] = []
        self._draft_sidecar_allocator: Optional[TpuPagedTokenToKVPoolAllocator] = None
        self._draft_sidecar_pages_per_id = 0
        self._draft_overflow_pool: Optional[TpuUnifiedSWAKVPool] = None
        self.target_swa_to_draft_swa: Optional[torch.Tensor] = None
        self.draft_full_to_swa_index_mapping: Optional[torch.Tensor] = None
        super().__init__(
            size=size,
            size_swa=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache,
            need_sort=need_sort,
            req_to_token_pool=req_to_token_pool,
        )
        assert not self._swa_req_ring
        self.full_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.kv_pool,
            need_sort=need_sort,
        )
        self.swa_attn_allocator = _SharedPagesAllocator(
            page_store=self._page_store,
            size=size,
            page_size=page_size,
            dtype=dtype,
            device=device,
            kvcache=kvcache.kv_pool,
            need_sort=need_sort,
            group_pages=kvcache.swa_group_pages if self._swa_groups > 1 else None,
        )
        self.clear()

    def clear(self) -> None:
        self.swa_attn_allocator.clear()
        self.full_attn_allocator.clear()
        self.full_to_swa_index_mapping[:-1].fill_(0)
        if self._draft_sidecar_allocator is not None:
            self._draft_sidecar_allocator.clear()
        if self._draft_overflow_pool is not None:
            self._draft_overflow_pool.swa_group_pages.zero_()
        if self.target_swa_to_draft_swa is not None:
            self.target_swa_to_draft_swa.zero_()
        if self.draft_full_to_swa_index_mapping is not None:
            self.draft_full_to_swa_index_mapping[:-1].fill_(0)
        self.free_group = None
        self.swa_free_group = []
        self.swa_page_ids_group = []

    def extra_fence_page_pools(self) -> Tuple[TpuPagedTokenToKVPoolAllocator, ...]:
        if self._draft_sidecar_allocator is None:
            return ()
        return (self._draft_sidecar_allocator,)

    def _sidecar_free_swa_ids(self) -> Optional[int]:
        if self._draft_sidecar_allocator is None or self._draft_sidecar_pages_per_id <= 0:
            return None
        free_tokens = self._draft_sidecar_allocator.available_size()
        return (free_tokens // self.page_size) // self._draft_sidecar_pages_per_id

    def derive_draft_pool_sizes(self, draft_kvc, sizes) -> dict:
        del sizes
        scale = int(getattr(draft_kvc, "loc_space_scale", 1) or 1)
        n_t = self.size_full // self.page_size
        ps = self.page_size * scale
        if not getattr(draft_kvc, "is_hybrid_swa", False):
            return {"max_total_num_tokens": n_t * ps}
        draft_layout = _draft_unified_layout_or_none(draft_kvc)
        if draft_layout is not None:
            n_d = draft_case2b_num_pages(
                n_t, self._swa_groups, draft_layout.num_swa_groups
            )
            draft_tokens = n_d * ps
            return {
                "max_total_num_tokens": draft_tokens,
                "full_max_total_num_tokens": draft_tokens,
                "swa_max_total_num_tokens": draft_tokens,
            }
        swa_ids = getattr(draft_kvc.layer_info, "swa_attention_layer_ids", None) or ()
        full_tokens = n_t * ps
        if getattr(draft_kvc, "draft_swa_full_capacity", False) or len(swa_ids) == 0:
            return {
                "max_total_num_tokens": full_tokens,
                "full_max_total_num_tokens": full_tokens,
                "swa_max_total_num_tokens": full_tokens,
            }
        swa_tokens = draft_case2c_swa_num_pages(n_t, self._swa_groups) * ps
        return {
            "max_total_num_tokens": full_tokens,
            "full_max_total_num_tokens": full_tokens,
            "swa_max_total_num_tokens": swa_tokens,
        }

    def _install_sidecar_allocator(
        self, sidecar: TpuPagedTokenToKVPoolAllocator, pages_per_id: int
    ) -> None:
        assert self._draft_sidecar_allocator is None, (
            "At most one stateful SWA draft KV pool can be coupled to "
            "TpuUnifiedSWATokenToKVPoolAllocator."
        )
        assert self.swa_attn_allocator.allocated_pages == 0, (
            "Draft KV pool must be attached before SWA pages are allocated."
        )
        self._draft_sidecar_allocator = sidecar
        self._draft_sidecar_pages_per_id = pages_per_id
        fence = getattr(self.full_attn_allocator, "kv_free_fence", None)
        if fence is not None:
            fence.attach(sidecar)

    def attach_draft_kv_pool(
        self,
        draft_pool: KVCache,
        *,
        draft_swa_full_capacity: bool = False,
    ) -> None:
        if self.draft_kv_pool is None:
            self.draft_kv_pool = draft_pool
        self.draft_kv_pools.append(draft_pool)

        if draft_swa_full_capacity:
            draft_pool._swa_uses_full_indices = True
            n = self.size_full + self.page_size
            identity_mapping = torch.arange(
                n + 1,
                dtype=torch.int64,
                device=self.full_to_swa_index_mapping.device,
            )
            identity_mapping[-1] = -1
            draft_pool.register_mapping(identity_mapping)
            return

        if isinstance(draft_pool, TpuUnifiedSWAKVPool):
            draft_pool.register_mapping(self.full_to_swa_index_mapping)
            m_d = draft_pool.layout.num_swa_groups
            m_t = self._swa_groups
            if m_d <= 1:
                return
            if m_d <= m_t:
                draft_pool.swa_group_pages = self._kvcache.swa_group_pages[: m_d - 1]
                return
            n_t = self.size_full // self.page_size
            n_d = draft_pool.size // self.page_size
            extra_groups = m_d - m_t
            num_overflow_pages = n_d - n_t
            if num_overflow_pages < extra_groups:
                raise ValueError(
                    f"Draft unified pool size ({n_d} pages) is too small for "
                    f"overflow groups (target={n_t} pages, M_T={m_t}, M_D={m_d})."
                )
            sidecar = TpuPagedTokenToKVPoolAllocator(
                size=num_overflow_pages * self.page_size,
                page_size=self.page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=draft_pool.kv_pool,
                need_sort=self.need_sort,
            )
            self._install_sidecar_allocator(sidecar, extra_groups)
            self._draft_overflow_pool = draft_pool

            def _ensure_overflow(num_pages: int) -> bool:
                return sidecar._ensure_free_pages(num_pages * extra_groups)

            def _take_overflow(base_pages: torch.Tensor) -> None:
                n = int(base_pages.numel())
                need = n * extra_groups
                ok = sidecar._ensure_free_pages(need)
                assert ok
                raw = sidecar.free_pages[:need]
                sidecar.free_pages = sidecar.free_pages[need:]
                sidecar._mark_handed_out(raw)
                cols = base_pages.long()
                if m_t > 1:
                    draft_pool.swa_group_pages[: m_t - 1, cols] = (
                        self._kvcache.swa_group_pages[:, cols]
                    )
                draft_pool.swa_group_pages[m_t - 1 :, cols] = (raw + n_t).view(
                    extra_groups, n
                )

            def _release_overflow(flat_base: torch.Tensor) -> None:
                overflow = (
                    draft_pool.swa_group_pages[m_t - 1 :, flat_base].reshape(-1) - n_t
                )
                draft_pool.swa_group_pages[:, flat_base] = 0
                valid = overflow[overflow > 0]
                if valid.numel() > 0:
                    sidecar._release_page_ids(valid)

            self.swa_attn_allocator._on_ensure_pages = _ensure_overflow
            self.swa_attn_allocator._on_take_pages = _take_overflow
            self.swa_attn_allocator._on_release_pages = _release_overflow
            return

        if isinstance(draft_pool, SWAKVPool):
            if draft_pool.swa_layer_nums == 0:
                draft_pool.register_mapping(self.full_to_swa_index_mapping)
                return
            n_t = self.size_full // self.page_size
            sidecar = TpuPagedTokenToKVPoolAllocator(
                size=draft_pool.size_swa,
                page_size=self.page_size,
                dtype=self.dtype,
                device=self.device,
                kvcache=draft_pool.swa_kv_pool,
                need_sort=self.need_sort,
            )
            self._install_sidecar_allocator(sidecar, 1)
            self.target_swa_to_draft_swa = torch.zeros(
                n_t + 1, dtype=torch.int32, device="cpu"
            )
            self.draft_full_to_swa_index_mapping = (
                self.full_to_swa_index_mapping.clone()
            )
            draft_pool.register_mapping(self.draft_full_to_swa_index_mapping)

            def _ensure_compact(num_pages: int) -> bool:
                return sidecar._ensure_free_pages(num_pages)

            def _take_compact(base_pages: torch.Tensor) -> None:
                n = int(base_pages.numel())
                ok = sidecar._ensure_free_pages(n)
                assert ok
                draft_pages = sidecar.free_pages[:n]
                sidecar.free_pages = sidecar.free_pages[n:]
                sidecar._mark_handed_out(draft_pages)
                self.target_swa_to_draft_swa[base_pages.long()] = draft_pages

            def _release_compact(flat_base: torch.Tensor) -> None:
                draft_pages = self.target_swa_to_draft_swa[flat_base]
                self.target_swa_to_draft_swa[flat_base] = 0
                valid = draft_pages[draft_pages > 0]
                if valid.numel() > 0:
                    sidecar._release_page_ids(valid)

            self.swa_attn_allocator._on_ensure_pages = _ensure_compact
            self.swa_attn_allocator._on_take_pages = _take_compact
            self.swa_attn_allocator._on_release_pages = _release_compact

    def translate_target_swa_pages_to_draft(
        self, target_swa_pages: torch.Tensor
    ) -> torch.Tensor:
        if self.target_swa_to_draft_swa is None:
            return target_swa_pages
        pages_cpu = target_swa_pages.to(device="cpu", dtype=torch.int64)
        out = torch.where(
            pages_cpu > 0,
            self.target_swa_to_draft_swa[pages_cpu.clamp(min=0)].to(torch.int64),
            pages_cpu,
        )
        return out.to(device=target_swa_pages.device, dtype=target_swa_pages.dtype)

    def _translate_swa_slots_for_draft(self, swa_indices: torch.Tensor) -> torch.Tensor:
        assert self.target_swa_to_draft_swa is not None
        ps = self.page_size
        swa_cpu = swa_indices.to(device="cpu", dtype=torch.int64)
        pages = torch.div(swa_cpu.clamp(min=0), ps, rounding_mode="floor")
        draft_pages = self.target_swa_to_draft_swa[pages].to(torch.int64)
        mapped = draft_pages * ps + (swa_cpu % ps)
        out = torch.where(swa_cpu > 0, mapped, swa_cpu)
        assert self.draft_full_to_swa_index_mapping is not None
        return out.to(
            device=self.draft_full_to_swa_index_mapping.device,
            dtype=self.draft_full_to_swa_index_mapping.dtype,
        )

    def set_full_to_swa_mapping(
        self, full_indices: torch.Tensor, swa_indices: torch.Tensor
    ) -> None:
        super().set_full_to_swa_mapping(full_indices, swa_indices)
        if self.draft_full_to_swa_index_mapping is None or full_indices.numel() == 0:
            return
        mapping = self.draft_full_to_swa_index_mapping
        mapping[full_indices.to(mapping.device, torch.int64)] = (
            self._translate_swa_slots_for_draft(swa_indices)
        )

    def clear_full_to_swa_mapping(self, full_indices: torch.Tensor) -> None:
        super().clear_full_to_swa_mapping(full_indices)
        if self.draft_full_to_swa_index_mapping is None or full_indices.numel() == 0:
            return
        mapping = self.draft_full_to_swa_index_mapping
        mapping[full_indices.to(mapping.device, torch.int64)] = 0

    def free_swa(self, free_index: torch.Tensor):
        if (
            self.draft_full_to_swa_index_mapping is not None
            and not self._swa_req_ring
            and free_index.numel() > 0
        ):
            mapping = self.draft_full_to_swa_index_mapping
            mapping[free_index.to(mapping.device)] = 0
        return super().free_swa(free_index)

    def _free_page_count(self) -> int:
        return self._page_store.num_free_pages()

    def _page_ids_needed(self, num_full_pages: int, num_swa_pages: int) -> int:
        return num_full_pages + self._swa_groups * num_swa_pages

    def max_extend_tokens(self) -> int:
        return (self._free_page_count() // (1 + self._swa_groups)) * self.page_size

    def max_request_tokens(self, sliding_window_size: Optional[int]) -> int:
        """Return the maximum single-request context length that fits in this pool."""
        return _max_request_tokens(
            self.size_full,
            self.page_size,
            sliding_window_size,
            self._swa_groups,
        )

    def available_size(self) -> int:
        return self.max_extend_tokens()

    def full_available_size(self) -> int:
        return self._free_page_count() * self.page_size

    def swa_available_size(self) -> int:
        avail = (self._free_page_count() // self._swa_groups) * self.page_size
        sidecar_ids = self._sidecar_free_swa_ids()
        if sidecar_ids is not None:
            avail = min(avail, sidecar_ids * self.page_size)
        return avail

    def conserve_full_available_size(self) -> int:
        return self._size_full - self.full_attn_allocator.allocated_count()

    def conserve_swa_available_size(self) -> int:
        return self._size_swa - self.swa_attn_allocator.allocated_count()

    _conserve_full_available_size = conserve_full_available_size
    _conserve_swa_available_size = conserve_swa_available_size

    @property
    def current_full_capacity(self) -> int:
        return self.full_available_size() + self.full_attn_allocator.allocated_count()

    @property
    def current_swa_capacity(self) -> int:
        return self.swa_available_size() + self.swa_attn_allocator.allocated_count()

    def swa_capacity_and_available(self, *, full_capacity, swa_capacity):
        del full_capacity, swa_capacity
        return (
            (self.current_full_capacity, self.full_available_size()),
            (self.current_swa_capacity, self.swa_available_size()),
        )

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        from sglang.srt.mem_cache.prefill_budget import SharedSWAPrefillBudget

        return SharedSWAPrefillBudget(
            self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens
        )

    def new_pages_available(self, num_full_pages: int, num_swa_pages: int) -> bool:
        if min(num_full_pages, num_swa_pages) < 0:
            return False
        sidecar_ids = self._sidecar_free_swa_ids()
        if sidecar_ids is not None and num_swa_pages > sidecar_ids:
            return False
        return (
            self._page_ids_needed(num_full_pages, num_swa_pages)
            <= self._free_page_count()
        )

    def prealloc_fits_assumes_reclaim(self) -> bool:
        """Whether ``prealloc_fits`` already accounts for evictable RadixCache pages via ``can_reserve``."""
        return True

    def prealloc_ceiling_fits(
        self, full_tokens: int | float, swa_tokens: int | float
    ) -> bool:
        """Return True if a request needing ``(full_tokens, swa_tokens)`` can ever fit in an empty pool."""
        return self.can_reserve(
            full_tokens, swa_tokens, empty_pool=True, require_token_slack=True
        )

    def prealloc_fits(
        self,
        tree_cache,
        full_tokens: int,
        swa_tokens: int,
        *,
        full_budget_tokens: int,
        swa_budget_tokens: int | None = None,
    ) -> bool:
        """Price Full and SWA preallocation together against the shared page store."""
        if full_tokens > full_budget_tokens or (
            swa_budget_tokens is not None and swa_tokens > swa_budget_tokens
        ):
            return False
        full_evictable_tokens = (
            0 if tree_cache is None else tree_cache.full_evictable_size()
        )
        swa_evictable_tokens = (
            0 if tree_cache is None else tree_cache.swa_evictable_size()
        )
        full_tokens += max(
            0, self.full_available_size() + full_evictable_tokens - full_budget_tokens
        )
        if swa_budget_tokens is not None:
            swa_tokens += max(
                0,
                self.swa_available_size() + swa_evictable_tokens - swa_budget_tokens,
            )
        return self.can_reserve(
            full_tokens,
            swa_tokens,
            full_evictable_tokens=full_evictable_tokens,
            swa_evictable_tokens=swa_evictable_tokens,
        )

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        out_indices_cpu: Optional[torch.Tensor] = None,
    ):
        num_new_pages = int((seq_lens_cpu % self.page_size == 1).sum())
        if not self.new_pages_available(num_new_pages, num_new_pages):
            return None
        res = super().alloc_decode(
            seq_lens, seq_lens_cpu, last_loc, out_indices_cpu=out_indices_cpu
        )
        if res is not None and self.draft_full_to_swa_index_mapping is not None:
            full_src = out_indices_cpu if out_indices_cpu is not None else res
            full_idx = full_src.to(self.full_to_swa_index_mapping.device, torch.int64)
            swa_idx = self.full_to_swa_index_mapping[full_idx]
            self.draft_full_to_swa_index_mapping[
                full_idx.to(self.draft_full_to_swa_index_mapping.device)
            ] = self._translate_swa_slots_for_draft(swa_idx)
        return res

    def can_reserve(
        self,
        full_tokens: int | float,
        swa_tokens: int | float,
        *,
        full_evictable_tokens: int = 0,
        swa_evictable_tokens: int = 0,
        empty_pool: bool = False,
        require_token_slack: bool = False,
    ) -> bool:
        if full_tokens < 0 or swa_tokens < 0:
            return False
        if require_token_slack and (
            full_tokens >= self.size_full or swa_tokens >= self.size_swa
        ):
            return False
        ps = self.page_size
        need_f = (math.ceil(full_tokens) + ps - 1) // ps
        need_s = (math.ceil(swa_tokens) + ps - 1) // ps
        need = self._page_ids_needed(need_f, need_s)
        if empty_pool:
            if (
                self._draft_sidecar_allocator is not None
                and self._draft_sidecar_pages_per_id > 0
            ):
                cap_ids = (
                    self._draft_sidecar_allocator.size // ps
                ) // self._draft_sidecar_pages_per_id
                if need_s > cap_ids:
                    return False
            return need <= self.size_full // ps
        free = self._free_page_count()
        sidecar_ids = self._sidecar_free_swa_ids()
        if need <= free and (sidecar_ids is None or need_s <= sidecar_ids):
            return True
        ms = self._swa_groups
        max_f = min(
            self.full_attn_allocator.allocated_pages,
            max(0, int(full_evictable_tokens)) // ps,
        )
        max_s = min(
            self.swa_attn_allocator.allocated_pages // ms,
            max(0, int(swa_evictable_tokens)) // ps,
        )
        if sidecar_ids is not None and need_s > sidecar_ids + max_s:
            return False
        return need <= free + self._page_ids_needed(max_f, max_s)

    def max_new_tokens_for_memory(
        self,
        input_tokens: int,
        max_new_tokens: int,
        *,
        token_capacity: int,
        sliding_window_size: int | None,
        chunk_size: int | None,
    ) -> int | None:
        del token_capacity
        from sglang.srt.mem_cache.prefill_budget import estimate_swa_kv_tokens

        def fits(n: int) -> bool:
            return self.can_reserve(
                input_tokens + n + self.page_size,
                estimate_swa_kv_tokens(
                    input_tokens,
                    n,
                    sliding_window_size=sliding_window_size,
                    page_size=self.page_size,
                    allocation_limit=chunk_size,
                ),
                empty_pool=True,
                require_token_slack=True,
            )

        if not fits(0):
            return None
        if fits(max_new_tokens):
            return max_new_tokens
        lo, hi = 0, max_new_tokens
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if fits(mid) else (lo, mid - 1)
        return lo

    def evict_to_free_tokens(
        self, tree_cache, num_tokens: int, *, swa_num_tokens: Optional[int] = None
    ) -> bool | None:
        if tree_cache is None or not tree_cache.supports_prefix_sharing():
            return None
        req_swa = num_tokens if swa_num_tokens is None else swa_num_tokens
        ps = self.page_size
        need_f, need_s = (
            _cdiv(max(0, int(num_tokens)), ps),
            _cdiv(max(0, int(req_swa)), ps),
        )
        needed_pages = self._page_ids_needed(need_f, need_s)
        short = needed_pages - self._free_page_count()
        if short > 0:
            tree_cache.evict_for_alloc(
                EvictParams(num_tokens=short * ps, swa_num_tokens=0)
            )
        short = needed_pages - self._free_page_count()
        sidecar_ids = self._sidecar_free_swa_ids()
        sidecar_short = (
            0 if sidecar_ids is None else max(0, need_s - sidecar_ids)
        )
        if short > 0 or sidecar_short > 0:
            swa_pages_to_evict = max(_cdiv(max(0, short), self._swa_groups), sidecar_short)
            tree_cache.evict_for_alloc(
                EvictParams(
                    num_tokens=0,
                    swa_num_tokens=swa_pages_to_evict * ps,
                )
            )
        return self.new_pages_available(need_f, need_s)

    def reclaim_for_prealloc(
        self, tree_cache, full_tokens: int, swa_tokens: int
    ) -> str | None:
        ready = self.evict_to_free_tokens(
            tree_cache, full_tokens, swa_num_tokens=swa_tokens
        )
        if ready is None:
            ps = self.page_size
            ready = self.new_pages_available(
                _cdiv(max(0, int(full_tokens)), ps),
                _cdiv(max(0, int(swa_tokens)), ps),
            )
        return (
            None
            if ready
            else f"Unified FULL/SWA reclamation insufficient: needed=({full_tokens}, {swa_tokens})"
        )

    def debug_print(self) -> str:
        return (
            f"#free-pages: {self._free_page_count()}, "
            f"#full-pages: {self.full_attn_allocator.allocated_pages}, "
            f"#swa-pages: {self.swa_attn_allocator.allocated_pages}, "
        )


class TpuUnifiedSWAPoolConfigurator(MemoryPoolConfigurator):
    """Sizes the Logical-to-Physical unified Full+SWA KV pool from ``G = L_full``."""

    def __init__(self, kvc):
        _raise_if_unsupported(kvc)
        layout = _kvc_unified_layout(kvc)
        self._sliding_window_size = kvc.sliding_window_size
        self._num_swa_groups = layout.num_swa_groups
        self._full_layers_num = layout.num_shared_buffers
        self._full_per_token = self._compute_full_per_token(kvc, layout)
        self._page_bytes = (
            kvc.pool_page_size * self._full_per_token * self._full_layers_num
        )
        self._reserved_bytes = self._page_bytes
        self._align_pages = 8
        (
            self._draft_full_layers,
            self._draft_swa_layers,
            self._draft_full_per_token,
            self._draft_swa_per_token,
            self._draft_swa_full_capacity,
            self._draft_homogeneous,
        ) = self._extract_draft_pricing(kvc)

    def _extract_draft_pricing(
        self, kvc
    ) -> Tuple[int, int, int, int, bool, bool]:
        spec_algo = kvc.spec_algorithm
        if not _has_draft_kv_pool(spec_algo):
            return 0, 0, 0, 0, False, True
        aux = getattr(kvc, "spec_aux_config", None)
        if aux is None:
            return 0, 0, 0, 0, False, True
        l_full = int(getattr(aux, "draft_full_num_layers", 0) or 0)
        l_swa = int(getattr(aux, "draft_swa_num_layers", 0) or 0)
        b_full = getattr(aux, "draft_full_bytes_per_token", None)
        b_swa = getattr(aux, "draft_swa_bytes_per_token", None)
        swa_full_cap = bool(getattr(aux, "draft_swa_uses_full_capacity", False))
        homogeneous = bool(getattr(aux, "draft_kv_heads_homogeneous", True))
        if l_full == 0 and l_swa == 0:
            if spec_algo.is_dflash_family():
                dflash_layers = int(getattr(aux, "dflash_draft_num_layers", 0) or 0)
                cell = int(getattr(aux, "dflash_draft_cell_size_per_token", 0) or 0)
                if dflash_layers > 0:
                    l_full = dflash_layers
                    b_full = (
                        cell // dflash_layers if cell > 0 else self._full_per_token
                    )
            else:
                eagle_total = int(getattr(aux, "eagle_draft_num_layers", 0) or 0)
                eagle_swa = int(getattr(aux, "eagle_draft_swa_num_layers", 0) or 0)
                if eagle_total > 0:
                    l_swa = min(eagle_swa, eagle_total)
                    l_full = eagle_total - l_swa
                    if l_swa > 0 and l_full == 0 and eagle_total == l_swa:
                        swa_full_cap = False
        resolved_full = (
            self._full_per_token if b_full is None else int(b_full)
        )
        resolved_swa = (
            resolved_full if b_swa is None else int(b_swa)
        )
        return l_full, l_swa, resolved_full, resolved_swa, swa_full_cap, homogeneous

    def _draft_bytes_for_target_pages(self, n_t: int, page_size: int) -> int:
        l_full = self._draft_full_layers
        l_swa = self._draft_swa_layers
        if l_full <= 0 and l_swa <= 0:
            return 0
        b_full = self._draft_full_per_token
        b_swa = self._draft_swa_per_token
        if l_swa <= 0:
            return (n_t + 1) * page_size * l_full * b_full
        if self._draft_swa_full_capacity:
            return (n_t + 1) * page_size * (l_full * b_full + l_swa * b_swa)
        if l_full > 0 and self._draft_homogeneous:
            m_d = num_swa_groups(l_full, l_swa)
            n_d = draft_case2b_num_pages(
                n_t, self._num_swa_groups, m_d, align_pages=self._align_pages
            )
            return (n_d + 1) * page_size * l_full * b_full
        full_bytes = (n_t + 1) * page_size * l_full * b_full if l_full > 0 else 0
        n_d = draft_case2c_swa_num_pages(
            n_t, self._num_swa_groups, align_pages=self._align_pages
        )
        swa_bytes = (n_d + 1) * page_size * l_swa * b_swa
        return full_bytes + swa_bytes

    @staticmethod
    def _compute_full_per_token(kvc, layout: UnifiedPageLayout) -> int:
        if kvc.kv_cache_dtype_str == "mxint8":
            return _mxint8_unified_bytes_per_token(kvc, layout)
        return kvc._tpu_mha_pool_class().kv_bytes_per_token_per_layer(
            num_kv_heads=layout.page_head_num,
            head_dim=layout.page_head_dim,
            v_head_dim=layout.page_head_dim,
            store_dtype=kvc.kv_cache_dtype,
        )

    def _window_overhead(self, page_size: int) -> int:
        return self._num_swa_groups * _window_tokens(
            self._sliding_window_size, page_size
        )

    def _validate_min_tokens(self, shared_tokens: int, page_size: int) -> None:
        if self._sliding_window_size is not None:
            min_tokens = (1 + self._num_swa_groups) * _window_tokens(
                self._sliding_window_size, page_size
            )
        else:
            min_tokens = (1 + self._num_swa_groups) * page_size
        if shared_tokens <= min_tokens:
            raise ValueError(
                f"Unified KV pool ({shared_tokens} page-id tokens) cannot hold "
                f"even one request: need > {min_tokens} tokens."
            )

    def _total_bytes_for_target_pages(self, n_t: int, page_size: int) -> int:
        page_bytes = page_size * self._full_per_token * self._full_layers_num
        return (n_t + 1) * page_bytes + self._draft_bytes_for_target_pages(
            n_t, page_size
        )

    def calculate_total_physical_pages(
        self, available_bytes: int, page_size: int
    ) -> int:
        """Compute aligned physical page count directly from free HBM quota and Full + Draft layers."""
        page_bytes = page_size * self._full_per_token * self._full_layers_num
        if page_bytes <= 0:
            return 0
        if self._draft_full_layers <= 0 and self._draft_swa_layers <= 0:
            usable = max(0, available_bytes - self._reserved_bytes)
            raw_pages = usable // page_bytes
            return (raw_pages // self._align_pages) * self._align_pages

        max_k = max(0, available_bytes - page_bytes) // page_bytes // self._align_pages
        if (
            max_k <= 0
            or self._total_bytes_for_target_pages(self._align_pages, page_size)
            > available_bytes
        ):
            return 0
        lo, hi = 1, max_k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if (
                self._total_bytes_for_target_pages(mid * self._align_pages, page_size)
                <= available_bytes
            ):
                lo = mid
            else:
                hi = mid - 1
        return lo * self._align_pages

    def max_request_tokens(self, shared_tokens: int, page_size: int) -> int:
        """Return the maximum single-request context length that fits in ``shared_tokens``
        (reserving ``M_swa`` groups only for the trailing sliding window, enabling 256K+ sessions)."""
        return _max_request_tokens(
            shared_tokens,
            page_size,
            self._sliding_window_size,
            self._num_swa_groups,
        )

    def calculate_pool_sizes(
        self, available_bytes: int, page_size: int
    ) -> MemoryPoolConfig:
        shared_tokens = (
            self.calculate_total_physical_pages(available_bytes, page_size) * page_size
        )
        self._validate_min_tokens(shared_tokens, page_size)
        return MemoryPoolConfig(
            max_total_num_tokens=self.max_request_tokens(shared_tokens, page_size),
            full_max_total_num_tokens=shared_tokens,
            swa_max_total_num_tokens=shared_tokens,
        )

    def calculate_pool_sizes_from_max_tokens(
        self, max_total_num_tokens: int, page_size: int
    ) -> MemoryPoolConfig:
        full_tokens = (max(0, max_total_num_tokens) // page_size) * page_size
        raw_shared = (
            full_tokens + self._window_overhead(page_size)
            if self._sliding_window_size is not None
            else full_tokens * (1 + self._num_swa_groups)
        )
        align = self._align_pages * page_size
        shared_tokens = _cdiv(raw_shared, align) * align
        self._validate_min_tokens(shared_tokens, page_size)
        return MemoryPoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=shared_tokens,
            swa_max_total_num_tokens=shared_tokens,
        )


def maybe_create_unified_pool_configurator(kvc) -> Optional[MemoryPoolConfigurator]:
    """Return ``TpuUnifiedSWAPoolConfigurator(kvc)`` when unified KV pool is enabled, else ``None``."""
    if not unified_kv_pool_enabled(kvc) or getattr(kvc, "is_draft_worker", False):
        return None
    return TpuUnifiedSWAPoolConfigurator(kvc)


def maybe_build_unified_kv_pool(
    kvc,
    *,
    full_max_total_num_tokens: Optional[int],
    swa_max_total_num_tokens: Optional[int],
) -> Optional[TpuUnifiedSWAKVPool]:
    """Build ``TpuUnifiedSWAKVPool`` for ``kvc`` when unified KV pool is enabled, else ``None``."""
    if not unified_kv_pool_enabled(kvc):
        return None
    assert full_max_total_num_tokens == swa_max_total_num_tokens
    tp = get_parallel().attn_tp_size
    dcp = get_parallel().attn_dcp_size
    if dcp > 1 and getattr(kvc.model_config, "is_draft_model", False):
        dcp = 1
    head_dim = kvc.model_config.head_dim
    return TpuUnifiedSWAKVPool(
        size=full_max_total_num_tokens,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        full=AttnLayerSpec(
            head_num=kvc.model_config.get_num_kv_heads(tp, dcp),
            head_dim=head_dim,
            layer_ids=kvc.layer_info.full_attention_layer_ids,
        ),
        swa=AttnLayerSpec(
            head_num=kvc.model_config.get_swa_num_kv_heads(tp, dcp),
            head_dim=kvc.model_config.swa_head_dim or head_dim,
            layer_ids=kvc.layer_info.swa_attention_layer_ids,
        ),
        device=kvc.device,
        token_to_kv_pool_class=kvc._tpu_mha_pool_class(),
        pool_kwargs=(
            (kvc._mxint8_pool_kwargs() or None)
            if hasattr(kvc, "_mxint8_pool_kwargs")
            else None
        ),
        start_layer=kvc.layer_info.start_layer,
        enable_memory_saver=get_exec().features.enable_memory_saver,
    )


def maybe_build_unified_kv_pool_allocator(
    kvc,
    *,
    sizes,
    token_to_kv_pool,
    req_to_token_pool,
    need_sort: bool,
) -> Optional[TpuUnifiedSWATokenToKVPoolAllocator]:
    """Build ``TpuUnifiedSWATokenToKVPoolAllocator`` for ``kvc`` when unified KV pool is enabled, else ``None``."""
    if getattr(kvc, "is_draft_worker", False) or not unified_kv_pool_enabled(kvc):
        return None
    return TpuUnifiedSWATokenToKVPoolAllocator(
        sizes.full_max_total_num_tokens,
        page_size=get_schedule().page_size,
        dtype=kvc.kv_cache_dtype,
        device=kvc.device,
        kvcache=token_to_kv_pool,
        need_sort=need_sort,
        req_to_token_pool=req_to_token_pool,
    )
