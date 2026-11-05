from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import msgspec

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.runtime_context import (
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils import is_tpu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_MUSE_LAYER_OUTPUT_DRAFT_ARCHITECTURES = frozenset(
    {"DFlash2DraftModel", "MuseGlimmerAssistantModel"}
)


def _map_muse_target_layer_ids(*, target_hf_config, draft_hf_config, layer_ids):
    architectures = getattr(draft_hf_config, "architectures", None) or []
    uses_layer_outputs = getattr(
        target_hf_config, "model_type", None
    ) == "muse_glimmer" and bool(
        _MUSE_LAYER_OUTPUT_DRAFT_ARCHITECTURES.intersection(architectures)
    )
    return [i + 1 for i in layer_ids] if uses_layer_outputs else layer_ids


class SpecAuxHiddenStateConfig(msgspec.Struct, kw_only=True):
    eagle_use_aux_hidden_state: bool = False
    eagle_draft_num_layers: Optional[int] = None
    # Draft layers whose KV cache uses the target SWA pool capacity.
    eagle_draft_swa_num_layers: Optional[int] = None
    eagle_aux_hidden_state_layer_ids: Any = None
    dflash_use_aux_hidden_state: bool = False
    dflash_draft_num_layers: Optional[int] = None
    dflash_target_layer_ids: Any = None
    # DFLASH draft KV bytes/token; None when unresolved.
    dflash_draft_cell_size_per_token: int | None = None
    # Preserve heads/dim for layout-aware MXINT8 sizing; raw payload bytes lose it.
    dflash_draft_kv_geometry: tuple[int, int] | None = None
    # Per-layer-type draft KV geometry for co-priced unified Target+Draft sizing.
    draft_full_num_layers: int = 0
    draft_swa_num_layers: int = 0
    draft_swa_uses_full_capacity: bool = False
    draft_full_bytes_per_token: int | None = None
    draft_swa_bytes_per_token: int | None = None
    draft_kv_heads_homogeneous: bool = True
    draft_kv_cache_dtype_str: str | None = None


def resolve_spec_aux_hidden_state_config(
    *,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> SpecAuxHiddenStateConfig:
    config = SpecAuxHiddenStateConfig()
    _resolve_eagle_aux_hidden_state(
        config=config,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
    )
    _resolve_dflash_aux_hidden_state(
        config=config,
        server_args=server_args,
        model_config=model_config,
        spec_algorithm=spec_algorithm,
        is_draft_worker=is_draft_worker,
    )
    return config


def _resolve_eagle_aux_hidden_state(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> None:
    if not (
        (spec_algorithm.is_eagle() or spec_algorithm.is_standalone())
        and not is_draft_worker
    ):
        return

    draft_model_config = model_config
    if get_spec().speculative_draft_model_path:
        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=get_spec().speculative_draft_model_path,
            model_revision=get_spec().speculative_draft_model_revision,
            is_draft_model=True,
        )
    num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
    if num_nextn_predict_layers is not None:
        config.eagle_draft_num_layers = int(num_nextn_predict_layers)
    elif get_spec().speculative_draft_model_path:
        config.eagle_draft_num_layers = int(
            max(
                draft_model_config.num_hidden_layers,
                draft_model_config.num_attention_layers,
            )
        )
    else:
        return

    if draft_model_config.is_hybrid_swa and not draft_model_config.is_deepseek_v4_arch:
        config.eagle_draft_swa_num_layers = len(
            draft_model_config.swa_attention_layer_ids
        )

    _populate_draft_kv_pricing(
        config=config,
        server_args=server_args,
        draft_model_config=draft_model_config,
        draft_num_layers=int(config.eagle_draft_num_layers),
        is_dflash=False,
        is_mtp_in_model=(
            num_nextn_predict_layers is not None
            and not bool(get_spec().speculative_draft_model_path)
        ),
    )

    if spec_algorithm.is_eagle3():
        config.eagle_use_aux_hidden_state = True
        try:
            eagle_config = getattr(draft_model_config.hf_config, "eagle_config", None)
            config.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
            config.eagle_aux_hidden_state_layer_ids = eagle_config[
                "eagle_aux_hidden_state_layer_ids"
            ]
        except Exception:
            config.eagle_aux_hidden_state_layer_ids = None


def _resolve_dflash_aux_hidden_state(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    model_config: ModelConfig,
    spec_algorithm: SpeculativeAlgorithm,
    is_draft_worker: bool,
) -> None:
    if spec_algorithm.is_dflash_family() and not is_draft_worker:
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

        # Select target layers to capture for building draft context features.
        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(get_spec().speculative_draft_model_path),
            model_revision=get_spec().speculative_draft_model_revision,
            is_draft_model=True,
        )
        dflash_draft_config = parse_dflash_draft_config(
            draft_hf_config=draft_model_config.hf_config
        )
        draft_num_layers = dflash_draft_config.require_num_layers()
        trained_target_layers = dflash_draft_config.num_target_layers

        target_num_layers = getattr(
            model_config.hf_text_config, "num_hidden_layers", None
        )
        if target_num_layers is None:
            raise ValueError(
                "Block-draft-with-target-kv spec requires target num_hidden_layers "
                f"in config. Got target={target_num_layers}."
            )
        target_num_layers = int(target_num_layers)
        # Loop models: target layer ids span num_hidden_layers * num_loops.
        num_loops = getattr(model_config.hf_text_config, "num_loops", 1)
        target_num_layers = target_num_layers * int(num_loops)

        if (
            trained_target_layers is not None
            and trained_target_layers != target_num_layers
        ):
            logger.warning(
                "Draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                "selecting capture layers based on the runtime target model.",
                trained_target_layers,
                target_num_layers,
            )

        target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
            target_num_layers=int(target_num_layers),
            draft_num_layers=int(draft_num_layers),
        )

        # These Muse drafts use HF layer-output ids, while the Muse target captures
        # before each layer. Legacy Muse drafts already store layer-input ids.
        target_layer_ids = _map_muse_target_layer_ids(
            target_hf_config=model_config.hf_config,
            draft_hf_config=draft_model_config.hf_config,
            layer_ids=target_layer_ids,
        )

        if spec_algorithm.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_config import (
                parse_dspark_draft_config,
            )

            dspark_draft_config = parse_dspark_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            if not dspark_draft_config.require_markov():
                raise ValueError(
                    "DSPARK requires markov_rank > 0 in the draft config, "
                    f"got markov_rank={dspark_draft_config.markov_rank}."
                )
            if dspark_draft_config.target_layer_ids is not None:
                target_layer_ids = list(dspark_draft_config.target_layer_ids)

        attn_tp_size = _attn_tp_size()
        config.dflash_use_aux_hidden_state = True
        config.dflash_draft_num_layers = int(draft_num_layers)
        config.dflash_target_layer_ids = target_layer_ids
        config.dflash_draft_cell_size_per_token = _resolve_dflash_draft_cell_size(
            draft_model_config=draft_model_config,
            draft_num_layers=int(draft_num_layers),
            tp_size=attn_tp_size,
        )
        if config.dflash_draft_cell_size_per_token is not None:
            config.dflash_draft_kv_geometry = (
                draft_model_config.get_num_kv_heads(attn_tp_size),
                draft_model_config.head_dim,
            )
        _populate_draft_kv_pricing(
            config=config,
            server_args=server_args,
            draft_model_config=draft_model_config,
            draft_num_layers=int(draft_num_layers),
            is_dflash=True,
            is_mtp_in_model=False,
        )


def _attn_tp_size() -> int:
    parallel = get_parallel()
    return int(getattr(parallel, "attn_tp_size", None) or parallel.tp_size)


def _resolve_tpu_draft_layer_bytes(
    *,
    server_args: ServerArgs,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    store_dtype,
    dtype_str: str | None,
    use_blocked_mxint8: bool,
) -> int | None:
    if not is_tpu():
        return None
    from sglang_tpu.kv_mxint8 import GROUP_SIZE as MXINT8_DEFAULT_GROUP_SIZE
    from sglang_tpu.kv_mxint8 import (
        cache_bytes_per_token as mxint8_cache_bytes_per_token,
    )
    from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
    from sglang_tpu.tpu_platform import _mha_kv_pool_cls

    if dtype_str == "mxint8":
        try:
            page_size = int(get_schedule().page_size)
        except Exception:  # noqa: BLE001
            page_size = int(getattr(server_args, "page_size", None) or 32)
        group_size = int(
            getattr(server_args, "tpu_mxint8_group_size", None)
            or MXINT8_DEFAULT_GROUP_SIZE
        )
        layout = "blocked" if use_blocked_mxint8 else "merged"
        return int(
            mxint8_cache_bytes_per_token(
                num_kv_heads,
                head_dim,
                page_size,
                group_size=group_size,
                layout=layout,
            )
        )

    backend = get_spec().speculative_draft_attention_backend
    if backend is None:
        try:
            backend = getattr(get_exec().kernel, "attention_backend", None)
        except Exception:  # noqa: BLE001
            backend = getattr(server_args, "attention_backend", None)
    pool_cls = _mha_kv_pool_cls(backend) if backend is not None else None
    if pool_cls is None or not issubclass(pool_cls, TpuFusedMHATokenToKVPool):
        return None
    return int(
        pool_cls.kv_bytes_per_token_per_layer(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            store_dtype=store_dtype,
        )
    )


def _populate_draft_kv_pricing(
    *,
    config: SpecAuxHiddenStateConfig,
    server_args: ServerArgs,
    draft_model_config: ModelConfig,
    draft_num_layers: int,
    is_dflash: bool,
    is_mtp_in_model: bool,
) -> None:
    if draft_num_layers <= 0:
        return
    swa_num_layers = 0
    if draft_model_config.is_hybrid_swa and not getattr(
        draft_model_config, "is_deepseek_v4_arch", False
    ):
        raw_swa_ids = getattr(draft_model_config, "swa_attention_layer_ids", None) or ()
        if is_mtp_in_model:
            swa_num_layers = min(draft_num_layers, len(raw_swa_ids))
        else:
            swa_num_layers = sum(1 for lid in raw_swa_ids if 0 <= lid < draft_num_layers)
    full_num_layers = max(0, draft_num_layers - swa_num_layers)
    config.draft_full_num_layers = full_num_layers
    config.draft_swa_num_layers = swa_num_layers
    config.draft_swa_uses_full_capacity = bool(is_mtp_in_model and swa_num_layers > 0)

    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
    import torch

    try:
        attn_tp_size = _attn_tp_size()
        dtype_str, draft_kv_cache_dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=get_model().kv_cache_dtype,
            speculative_draft_kv_cache_dtype=(
                get_spec().speculative_draft_kv_cache_dtype
            ),
            model=None,
            model_dtype=draft_model_config.dtype,
            is_draft_worker=True,
            is_dflash=is_dflash,
            speculative_draft_attention_backend=(
                get_spec().speculative_draft_attention_backend
            ),
        )
        config.draft_kv_cache_dtype_str = dtype_str
        full_heads = draft_model_config.get_num_kv_heads(attn_tp_size)
        swa_heads = (
            draft_model_config.get_swa_num_kv_heads(attn_tp_size)
            if hasattr(draft_model_config, "get_swa_num_kv_heads")
            else full_heads
        )
        full_dim = int(draft_model_config.head_dim)
        full_v_dim = int(getattr(draft_model_config, "v_head_dim", None) or full_dim)
        swa_dim = int(getattr(draft_model_config, "swa_head_dim", None) or full_dim)
        swa_v_dim = int(
            getattr(draft_model_config, "swa_v_head_dim", None) or full_v_dim
        )
        homogeneous = (
            full_heads == swa_heads
            and full_dim == swa_dim
            and full_v_dim == swa_v_dim
        )
        config.draft_kv_heads_homogeneous = homogeneous
        use_unified_draft = (
            full_num_layers > 0
            and swa_num_layers > 0
            and homogeneous
            and not config.draft_swa_uses_full_capacity
        )
        if dtype_str == "mxint8" and use_unified_draft:
            try:
                _resolve_tpu_draft_layer_bytes(
                    server_args=server_args,
                    num_kv_heads=full_heads,
                    head_dim=full_dim,
                    v_head_dim=full_v_dim,
                    store_dtype=draft_kv_cache_dtype,
                    dtype_str=dtype_str,
                    use_blocked_mxint8=True,
                )
            except (ValueError, NotImplementedError):
                use_unified_draft = False
                config.draft_kv_heads_homogeneous = False
        full_mxint8_blocked = use_unified_draft or (
            getattr(server_args, "tpu_mxint8_full_kv_layout", "merged") == "blocked"
        )
        dtype_size = torch._utils._element_size(draft_kv_cache_dtype)

        tpu_full = _resolve_tpu_draft_layer_bytes(
            server_args=server_args,
            num_kv_heads=full_heads,
            head_dim=full_dim,
            v_head_dim=full_v_dim,
            store_dtype=draft_kv_cache_dtype,
            dtype_str=dtype_str,
            use_blocked_mxint8=full_mxint8_blocked,
        )
        config.draft_full_bytes_per_token = (
            tpu_full
            if tpu_full is not None
            else int(full_heads * (full_dim + full_v_dim) * dtype_size)
        )
        tpu_swa = _resolve_tpu_draft_layer_bytes(
            server_args=server_args,
            num_kv_heads=swa_heads,
            head_dim=swa_dim,
            v_head_dim=swa_v_dim,
            store_dtype=draft_kv_cache_dtype,
            dtype_str=dtype_str,
            use_blocked_mxint8=use_unified_draft,
        )
        config.draft_swa_bytes_per_token = (
            tpu_swa
            if tpu_swa is not None
            else int(swa_heads * (swa_dim + swa_v_dim) * dtype_size)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not resolve draft KV per-layer bytes/token (%s); falling back to "
            "target per-token byte scaling.",
            e,
        )


def _resolve_dflash_draft_cell_size(
    *,
    draft_model_config: ModelConfig,
    draft_num_layers: int,
    tp_size: int | None = None,
) -> int | None:
    """Bytes/token the DFLASH draft KV pool will cost the target's pool budget.

    Resolved from the draft's own attention geometry and the KV dtype the draft
    worker will actually resolve. Returns None if anything is unresolvable,
    leaving callers on layer-count scaling.
    """
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
    from sglang.srt.speculative.dflash_utils import dflash_draft_cell_size_per_token

    try:
        resolved_tp = _attn_tp_size() if tp_size is None else int(tp_size)
        _, draft_kv_cache_dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=get_model().kv_cache_dtype,
            speculative_draft_kv_cache_dtype=(
                get_spec().speculative_draft_kv_cache_dtype
            ),
            model=None,
            model_dtype=draft_model_config.dtype,
            is_draft_worker=True,
            is_dflash=True,
            speculative_draft_attention_backend=(
                get_spec().speculative_draft_attention_backend
            ),
        )
        return dflash_draft_cell_size_per_token(
            draft_model_config=draft_model_config,
            draft_num_layers=draft_num_layers,
            draft_kv_cache_dtype=draft_kv_cache_dtype,
            tp_size=resolved_tp,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not resolve DFLASH draft KV bytes/token (%s); falling back to "
            "layer-count scaling for the KV pool budget.",
            e,
        )
        return None
