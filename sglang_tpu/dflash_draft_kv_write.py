# SPDX-License-Identifier: Apache-2.0
"""Materialize DFlash draft KV into the fused TPU pool.

Host side of kv_fused_rope_write: project K/V per draft layer, decompose the
destination slots into contiguous spans, gather the rope tables, then hand it
all to the single Pallas launch that rotates and scatters every layer. Extend
and verify differ only in how the spans come out, so one path serves both.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.models.dflash import DFlashDraftModel
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
from sglang_tpu.unified_kv_pool import TpuUnifiedSWAKVPool

DraftKVPool = TpuFusedMHATokenToKVPool | SWAKVPool | TpuUnifiedSWAKVPool


def draft_kv_has_swa(kv_pool: object) -> bool:
    """True when the draft KV pool stores one or more SWA layers."""
    if isinstance(kv_pool, TpuUnifiedSWAKVPool):
        return kv_pool.layout.num_swa_groups > 0
    if isinstance(kv_pool, SWAKVPool):
        return kv_pool.swa_layer_nums > 0
    return False


def draft_kv_num_addr_groups(kv_pool: object) -> int:
    """Number of destination address groups expected by dflash_write_draft_kv."""
    if isinstance(kv_pool, TpuUnifiedSWAKVPool):
        return 1 + kv_pool.layout.num_swa_groups
    if isinstance(kv_pool, SWAKVPool):
        if kv_pool.full_layer_nums > 0 and kv_pool.swa_layer_nums > 0:
            return 2
        return 1
    return 1


def draft_kv_fence_tail(kv_pool: DraftKVPool) -> torch.Tensor:
    """1-element slice read after the last KV write for KVFreeFence ordering."""
    if isinstance(kv_pool, TpuUnifiedSWAKVPool):
        inner = kv_pool.kv_pool
        if not isinstance(inner, TpuFusedMHATokenToKVPool):
            raise TypeError(
                f"DFLASH unified draft KV pool needs TpuFusedMHATokenToKVPool, got {type(inner)}."
            )
        return inner.kv_buffer[-1][0, 0, 0, 0, :1]
    if isinstance(kv_pool, SWAKVPool):
        sub = (
            kv_pool.swa_kv_pool
            if kv_pool.swa_layer_nums > 0
            else kv_pool.full_kv_pool
        )
        if not isinstance(sub, TpuFusedMHATokenToKVPool):
            raise TypeError(
                f"DFLASH SWA draft KV pool needs TpuFusedMHATokenToKVPool, got {type(sub)}."
            )
        return sub.kv_buffer[-1][0, 0, 0, 0, :1]
    if isinstance(kv_pool, TpuFusedMHATokenToKVPool):
        return kv_pool.kv_buffer[-1][0, 0, 0, 0, :1]
    raise TypeError(
        f"Unsupported draft KV pool for DFLASH writer: {type(kv_pool)}."
    )


def map_draft_kv_locs(kv_pool: object, full_locs: torch.Tensor) -> torch.Tensor:
    """Translate full-pool token locations into draft KV destination slot(s).

    Returns:
      - ``(*full_locs.shape,)`` when ``draft_kv_num_addr_groups(kv_pool) == 1``
        (FA-only pool, or SWA-only ``SWAKVPool`` mapped to ``[0 .. size_swa]``).
      - ``(G, *full_locs.shape)`` when ``G > 1`` (row 0 = FA slots, rows ``1..G-1`` =
        SWA group slots with ``-1`` on unmapped or dummy positions).
    """
    if not draft_kv_has_swa(kv_pool):
        return full_locs

    orig_device = full_locs.device
    orig_dtype = full_locs.dtype
    if isinstance(kv_pool, SWAKVPool):
        assert kv_pool.full_to_swa_index_mapping is not None
        mapping = kv_pool.full_to_swa_index_mapping
        locs_m = full_locs.to(device=mapping.device, dtype=torch.long)
        raw_swa = mapping[locs_m]
        if kv_pool.full_layer_nums == 0:
            return raw_swa.to(dtype=orig_dtype, device=orig_device)
        valid_swa = (locs_m > 0) & (raw_swa > 0)
        swa_row = torch.where(valid_swa, raw_swa, -1).to(dtype=orig_dtype)
        return torch.stack(
            [locs_m.to(dtype=orig_dtype), swa_row], dim=0
        ).to(device=orig_device)

    if isinstance(kv_pool, TpuUnifiedSWAKVPool):
        assert kv_pool.full_to_swa_index_mapping is not None
        mapping = kv_pool.full_to_swa_index_mapping
        locs_cpu = full_locs.to(device="cpu", dtype=torch.long)
        raw_swa0 = mapping[locs_cpu.to(device=mapping.device)].to(device="cpu")
        valid_swa = (locs_cpu > 0) & (raw_swa0 > 0)
        rows = [
            locs_cpu.to(dtype=orig_dtype),
            torch.where(valid_swa, raw_swa0, -1).to(dtype=orig_dtype),
        ]
        m_d = kv_pool.layout.num_swa_groups
        if m_d > 1:
            ps = int(kv_pool.page_size)
            safe_pages = torch.where(valid_swa, raw_swa0 // ps, 0).to(torch.long)
            offsets = (raw_swa0 % ps).to(dtype=orig_dtype)
            for g in range(1, m_d):
                phys_pages = kv_pool.swa_group_pages[g - 1][safe_pages].to(
                    dtype=orig_dtype
                )
                rows.append(
                    torch.where(
                        valid_swa & (phys_pages > 0),
                        phys_pages * ps + offsets,
                        -1,
                    )
                )
        return torch.stack(rows, dim=0).to(device=orig_device)

    return full_locs


def _layer_id(layer: object, fallback_idx: int) -> int:
    attn = getattr(layer, "self_attn", None)
    radix_attn = getattr(attn, "attn", None)
    lid = getattr(radix_attn, "layer_id", None)
    return fallback_idx if lid is None else int(lid)


def _check_unpacked_fused_pool(pool: object) -> TpuFusedMHATokenToKVPool:
    if not isinstance(pool, TpuFusedMHATokenToKVPool):
        raise TypeError(
            f"DFLASH draft KV writes require TpuFusedMHATokenToKVPool, got {type(pool)}."
        )
    # Packed pools (SGLANG_TPU_KV_PACK_HEADS) hold a rank's heads side by side in
    # one row, a layout these writers don't produce.
    if pool.kv_pack_heads > 1:
        raise NotImplementedError("DFLASH draft KV writes need an unpacked KV pool")
    return pool


def _project_layers_kv(
    draft_model: DFlashDraftModel,
    layers: list[object],
    ctx_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn0 = layers[0].self_attn
    k = ctx_hidden.new_empty(
        (
            len(layers),
            int(ctx_hidden.shape[0]),
            attn0.num_kv_heads,
            attn0.head_dim,
        )
    )
    v = torch.empty_like(k)
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        layer_ctx_hidden = draft_model.prepare_context_hidden_for_kv(layer, ctx_hidden)
        layer_k, layer_v = attn.kv_proj_only(layer_ctx_hidden)
        k[idx] = attn.apply_k_norm(layer_k).view_as(k[idx])
        v[idx] = layer_v.view_as(v[idx])
    return k, v


def dflash_write_draft_kv(
    *,
    draft_model: DFlashDraftModel,
    kv_pool: DraftKVPool,
    ctx_hidden: torch.Tensor,
    ctx_positions: torch.Tensor,
    ctx_cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
    max_num_reqs: int,
) -> None:
    """Write context K/V into the draft pool, covering extend and verify.

    Projections stay per layer (they are GEMMs), but rope and the scatter run
    once for all layers, so K/V crosses HBM a single time.

    Args:
      draft_model: DFlash draft model owning the layers and the projections.
      kv_pool: Draft TPU KV pool (FA-only, SWAKVPool, or TpuUnifiedSWAKVPool).
      ctx_hidden: Projected context hidden states [num_tokens, hidden].
      ctx_positions: Token position IDs [num_tokens].
      ctx_cache_loc: Flat destination slot(s) per token ([num_tokens] or
        [G, num_tokens]); unused on the verify path, where the spans come from
        cache_loc_2d.
      cache_loc_2d: Verify's dense slots ([bs, block_size] or
        [G, bs, block_size]), or None on extend.
      commit_lens: Verify's accepted counts [bs], or None on extend.
      max_num_reqs: Upper bound on requests in one extend batch.
    """
    # kv_fused_rope_write imports jax eagerly; keep the span math and the rope
    # gather above importable without it.
    from sglang_tpu.kv_fused_rope_write import (
        fused_rope_write_kv,
    )

    dst_starts, src_starts, span_lens, max_span_len = get_contiguous_slot_spans(
        page_size=int(kv_pool.page_size),
        cache_loc=ctx_cache_loc,
        cache_loc_2d=cache_loc_2d,
        commit_lens=commit_lens,
        max_num_reqs=max_num_reqs,
    )
    cos, sin = _rope_tables(
        draft_model=draft_model, positions=ctx_positions, dtype=ctx_hidden.dtype
    )
    all_layers = list(draft_model.layers)

    if isinstance(kv_pool, TpuUnifiedSWAKVPool):
        inner = _check_unpacked_fused_pool(kv_pool.kv_pool)
        k, v = _project_layers_kv(draft_model, all_layers, ctx_hidden)
        kv_bufs: list[torch.Tensor] = []
        buf_to_slot: dict[int, int] = {}
        l2b: list[int] = []
        l2g: list[int] = []
        for idx, layer in enumerate(all_layers):
            lid = _layer_id(layer, idx)
            _, is_swa = kv_pool.layers_mapping[lid]
            raw_buf = kv_pool._buffer_idx(lid)
            slot = buf_to_slot.get(raw_buf)
            if slot is None:
                slot = len(kv_bufs)
                buf_to_slot[raw_buf] = slot
                kv_bufs.append(inner.kv_buffer[raw_buf])
            l2b.append(slot)
            l2g.append(0 if not is_swa else 1 + kv_pool.swa_page_group(lid))
        fused_rope_write_kv(
            dst_starts=dst_starts,
            src_starts=src_starts,
            span_lens=span_lens,
            k=k,
            v=v,
            cos=cos,
            sin=sin,
            kv_bufs=kv_bufs,
            max_span_len=max_span_len,
            layer_to_buf=tuple(l2b),
            layer_to_addr_group=tuple(l2g),
        )
        return

    if isinstance(kv_pool, SWAKVPool):
        if kv_pool.swa_layer_nums == 0:
            sub = _check_unpacked_fused_pool(kv_pool.full_kv_pool)
            k, v = _project_layers_kv(draft_model, all_layers, ctx_hidden)
            fused_rope_write_kv(
                dst_starts=dst_starts if dst_starts.ndim == 1 else dst_starts[0],
                src_starts=src_starts,
                span_lens=span_lens,
                k=k,
                v=v,
                cos=cos,
                sin=sin,
                kv_bufs=sub.kv_buffer,
                max_span_len=max_span_len,
            )
            return
        if kv_pool.full_layer_nums == 0:
            sub = _check_unpacked_fused_pool(kv_pool.swa_kv_pool)
            k, v = _project_layers_kv(draft_model, all_layers, ctx_hidden)
            fused_rope_write_kv(
                dst_starts=dst_starts if dst_starts.ndim == 1 else dst_starts[-1],
                src_starts=src_starts,
                span_lens=span_lens,
                k=k,
                v=v,
                cos=cos,
                sin=sin,
                kv_bufs=sub.kv_buffer,
                max_span_len=max_span_len,
            )
            return
        full_sub = _check_unpacked_fused_pool(kv_pool.full_kv_pool)
        swa_sub = _check_unpacked_fused_pool(kv_pool.swa_kv_pool)
        if dst_starts.ndim != 2 or int(dst_starts.shape[0]) != 2:
            raise ValueError(
                "Mixed SWAKVPool draft KV write requires 2 address groups "
                f"(FA + SWA), got dst_starts shape={tuple(dst_starts.shape)}."
            )
        full_layers: list[object] = []
        full_l2b: list[int] = []
        swa_layers: list[object] = []
        swa_l2b: list[int] = []
        for idx, layer in enumerate(all_layers):
            lid = _layer_id(layer, idx)
            sub_idx, is_swa = kv_pool.layers_mapping[lid]
            if is_swa:
                swa_layers.append(layer)
                swa_l2b.append(sub_idx)
            else:
                full_layers.append(layer)
                full_l2b.append(sub_idx)
        if full_layers:
            k_full, v_full = _project_layers_kv(draft_model, full_layers, ctx_hidden)
            fused_rope_write_kv(
                dst_starts=dst_starts[0],
                src_starts=src_starts,
                span_lens=span_lens,
                k=k_full,
                v=v_full,
                cos=cos,
                sin=sin,
                kv_bufs=full_sub.kv_buffer,
                max_span_len=max_span_len,
                layer_to_buf=(
                    None
                    if full_l2b == list(range(len(full_sub.kv_buffer)))
                    else tuple(full_l2b)
                ),
            )
        if swa_layers:
            k_swa, v_swa = _project_layers_kv(draft_model, swa_layers, ctx_hidden)
            fused_rope_write_kv(
                dst_starts=dst_starts[1],
                src_starts=src_starts,
                span_lens=span_lens,
                k=k_swa,
                v=v_swa,
                cos=cos,
                sin=sin,
                kv_bufs=swa_sub.kv_buffer,
                max_span_len=max_span_len,
                layer_to_buf=(
                    None
                    if swa_l2b == list(range(len(swa_sub.kv_buffer)))
                    else tuple(swa_l2b)
                ),
            )
        return

    fused_pool = _check_unpacked_fused_pool(kv_pool)
    k, v = _project_layers_kv(draft_model, all_layers, ctx_hidden)
    fused_rope_write_kv(
        dst_starts=dst_starts,
        src_starts=src_starts,
        span_lens=span_lens,
        k=k,
        v=v,
        cos=cos,
        sin=sin,
        kv_bufs=fused_pool.kv_buffer,
        max_span_len=max_span_len,
    )


def dflash_write_draft_kv_ref(
    *,
    draft_model: DFlashDraftModel,
    kv_pool: DraftKVPool,
    ctx_hidden: torch.Tensor,
    ctx_positions: torch.Tensor,
    ctx_cache_loc: torch.Tensor,
) -> None:
    """Per-layer equivalent of dflash_write_draft_kv, for tests and benchmarks.

    Independent route: each layer runs its own apply_k_rope and its own eager
    scatter, so neither the fused rotation nor the span addressing is in the
    path. That is what makes it a reference, and why it costs num_layers
    scatters and num_layers K/V passes over HBM.

    Takes flat per-token slots only, so verify's uncommitted tokens land on slot
    0 as garbage; dflash_write_draft_kv skips them instead.

    Args:
      draft_model: DFlash draft model owning the layers and the projections.
      kv_pool: Draft TPU KV pool (FA-only, SWAKVPool, or TpuUnifiedSWAKVPool).
      ctx_hidden: Projected context hidden states [num_tokens, hidden].
      ctx_positions: Token position IDs [num_tokens].
      ctx_cache_loc: Flat destination slot(s) per token ([num_tokens] or
        [G, num_tokens]).
    """
    for layer_idx, layer in enumerate(draft_model.layers):
        attn = layer.self_attn
        layer_ctx_hidden = draft_model.prepare_context_hidden_for_kv(layer, ctx_hidden)
        k, v = attn.kv_proj_only(layer_ctx_hidden)
        k = attn.apply_k_norm(k)
        k = attn.apply_k_rope(ctx_positions, k)
        lid = _layer_id(layer, layer_idx)
        sub_pool, buf, addr_group = _resolve_layer_pool_and_group(kv_pool, lid)
        layer_loc = (
            ctx_cache_loc if ctx_cache_loc.ndim == 1 else ctx_cache_loc[addr_group]
        )
        _scatter_layer_kv(
            kv_pool=sub_pool,
            layer_idx=lid,
            loc=layer_loc,
            k=k.view(-1, attn.num_kv_heads, attn.head_dim),
            v=v.view(-1, attn.num_kv_heads, attn.head_dim),
            buf_override=buf,
        )


def _resolve_layer_pool_and_group(
    kv_pool: DraftKVPool, layer_id: int
) -> tuple[TpuFusedMHATokenToKVPool, torch.Tensor, int]:
    if isinstance(kv_pool, TpuUnifiedSWAKVPool):
        inner = _check_unpacked_fused_pool(kv_pool.kv_pool)
        _, is_swa = kv_pool.layers_mapping[layer_id]
        group = 0 if not is_swa else 1 + kv_pool.swa_page_group(layer_id)
        return inner, kv_pool.get_fused_kv_buffer(layer_id), group
    if isinstance(kv_pool, SWAKVPool):
        sub_idx, is_swa = kv_pool.layers_mapping[layer_id]
        if is_swa:
            sub = _check_unpacked_fused_pool(kv_pool.swa_kv_pool)
            group = 1 if kv_pool.full_layer_nums > 0 else 0
        else:
            sub = _check_unpacked_fused_pool(kv_pool.full_kv_pool)
            group = 0
        return sub, sub.get_fused_kv_buffer(sub_idx), group
    fused = _check_unpacked_fused_pool(kv_pool)
    return fused, fused.get_fused_kv_buffer(layer_id), 0


def _scatter_layer_kv(
    *,
    kv_pool: TpuFusedMHATokenToKVPool,
    layer_idx: int,
    loc: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    buf_override: torch.Tensor | None = None,
) -> None:
    """Interleave one layer's K/V per head and scatter into its 5D pool buffer.

    Args:
      kv_pool: Fused TPU KV pool.
      layer_idx: Layer index within the pool.
      loc: Flat slot index per token [num_tokens] (negative entries are skipped).
      k: Rotated keys [num_tokens, head_num, head_dim].
      v: Values [num_tokens, head_num, v_head_dim].
      buf_override: Optional pre-resolved 5D layer buffer.
    """
    # Packed pools (SGLANG_TPU_KV_PACK_HEADS) hold a rank's heads side by side in
    # one row, a layout these writers don't produce.
    if kv_pool.kv_pack_heads > 1:
        raise NotImplementedError("DFLASH draft KV writes need an unpacked KV pool")
    buf = (
        kv_pool.get_fused_kv_buffer(layer_idx)
        if buf_override is None
        else buf_override
    )
    num_tokens = int(loc.shape[0])
    head_dim = kv_pool.head_dim

    if kv_pool.v_head_dim < head_dim:
        v = torch.nn.functional.pad(v, (0, head_dim - kv_pool.v_head_dim))
    kv = torch.stack([k, v], dim=2).reshape(num_tokens, kv_pool.head_num * 2, head_dim)
    kv = torch.nn.functional.pad(
        kv,
        (
            0,
            kv_pool.padded_head_dim - head_dim,
            0,
            kv_pool.num_kv_heads_x2 - kv_pool.head_num * 2,
        ),
    )
    formatted = kv.view(num_tokens, *buf.shape[2:]).to(kv_pool.store_dtype)
    if bool((loc < 0).any()):
        valid = loc >= 0
        if bool(valid.any()):
            buf.flatten(0, 1).index_copy_(
                0,
                loc[valid].to(torch.long),
                formatted[valid],
            )
        return
    buf.flatten(0, 1).index_copy_(
        0,
        loc.to(torch.long),
        formatted,
    )


def get_contiguous_slot_spans(
    *,
    page_size: int,
    cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
    max_num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Decompose a draft KV write into spans of contiguous destination slots.

    Slots are consecutive only within a page -- the invariant the RPA attention
    kernel relies on too -- so every span has to stay inside one page. Both
    branches pad their span count to a static worst case; unused spans come out
    empty and the kernel skips them.

    Args:
      page_size: Draft KV pool page size.
      cache_loc: Flat destination slot(s) per token ([num_tokens] or
        [G, num_tokens]).
      cache_loc_2d: Verify's dense slots ([bs, block_size] or
        [G, bs, block_size]), or None.
      commit_lens: Verify's accepted counts [bs], or None.
      max_num_reqs: Upper bound on requests in one extend batch.

    Returns:
      Tuple of dst_starts ([num_spans] or [G, num_spans]), src_starts, span_lens
      (each [num_spans]) and the span length bound.
    """
    if cache_loc_2d is not None:
        assert commit_lens is not None
        return _verify_spans(
            page_size=page_size, cache_loc_2d=cache_loc_2d, commit_lens=commit_lens
        )
    return _extend_spans(
        page_size=page_size, cache_loc=cache_loc, max_num_reqs=max_num_reqs
    )


def _verify_spans(
    *, page_size: int, cache_loc_2d: torch.Tensor, commit_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split each request's committed prefix at its page boundary.

    A commit is at most block_size tokens and block_size fits in a page, so a
    request needs two spans: what fits in the page its prefix ends on, and the
    remainder, which starts wherever the next page landed. Uncommitted slots go
    unwritten instead of being masked onto slot 0.

    Args:
      page_size: Draft KV pool page size.
      cache_loc_2d: Verify's dense slots ([bs, block_size] or
        [G, bs, block_size]).
      commit_lens: Verify's accepted counts [bs].

    Returns:
      Tuple of dst_starts ([2 * bs] or [G, 2 * bs]), src_starts, span_lens
      (each [2 * bs]) and the span length bound.
    """
    if cache_loc_2d.ndim == 2:
        bs, block_size = (int(d) for d in cache_loc_2d.shape)
    elif cache_loc_2d.ndim == 3:
        _, bs, block_size = (int(d) for d in cache_loc_2d.shape)
    else:
        raise ValueError(
            f"DFLASH cache_loc_2d must be 2D or 3D, got shape={tuple(cache_loc_2d.shape)}."
        )
    assert block_size <= page_size, (
        f"DFLASH verify block ({block_size}) must fit in a page ({page_size}) "
        "for the two-span split to cover it"
    )
    # int32 end to end: the TPU backend fails to lower dynamic indexing over
    # int64, and slots fit comfortably.
    slots = cache_loc_2d.to(torch.int32)
    src_starts = torch.arange(
        0, bs * block_size, block_size, dtype=torch.int32, device=slots.device
    )
    lens = commit_lens.to(torch.int32)
    offsets = torch.arange(block_size, dtype=torch.int32, device=slots.device)
    if slots.ndim == 2:
        head_dst = slots[:, 0]
        head_lens = torch.minimum(
            lens, (page_size - head_dst % page_size).to(torch.int32)
        )
        tail_lens = lens - head_lens
        # Where the spill lands, selected without a gather. A commit that stays in
        # its page picks some unwritten slot, or none at all when it fills the row;
        # either way the tail length is zero.
        tail_dst = (slots * (offsets[None, :] == head_lens[:, None])).sum(
            dim=1, dtype=torch.int32
        )
        dst_starts = torch.cat([head_dst, tail_dst])
    else:
        head_dst_0 = slots[0, :, 0]
        head_lens = torch.minimum(
            lens, (page_size - head_dst_0 % page_size).to(torch.int32)
        )
        tail_lens = lens - head_lens
        head_dst = slots[:, :, 0]
        tail_mask = offsets[None, None, :] == head_lens[None, :, None]
        raw_tail = (slots * tail_mask).sum(dim=2, dtype=torch.int32)
        tail_dst = torch.where(tail_lens[None, :] > 0, raw_tail, -1)
        dst_starts = torch.cat([head_dst, tail_dst], dim=1)
    return (
        dst_starts,
        torch.cat([src_starts, src_starts + head_lens]),
        torch.cat([head_lens, tail_lens]),
        block_size,
    )


def _extend_spans(
    *, page_size: int, cache_loc: torch.Tensor, max_num_reqs: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Cut a flat extend write into runs, reading the runs off the slots.

    An extend batch concatenates requests, and each one resumes at its own page
    offset, so the run structure cannot be derived from the first slot's
    alignment. Deriving it from the slots themselves covers page boundaries,
    request boundaries, and unmapped prefix/padding zero slots alike.

    Args:
      page_size: Draft KV pool page size.
      cache_loc: Flat destination slot(s) per token ([num_tokens] or
        [G, num_tokens]).
      max_num_reqs: Upper bound on requests in one extend batch.

    Returns:
      Tuple of dst_starts ([num_spans] or [G, num_spans]), src_starts, span_lens
      (each [num_spans]) and the span length bound.
    """
    if cache_loc.ndim not in (1, 2):
        raise ValueError(
            f"DFLASH cache_loc must be 1D or 2D, got shape={tuple(cache_loc.shape)}."
        )
    num_tokens = int(cache_loc.shape[-1])
    device = cache_loc.device
    # Stay in int32 end to end: the TPU backend fails to lower a dynamic gather
    # over int64, and both slots and token counts fit comfortably.
    loc_2d = (
        cache_loc.unsqueeze(0).to(torch.int32)
        if cache_loc.ndim == 1
        else cache_loc.to(torch.int32)
    )
    loc0 = loc_2d[0]
    valid = loc0 > 0
    # Pages handed out back to back are address-adjacent, so a page start ends a
    # run even where the addresses continue -- otherwise a span could outrun
    # max_span_len.
    step_breaks = (loc0[1:] != loc0[:-1] + 1) | (loc0[1:] % page_size == 0)
    num_groups = int(loc_2d.shape[0])
    if num_groups > 1:
        lg = loc_2d[1:]
        both_neg = (lg[:, 1:] < 0) & (lg[:, :-1] < 0)
        swa_breaks = (
            ~both_neg
            & ((lg[:, 1:] != lg[:, :-1] + 1) | ((lg[:, 1:] >= 0) & (lg[:, 1:] % page_size == 0)))
        ).any(dim=0)
        step_breaks = step_breaks | swa_breaks

    is_run_start = torch.cat(
        [
            valid[:1],
            valid[1:] & (~valid[:-1] | step_breaks),
        ]
    )
    active_run_id = (
        torch.cumsum(is_run_start.to(torch.int32), dim=0, dtype=torch.int32) - 1
    )
    req_bound = min(max_num_reqs, num_tokens)
    num_spans = (
        num_tokens // page_size
        + (req_bound if num_groups == 1 else 2 * req_bound)
        + 1
    )
    span_ids = torch.arange(num_spans, device=device, dtype=torch.int32)
    token_idx = torch.arange(num_tokens, device=device, dtype=torch.int32)
    in_span = valid[None, :] & (active_run_id[None, :] == span_ids[:, None])
    span_lens = in_span.sum(dim=1, dtype=torch.int32)
    src_starts = (
        torch.where(in_span, token_idx[None, :], num_tokens).min(dim=1).values
    )
    safe_src = src_starts.clamp(max=max(num_tokens - 1, 0))
    dst_starts = (
        torch.index_select(loc0, 0, safe_src)
        if cache_loc.ndim == 1
        else torch.index_select(loc_2d, 1, safe_src)
    )
    return (
        dst_starts,
        src_starts,
        span_lens,
        page_size,
    )


def _rope_tables(
    *, draft_model: DFlashDraftModel, positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the draft rope cos/sin rows for positions.

    cos_sin_cache is [max_position, rotary_dim], cos first half and sin second
    -- the only layout the fused write kernel rotates.

    Args:
      draft_model: DFlash draft model whose rope tables are gathered.
      positions: Token position IDs [num_tokens].
      dtype: Dtype of the keys the tables multiply.

    Returns:
      Tuple of cos and sin, each [num_tokens, rotary_dim // 2].
    """
    attn = draft_model.layers[0].self_attn
    rotary = attn.rotary_emb
    # mrope indexes by 3D positions; the table variants widen cos_sin_cache past
    # rotary_dim. Both would gather silently wrong rows here.
    assert (
        not isinstance(rotary, MRotaryEmbedding)
        and rotary.is_neox_style
        and rotary.rotary_dim == attn.head_dim
        and int(rotary.cos_sin_cache.shape[-1]) == attn.head_dim
    ), f"DFLASH draft rope must be neox over head_dim, got {type(rotary).__name__}"
    rows = rotary.cos_sin_cache.index_select(0, positions.to(torch.int64)).to(dtype)
    cos, sin = rows.chunk(2, dim=-1)
    return cos, sin
