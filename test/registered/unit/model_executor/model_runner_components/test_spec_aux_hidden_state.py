import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStatePacker,
    pack_aux_hidden_states,
)
from sglang.srt.model_executor.model_runner_components.misc_utils import (
    resolve_aux_hidden_states_width,
)
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    SpecAuxHiddenStateConfig,
    _map_muse_target_layer_ids,
    _populate_draft_kv_pricing,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("target_model_type", "draft_architecture", "expected"),
    [
        ("muse_glimmer", "MuseGlimmerAssistantModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlash2DraftModel", [2, 14, 26, 38, 50]),
        ("muse_glimmer", "DFlashDraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "DFlash2DraftModel", [1, 13, 25, 37, 49]),
        ("qwen3", "MuseGlimmerAssistantModel", [1, 13, 25, 37, 49]),
    ],
)
def test_muse_target_layer_id_mapping(target_model_type, draft_architecture, expected):
    """The +1 belongs to Muse targets, which report layer outputs where the rest
    report layer inputs. The draft architecture alone does not earn it."""
    assert (
        _map_muse_target_layer_ids(
            target_hf_config=SimpleNamespace(model_type=target_model_type),
            draft_hf_config=SimpleNamespace(architectures=[draft_architecture]),
            layer_ids=[1, 13, 25, 37, 49],
        )
        == expected
    )


def test_packer_writes_into_graph_buffer_like_list_path():
    """A graph runner hands every graph size a row slice of one buffer; the
    packed output must equal the list path's concat and live in that storage,
    and a buffer that does not fit the captures must be rejected."""
    captures = [torch.randn(5, 4) for _ in range(3)]
    shared = torch.full((8, 12), float("nan"))
    packer = AuxHiddenStatePacker.for_batch(
        SimpleNamespace(aux_hidden_states_buffer=shared[:5]), 3
    )
    for hidden in captures:
        packer.append(hidden)
    packed = packer.finalize()

    torch.testing.assert_close(packed, pack_aux_hidden_states(list(captures)))
    assert packed.data_ptr() == shared.data_ptr()
    assert shared[5:].isnan().all()

    too_small = SimpleNamespace(aux_hidden_states_buffer=shared[:4])
    with pytest.raises(ValueError):
        AuxHiddenStatePacker.for_batch(too_small, 3).append(captures[0])


class _SharedAuxModel:
    def get_aux_hidden_states_width(self) -> int:
        return 12


@pytest.mark.parametrize(
    ("model", "dflash", "is_draft_worker", "expected"),
    [
        (_SharedAuxModel(), True, False, 12),
        # Only the DFlash family is known to consume them within the step.
        (_SharedAuxModel(), False, False, 0),
        (_SharedAuxModel(), True, True, 0),
        (object(), True, False, 0),
    ],
)
def test_only_dflash_target_models_share_aux_output(
    model, dflash, is_draft_worker, expected
):
    assert (
        resolve_aux_hidden_states_width(
            model=model,
            spec_algorithm=SimpleNamespace(is_dflash_family=lambda: dflash),
            is_draft_worker=is_draft_worker,
        )
        == expected
    )


def test_populate_draft_kv_pricing_heterogeneous_tpu_geometries():
    """Draft KV pricing resolves exact TPU 128-lane padded per-layer bytes,
    respects attn_tp_size under DP attention, and flags mxint8 blocked fallback."""
    from unittest.mock import patch

    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs(
        model_path="dummy",
        attention_backend="pallas_tpu_rpa_v3",
        kv_cache_dtype="fp8_e4m3",
        speculative_draft_kv_cache_dtype="bf16",
        page_size=128,
        tp_size=8,
    )
    # 1) Mixed homogeneous draft under DP attention (tp=8, attn_tp_size=2):
    # total_kv_heads=2 -> 1 per attn_tp rank; head_dim=64 pads to 128 on TPU:
    # kv_bytes_per_token_per_layer = (2 * 1) * 128 * 2 = 512 bytes (vs unpadded 256).
    draft_mc_homo = SimpleNamespace(
        is_hybrid_swa=True,
        is_deepseek_v4_arch=False,
        swa_attention_layer_ids=[0, 2, 3],
        dtype=torch.bfloat16,
        head_dim=64,
        v_head_dim=64,
        swa_head_dim=64,
        swa_v_head_dim=64,
        get_num_kv_heads=lambda tp: 2 // tp,
        get_swa_num_kv_heads=lambda tp: 2 // tp,
    )
    cfg = SpecAuxHiddenStateConfig()
    with (
        patch(
            "sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state.is_tpu",
            return_value=True,
        ),
        get_context().override_server_args(
            attention_backend="pallas_tpu_rpa_v3",
            kv_cache_dtype="fp8_e4m3",
            speculative_draft_kv_cache_dtype="bf16",
            speculative_draft_attention_backend=None,
            page_size=128,
            tp_size=8,
        ),
        get_parallel().override(
            attn_tp_size=2,
            attn_dp_size=4,
            attn_cp_size=1,
            tp_size=8,
            moe_tp_size=8,
            moe_ep_size=1,
            moe_dp_size=1,
        ),
    ):
        _populate_draft_kv_pricing(
            config=cfg,
            server_args=args,
            draft_model_config=draft_mc_homo,
            draft_num_layers=4,
            is_dflash=True,
            is_mtp_in_model=False,
        )
    assert cfg.draft_full_num_layers == 1
    assert cfg.draft_swa_num_layers == 3
    assert cfg.draft_swa_uses_full_capacity is False
    assert cfg.draft_kv_heads_homogeneous is True
    assert cfg.draft_kv_cache_dtype_str == "bf16"
    assert cfg.draft_full_bytes_per_token == 512
    assert cfg.draft_swa_bytes_per_token == 512

    # 2) Inhomogeneous mixed draft (H_FA=4, H_SWA=1 at attn_tp=1, head_dim=128) + MTP flag:
    draft_mc_inhomo = SimpleNamespace(
        is_hybrid_swa=True,
        is_deepseek_v4_arch=False,
        swa_attention_layer_ids=[0, 1],
        dtype=torch.bfloat16,
        head_dim=128,
        v_head_dim=128,
        swa_head_dim=128,
        swa_v_head_dim=128,
        get_num_kv_heads=lambda tp: 4 // tp,
        get_swa_num_kv_heads=lambda tp: 1 // tp,
    )
    cfg_inhomo = SpecAuxHiddenStateConfig()
    with (
        patch(
            "sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state.is_tpu",
            return_value=True,
        ),
        get_context().override_server_args(
            attention_backend="pallas_tpu_rpa_v3",
            kv_cache_dtype="bf16",
            speculative_draft_kv_cache_dtype=None,
            speculative_draft_attention_backend=None,
            page_size=128,
            tp_size=1,
        ),
        get_parallel().override(attn_tp_size=1, tp_size=1),
    ):
        _populate_draft_kv_pricing(
            config=cfg_inhomo,
            server_args=args,
            draft_model_config=draft_mc_inhomo,
            draft_num_layers=3,
            is_dflash=False,
            is_mtp_in_model=True,
        )
    assert cfg_inhomo.draft_full_num_layers == 1
    assert cfg_inhomo.draft_swa_num_layers == 2
    assert cfg_inhomo.draft_swa_uses_full_capacity is True
    assert cfg_inhomo.draft_kv_heads_homogeneous is False
    assert cfg_inhomo.draft_full_bytes_per_token == 2 * 4 * 128 * 2
    assert cfg_inhomo.draft_swa_bytes_per_token == 2 * 1 * 128 * 2

    # 3) Mixed mxint8 draft with unsupported blocked page_size=1024 falls back to Case 2c (merged):
    mx_args = ServerArgs(
        model_path="dummy",
        attention_backend="oot_mxint8_rpa",
        kv_cache_dtype="mxint8",
        page_size=1024,
    )
    cfg_mx = SpecAuxHiddenStateConfig()
    with (
        patch(
            "sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state.is_tpu",
            return_value=True,
        ),
        get_context().override_server_args(
            attention_backend="oot_mxint8_rpa",
            kv_cache_dtype="mxint8",
            speculative_draft_kv_cache_dtype="mxint8",
            speculative_draft_attention_backend="oot_mxint8_rpa",
            page_size=1024,
            tp_size=1,
        ),
        get_parallel().override(attn_tp_size=1, tp_size=1),
    ):
        _populate_draft_kv_pricing(
            config=cfg_mx,
            server_args=mx_args,
            draft_model_config=draft_mc_homo,
            draft_num_layers=4,
            is_dflash=False,
            is_mtp_in_model=False,
        )
    assert cfg_mx.draft_kv_heads_homogeneous is False
    assert cfg_mx.draft_full_bytes_per_token is not None
    assert cfg_mx.draft_swa_bytes_per_token == cfg_mx.draft_full_bytes_per_token


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
