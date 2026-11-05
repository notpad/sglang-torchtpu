"""CPU unit tests for TPU unified KV pool P/D disaggregation via Raiden."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import (
    PER_LAYER_KV_LAYOUT,
    UNIFIED_SWA_KV_LAYOUT,
    ReshardBlocks,
    ReshardChunk,
)
from sglang_tpu.unified_kv_pool import AttnLayerSpec, TpuUnifiedSWAKVPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "_disagg_raiden_transfer_harness",
    pathlib.Path(__file__).resolve().parents[3]
    / "tpu"
    / "unit"
    / "test_disagg_raiden_transfer.py",
)
_harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _harness
_spec.loader.exec_module(_harness)
RaidenRendezvousTestCase = _harness.RaidenRendezvousTestCase
_fake_tpu_sync = _harness._fake_tpu_sync
_FakeRaidenEngine = _harness._FakeRaidenEngine


def _pool(swa_layers=(1, 2, 4, 5, 6), *, head_num=2, dtype=torch.bfloat16):
    return TpuUnifiedSWAKVPool(
        size=64,
        page_size=4,
        dtype=dtype,
        full=AttnLayerSpec(head_num=head_num, head_dim=128, layer_ids=(0, 3)),
        swa=AttnLayerSpec(head_num=head_num, head_dim=128, layer_ids=swa_layers),
        device="cpu",
    )


def _fake_pb2_modules(engine_cls=None) -> dict[str, types.ModuleType]:
    class _Entries(list):
        def add(self):
            e = SimpleNamespace()
            self.append(e)
            return e

    pb2 = types.ModuleType("tpu_sync.rpc.raiden_service_pb2")
    pb2.MEMORY_TYPE_DRAM, pb2.MEMORY_TYPE_HBM = 1, 2
    pb2.StartTransferRequest = lambda: SimpleNamespace(
        shard_push_schedules=defaultdict(lambda: SimpleNamespace(entries=_Entries()))
    )
    mods = _fake_tpu_sync(engine_cls or _FakeRaidenEngine)
    mods["tpu_sync"].rpc = rpc = types.ModuleType("tpu_sync.rpc")
    rpc.raiden_service_pb2 = pb2
    mods["tpu_sync.rpc"] = rpc
    mods["tpu_sync.rpc.raiden_service_pb2"] = pb2
    return mods


class TestUnifiedRaidenPD(RaidenRendezvousTestCase):
    def _unified(self, mgr, pool: TpuUnifiedSWAKVPool, *, chunked: bool = False):
        mgr._unified_swa_pool, mgr.kv_layout = pool, UNIFIED_SWA_KV_LAYOUT
        mgr._kv_caches = pool.kv_pool.page_buffers()
        mgr._deferred_ack_targets, mgr._staging_outstanding = {}, {}
        mgr._deferred_ack_poisoned_rooms, mgr.deferred_bootstrap = set(), None
        mgr.chunked_transfer = chunked
        mgr._listener_start_read = False
        mgr._engine.transfer_address = mgr.raiden_endpoint
        mgr._engine.plan_host_blocks = lambda uuid, blks: list(range(len(blks)))
        mgr._engine.d2h = lambda *_: SimpleNamespace(
            is_ready=lambda: True, ok=lambda: True, error_message=str
        )
        if hasattr(mgr, "_push_executor"):
            mgr._push_executor.shutdown(wait=True)
            mgr._push_executor = SimpleNamespace(
                submit=lambda fn, *a, **k: fn(*a, **k), shutdown=lambda **_: None
            )
        return mgr

    def _pump(self, *mgrs):
        progressed = True
        while progressed:
            progressed = False
            for m in mgrs:
                frames = self.inboxes[m.endpoint].frames
                while frames:
                    m._dispatch(frames.pop(0))
                    progressed = True

    def test_single_engine_sizing_metrics_and_validation(self):
        for dtype, swa_ids in (
            (torch.bfloat16, (1, 2, 4, 5, 6)),
            (torch.bfloat16, (1, 2)),
            (torch.float8_e4m3fn, (1, 2, 4, 5, 6)),
        ):
            with self.subTest(dtype=dtype, swa_ids=swa_ids):
                pool = _pool(swa_ids, dtype=dtype)
                m = self._prefill()
                m._engine = None
                m.kv_args.state_types = [StateType.SWA]
                m.kv_args.kv_item_lens = pool.kv_pool.get_contiguous_buf_infos()[2]
                m.kv_item_lens_sum = sum(m.kv_args.kv_item_lens)
                m._max_request_blocks, m._swa_max_request_blocks = lambda: 8, lambda: 2
                m._num_slots = lambda *a, **k: 6
                built = []
                groups = get_raiden_kv_caches(pool)
                self.assertEqual(groups.swa, [])
                with patch.dict(
                    sys.modules, _fake_tpu_sync(lambda **kw: built.append(kw) or kw)
                ):
                    m.set_kv_caches(groups)
                m_swa = pool.layout.num_swa_groups
                self.assertEqual(len(built), 1)
                self.assertIsNone(m._swa_engine)
                self.assertEqual(m.kv_layout, UNIFIED_SWA_KV_LAYOUT)
                self.assertEqual(built[0]["max_blocks"], 8 + 2 * m_swa)
                self.assertEqual(m.state_item_lens_sum, m_swa * m.kv_item_lens_sum)

                sender = object.__new__(RaidenKVSender)
                sender.kv_mgr = m
                sender._transfer_metric = KVTransferMetric()
                sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
                m.get_kv_replica_factor = lambda: 1
                sender._record_transfer_indices(np.array([1, 2, 3]), [np.array([4, 5])])
                self.assertEqual(
                    sender.get_transfer_metric().transfer_total_bytes,
                    (3 + 2 * m_swa) * m.kv_item_lens_sum,
                )

        pool = _pool()
        m = self._prefill()
        m.kv_args.state_types = [StateType.SWA]
        m._pipeline_enabled = True
        with self.assertRaisesRegex(NotImplementedError, "enable-unified-memory"):
            m.set_kv_caches(get_raiden_kv_caches(pool))
        m._unified_swa_pool = pool
        for bad in (None, [], [None]):
            with self.assertRaisesRegex(RuntimeError, "missing its page indices"):
                m._swa_blocks_from_state_indices(bad)

    def test_pull_block_pairing_prefix_hits_and_layout_mismatch(self):
        pool = _pool()  # M_swa = 3
        p_swa, d_swa = pool.expand_swa_pages([5, 6, 7]), pool.expand_swa_pages([8, 9])

        # 1) Chunked pull (normal 2+2 Full + SWA, and cover_all_full where missing == [])
        for cover_all in (False, True):
            with self.subTest(chunked=True, cover_all=cover_all):
                pre = self._unified(self._prefill(), pool, chunked=True)
                dec = self._unified(self._decode(), pool, chunked=True)
                room, n_full = 8100 + int(cover_all), 2 if cover_all else 4
                dec.register_decode_request(
                    room,
                    list(range(20, 20 + n_full)),
                    d_swa,
                    None,
                    self._bootstrap_info(pre),
                )
                self._pump(pre, dec)
                pre.append_send_blocks(
                    room, [1, 2], swa_block_ids=p_swa[-6:] if cover_all else None
                )
                pre._announce_chunk(room, 0, [1, 2], total_pages=n_full)
                if not cover_all:
                    pre.append_send_blocks(room, [3, 4], swa_block_ids=p_swa[-6:])
                pre.finalize_send(room, aux_index=None)
                self._pump(pre, dec)

                self.assertTrue(dec.maybe_start_pull(room))
                dec._engine._stats = ([], [f"{room}#0"], [])
                dec._drain_engine()
                self.assertTrue(dec.maybe_start_pull(room))
                dec._engine._stats = ([], [str(room)], [])
                pre._engine._stats = ([str(room)], [], [])
                dec._drain_engine()
                pre._drain_engine()
                self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
                expected_tail = p_swa[-6:] if cover_all else [3, 4] + p_swa[-6:]
                expected_local = d_swa if cover_all else [22, 23] + d_swa
                self.assertEqual(
                    [(a[0], a[3], a[4]) for a, _ in dec._engine.started],
                    [
                        (f"{room}#0", [1, 2], [20, 21]),
                        (str(room), expected_tail, expected_local),
                    ],
                )

        # 2) Whole pull partial prefix hit (4->2 Full tail, 9->6 SWA tail), 100% hit, SWA-only
        for d_full, d_s, exp_r in (
            ([20, 21], d_swa, [([3, 4] + p_swa[-6:], [20, 21] + d_swa)]),
            ([], d_swa, [(p_swa[-6:], d_swa)]),
            ([], [], []),
        ):
            with self.subTest(d_full=d_full, swa=bool(d_s)):
                pre = self._unified(self._prefill(), pool, chunked=False)
                dec = self._unified(self._decode(), pool, chunked=False)
                room = 8110 + len(d_full) + len(d_s)
                dec.register_decode_request(
                    room, d_full, d_s, None, self._bootstrap_info(pre)
                )
                self._pump(pre, dec)
                pre.append_send_blocks(room, [1, 2, 3, 4], swa_block_ids=p_swa)
                pre.finalize_send(room, aux_index=None)
                self._pump(pre, dec)
                self.assertTrue(dec.maybe_start_pull(room))
                if exp_r:
                    dec._engine._stats = ([], [str(room)], [])
                    pre._engine._stats = ([str(room)], [], [])
                    dec._drain_engine()
                    pre._drain_engine()
                self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
                self.assertEqual([(a[3], a[4]) for a, _ in dec._engine.started], exp_r)

        # 3) Layout mismatch on CHUNK and BLOCKS
        for via_chunk in (False, True):
            pre = self._unified(self._prefill(), pool, chunked=via_chunk)
            dec = self._unified(self._decode(), pool, chunked=via_chunk)
            pre.kv_layout = PER_LAYER_KV_LAYOUT
            room = 8130 + int(via_chunk)
            dec.request_status[room] = KVPoll.WaitingForInput
            dec.register_decode_request(
                room, [1, 2], d_swa, None, self._bootstrap_info(pre)
            )
            self._pump(pre, dec)
            pre.append_send_blocks(room, [1, 2], swa_block_ids=p_swa[-6:])
            if via_chunk:
                pre._announce_chunk(room, 0, [1], total_pages=2)
            else:
                pre.finalize_send(room, aux_index=None)
            self._pump(pre, dec)
            self.assertEqual(dec.request_status[room], KVPoll.Failed)

    def test_reshard_push_block_pairing_swa_only_and_layout_mismatch(self):
        p_pool, d_pool = _pool(head_num=1), _pool(head_num=2)
        p_swa, d_swa = p_pool.expand_swa_pages([5, 6]), d_pool.expand_swa_pages([8, 9])

        def _trio(chunked: bool):
            ps = [
                self._unified(
                    self._prefill(attn_tp_rank=r, attn_tp_size=4),
                    p_pool,
                    chunked=chunked,
                )
                for r in (0, 1)
            ]
            dec = self._unified(
                self._decode(attn_tp_rank=0, attn_tp_size=2), d_pool, chunked=chunked
            )
            return ps, dec

        with patch.dict(sys.modules, _fake_pb2_modules()):
            for idx, (chunked, cover_all, swa_only) in enumerate(
                (
                    (True, False, False),
                    (True, True, False),
                    (False, False, False),
                    (False, False, True),  # regression: decode_blocks == []
                    (True, False, True),
                )
            ):
                with self.subTest(
                    chunked=chunked, cover_all=cover_all, swa_only=swa_only
                ):
                    ps, dec = _trio(chunked)
                    room = 8200 + idx
                    dec.register_decode_request(
                        room,
                        [] if swa_only else [20, 21],
                        d_swa,
                        None,
                        [
                            {"rank_ip": "127.0.0.1", "rank_port": p.rank_port}
                            for p in ps
                        ],
                    )
                    self._pump(*ps, dec)
                    for p in ps:
                        if chunked and not swa_only:
                            nc = 2 if cover_all else 1
                            p.append_send_blocks(
                                room,
                                [1, 2][:nc],
                                swa_block_ids=p_swa if cover_all else None,
                            )
                            p._announce_reshard_chunk(
                                room, 0, [1, 2][:nc], total_pages=2
                            )
                    self._pump(*ps, dec)
                    for p in ps:
                        if chunked and not swa_only and not cover_all:
                            p.append_send_blocks(room, [2], swa_block_ids=p_swa)
                        elif not (chunked and not swa_only):
                            p.append_send_blocks(room, [1, 2], swa_block_ids=p_swa)
                        p.finalize_send(room, aux_index=None)
                    self._pump(*ps, dec)

                    # Verify sender pushed exact paired Full + SWA blocks on layer 0
                    l0_pushes = [
                        (a[2], a[3]) for a, _ in ps[0]._engine.pushes if a[4] == 0
                    ]
                    if swa_only:
                        self.assertEqual(l0_pushes, [(p_swa, d_swa)])
                    elif chunked and cover_all:
                        self.assertEqual(
                            l0_pushes, [([1, 2], [20, 21]), (p_swa, d_swa)]
                        )
                    elif chunked:
                        self.assertEqual(
                            l0_pushes, [([1], [20]), ([2] + p_swa, [21] + d_swa)]
                        )
                    else:
                        self.assertEqual(
                            l0_pushes, [([1, 2] + p_swa, [20, 21] + d_swa)]
                        )

            for via_chunk in (False, True):
                ps, dec = _trio(via_chunk)
                room = 8250 + int(via_chunk)
                dec.register_decode_request(
                    room,
                    [20],
                    d_swa,
                    None,
                    [{"rank_ip": "127.0.0.1", "rank_port": p.rank_port} for p in ps],
                )
                self._pump(*ps, dec)
                bad = (
                    ReshardChunk(room, 0, 0, 1, [1], 0, kv_layout=PER_LAYER_KV_LAYOUT)
                    if via_chunk
                    else ReshardBlocks(
                        room, [1], 0, b"", p_swa, kv_layout=PER_LAYER_KV_LAYOUT
                    )
                )
                dec._dispatch(bad.to_frames())
                self.assertEqual(dec.request_status[room], KVPoll.Failed)

            for via_chunk in (False, True):
                with self.subTest(seq_along_lane_reshard=True, via_chunk=via_chunk):
                    ps, dec = _trio(via_chunk)
                    dec._seq_along_lane = True
                    room = 8260 + int(via_chunk)
                    dec.register_decode_request(
                        room,
                        [20],
                        d_swa,
                        None,
                        [
                            {"rank_ip": "127.0.0.1", "rank_port": p.rank_port}
                            for p in ps
                        ],
                    )
                    self._pump(*ps, dec)
                    for p in ps:
                        p.append_send_blocks(room, [1], swa_block_ids=p_swa)
                        if via_chunk:
                            p._announce_reshard_chunk(room, 0, [1], total_pages=1)
                        p.finalize_send(room, aux_index=None)
                    self._pump(*ps, dec)
                    self.assertTrue(dec.is_failed(room))
                    self.assertIn("5D token-major", dec.failure_records[room])
                    self.assertEqual(ps[0]._engine.pushes, [])

    def _wire_spec_draft(
        self,
        mgr,
        target_pool: TpuUnifiedSWAKVPool,
        draft_pool,
        *,
        chunked: bool = False,
        draft_total_heads: int | None = None,
        draft_swa_total_heads: int | None = None,
        target_total_heads: int | None = None,
    ):
        self._unified(mgr, target_pool, chunked=chunked)
        groups = get_raiden_kv_caches(
            target_pool,
            draft_pool,
            draft_total_kv_head_num=draft_total_heads,
            draft_swa_total_kv_head_num=draft_swa_total_heads,
            target_total_kv_head_num=target_total_heads,
        )
        mgr._draft_kv_caches = tuple(groups.draft_full)
        mgr._draft_swa_kv_caches = tuple(groups.draft_swa)
        mgr._draft_unified_swa_pool = groups.draft_unified_swa_pool
        mgr._draft_seq_along_lane = groups.draft_seq_along_lane
        mgr._draft_kv_head_num = groups.draft_kv_head_num
        mgr._draft_swa_kv_head_num = groups.draft_swa_kv_head_num
        mgr._draft_total_kv_head_num = draft_total_heads
        mgr._draft_swa_total_kv_head_num = (
            draft_swa_total_heads
            if draft_swa_total_heads is not None
            else draft_total_heads
        )
        if mgr._draft_kv_caches:
            mgr._draft_engine = _FakeRaidenEngine()
            mgr.draft_raiden_endpoint = f"127.0.0.1:{mgr._control_port + 6}"
            mgr._draft_engine.transfer_address = mgr.draft_raiden_endpoint
            mgr._draft_engine.plan_host_blocks = lambda uuid, blks: list(
                range(len(blks))
            )
            mgr._draft_engine.d2h = lambda *_: SimpleNamespace(
                is_ready=lambda: True, ok=lambda: True, error_message=str
            )
            mgr._draft_engine_max_blocks = 32
        if mgr._draft_swa_kv_caches:
            mgr._draft_swa_engine = _FakeRaidenEngine()
            mgr.draft_swa_raiden_endpoint = f"127.0.0.1:{mgr._control_port + 8}"
            mgr._draft_swa_engine.transfer_address = mgr.draft_swa_raiden_endpoint
            mgr._draft_swa_engine.plan_host_blocks = lambda uuid, blks: list(
                range(len(blks))
            )
            mgr._draft_swa_engine.d2h = lambda *_: SimpleNamespace(
                is_ready=lambda: True, ok=lambda: True, error_message=str
            )
            mgr._draft_swa_engine_max_blocks = 32
        if hasattr(mgr, "_pulling"):
            from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission

            mgr._pulling = StagingAdmission(
                mgr._pulling.capacity,
                mgr._pulling.swa_capacity,
                mgr._pulling.state_capacity,
                256 if mgr._draft_kv_caches else None,
                256 if mgr._draft_swa_kv_caches else None,
            )
        return mgr

    def test_heterogeneous_spec_draft_cache_groups_and_engine_wiring(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        target_pool = _pool()  # L_FA=2, L_SWA=5 -> G=2 buffers, M_T=3
        draft_2a = TpuFusedMHATokenToKVPool(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
        )
        draft_mtp = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        draft_mtp._swa_uses_full_indices = True
        draft_2b = TpuUnifiedSWAKVPool(
            size=80,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=(0,)),
            swa=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=(1, 2, 3, 4)),
            device="cpu",
        )  # G=1, M_D=4 > M_T=3
        draft_2c_swa = SWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[0, 1],
            full_attention_layer_ids=[],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        draft_2c_mixed = SWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=64,
            swa_attention_layer_ids=[1, 2],
            full_attention_layer_ids=[0],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
            swa_kv_pool_kwargs=dict(
                page_size=4,
                enable_memory_saver=False,
                device="cpu",
                head_num=1,
                head_dim=64,
            ),
        )

        cases = (
            ("2a", draft_2a, 3, 0, False, [0, 6], 8, 0),
            ("2a_mtp", draft_mtp, 1, 0, False, [0, 6], 8, 0),
            ("2b", draft_2b, 1, 0, True, [0, 6], 8 + 2 * 4, 0),
            ("2c_swa", draft_2c_swa, 0, 2, False, [0, 8], 0, 2),
            ("2c_mixed", draft_2c_mixed, 1, 2, False, [0, 6, 8], 8, 2),
        )
        for (
            label,
            d_pool,
            exp_df,
            exp_ds,
            is_u_d,
            exp_offsets,
            exp_d_max,
            exp_ds_max,
        ) in cases:
            with self.subTest(case=label):
                groups = get_raiden_kv_caches(target_pool, d_pool)
                self.assertEqual(len(groups.full), 2)
                self.assertEqual(len(groups.draft_full), exp_df)
                self.assertEqual(len(groups.draft_swa), exp_ds)
                self.assertEqual(groups.draft_unified_swa_pool is not None, is_u_d)

                m = self._prefill()
                m._engine = None
                m.kv_args.state_types = [StateType.SWA]
                m.kv_args.kv_item_lens = target_pool.kv_pool.get_contiguous_buf_infos()[
                    2
                ]
                m.kv_item_lens_sum = sum(m.kv_args.kv_item_lens)
                m._max_request_blocks, m._swa_max_request_blocks = lambda: 8, lambda: 2
                m._num_slots = lambda *a, **k: 6
                built = []
                with patch.dict(
                    sys.modules, _fake_tpu_sync(lambda **kw: built.append(kw) or kw)
                ):
                    m.set_kv_caches(groups)
                self.assertEqual(
                    [kw["local_control_port"] - m._control_port for kw in built],
                    exp_offsets,
                )
                self.assertEqual(m._draft_engine_max_blocks, exp_d_max)
                self.assertEqual(m._draft_swa_engine_max_blocks, exp_ds_max)

    def test_heterogeneous_spec_draft_pull_pairing_across_topologies(self):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool
        from sglang_tpu.unified_kv_pool import (
            TpuUnifiedSWATokenToKVPoolAllocator,
            draft_case2c_swa_num_pages,
        )

        class _MXINT8TargetSubPool(TpuFusedMHATokenToKVPool):
            def kv_bytes_per_token_per_layer(self, **_):
                return 72 * 128 // self.page_size

            def _create_buffers(self):
                self.kv_buffer = [
                    torch.zeros((self.num_blocks, 72, 128), dtype=torch.int8)
                    for _ in range(self.layer_num)
                ]

        # 1) Case 2a with mixed dtype: Target MXINT8 (3D int8) + Draft bf16 (5D)
        mx_target = TpuUnifiedSWAKVPool(
            size=64,
            page_size=4,
            dtype=torch.int8,
            full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=(0, 3)),
            swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=(1, 2, 4, 5)),
            device="cpu",
            token_to_kv_pool_class=_MXINT8TargetSubPool,
        )  # M_T = 2
        draft_2a = TpuFusedMHATokenToKVPool(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        p_swa_2a = mx_target.expand_swa_pages([5, 6])
        d_swa_2a = mx_target.expand_swa_pages([8, 9])

        for chunked, cover_all in ((True, False), (True, True), (False, False)):
            with self.subTest(case="2a_mxint8_bf16", chunked=chunked, cover=cover_all):
                pre = self._wire_spec_draft(
                    self._prefill(), mx_target, draft_2a, chunked=chunked
                )
                dec = self._wire_spec_draft(
                    self._decode(), mx_target, draft_2a, chunked=chunked
                )
                room, n_full = 9100 + int(chunked) * 2 + int(cover_all), (
                    2 if cover_all else 4
                )
                dec.register_decode_request(
                    room,
                    list(range(20, 20 + n_full)),
                    d_swa_2a,
                    None,
                    self._bootstrap_info(pre),
                )
                self._pump(pre, dec)
                if chunked:
                    pre.append_send_blocks(
                        room, [1, 2], swa_block_ids=p_swa_2a if cover_all else None
                    )
                    pre._announce_chunk(room, 0, [1, 2], total_pages=n_full)
                    if not cover_all:
                        pre.append_send_blocks(room, [3, 4], swa_block_ids=p_swa_2a)
                else:
                    pre.append_send_blocks(room, [1, 2, 3, 4], swa_block_ids=p_swa_2a)
                pre.finalize_send(room, aux_index=None)
                self._pump(pre, dec)

                self.assertTrue(dec.maybe_start_pull(room))
                if chunked:
                    dec._engine._stats = ([], [f"{room}#0"], [])
                    dec._draft_engine._stats = ([], [f"draft:{room}#0"], [])
                    dec._drain_engine()
                    self.assertTrue(dec.maybe_start_pull(room))
                dec._engine._stats = ([], [str(room)], [])
                pre._engine._stats = ([str(room)], [], [])
                dec._draft_engine._stats = ([], [f"draft:{room}"], [])
                pre._draft_engine._stats = ([f"draft:{room}"], [], [])
                dec._drain_engine()
                pre._drain_engine()
                self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
                draft_pulls = [
                    (a[0], a[3], a[4]) for a, _ in dec._draft_engine.started
                ]
                if chunked and cover_all:
                    self.assertEqual(
                        draft_pulls,
                        [
                            (f"draft:{room}#0", [1, 2], [20, 21]),
                            (f"draft:{room}", [2], [21]),
                        ],
                    )
                elif chunked:
                    self.assertEqual(
                        draft_pulls,
                        [
                            (f"draft:{room}#0", [1, 2], [20, 21]),
                            (f"draft:{room}", [3, 4], [22, 23]),
                        ],
                    )
                else:
                    self.assertEqual(
                        draft_pulls,
                        [(f"draft:{room}", [1, 2, 3, 4], [20, 21, 22, 23])],
                    )

        # 2) Case 2b (M_D=3 > M_T=2 unified draft) & Case 2c (mixed SWAKVPool draft with compact SWA IDs)
        target_pool = _pool(swa_layers=(1, 2, 4, 5))  # M_T = 2
        req_pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        p_alloc_2b = TpuUnifiedSWATokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=target_pool,
            need_sort=False,
            req_to_token_pool=req_pool,
        )
        draft_2b = TpuUnifiedSWAKVPool(
            size=80,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=(0,)),
            swa=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=(1, 2, 3)),
            device="cpu",
        )  # M_D = 3
        p_alloc_2b.attach_draft_kv_pool(draft_2b)
        swa_toks = p_alloc_2b.swa_attn_allocator.alloc(8)
        base_pages = (swa_toks[::4] // 4).tolist()
        p_swa_2b = target_pool.expand_swa_pages(base_pages)
        p_d_swa_2b = draft_2b.expand_swa_pages(base_pages)

        pre_2b = self._wire_spec_draft(
            self._prefill(), target_pool, draft_2b, chunked=False
        )
        dec_2b = self._wire_spec_draft(
            self._decode(), target_pool, draft_2b, chunked=False
        )
        dec_2b.register_decode_request(
            9120,
            [20, 21],
            p_swa_2b,
            None,
            self._bootstrap_info(pre_2b),
            local_draft_swa_blocks=p_d_swa_2b,
        )
        self._pump(pre_2b, dec_2b)
        pre_2b.append_send_blocks(
            9120, [1, 2], swa_block_ids=p_swa_2b, draft_swa_block_ids=p_d_swa_2b
        )
        pre_2b.finalize_send(9120, aux_index=None)
        self._pump(pre_2b, dec_2b)
        self.assertTrue(dec_2b.maybe_start_pull(9120))
        dec_2b._engine._stats = ([], ["9120"], [])
        dec_2b._draft_engine._stats = ([], ["draft:9120"], [])
        pre_2b._engine._stats = (["9120"], [], [])
        pre_2b._draft_engine._stats = (["draft:9120"], [], [])
        dec_2b._drain_engine()
        pre_2b._drain_engine()
        self.assertTrue(pre_2b.is_send_done(9120) and dec_2b.is_recv_done(9120))
        self.assertEqual(
            [(a[3], a[4]) for a, _ in dec_2b._draft_engine.started],
            [([1, 2] + p_d_swa_2b, [20, 21] + p_d_swa_2b)],
        )

        # Case 2c mixed SWAKVPool draft: verify compact SWA IDs travel on _draft_swa_engine!
        t_2c = _pool(swa_layers=(1, 2, 4, 5))
        alloc_2c = TpuUnifiedSWATokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=t_2c,
            need_sort=False,
            req_to_token_pool=req_pool,
        )
        n_c = draft_case2c_swa_num_pages(16, 2)
        draft_2c = SWAKVPool(
            size=64,
            size_swa=n_c * 4,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            swa_attention_layer_ids=[1, 2],
            full_attention_layer_ids=[0],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
            swa_kv_pool_kwargs=dict(
                page_size=4,
                enable_memory_saver=False,
                device="cpu",
                head_num=1,
                head_dim=64,
            ),
        )
        alloc_2c.attach_draft_kv_pool(draft_2c)
        toks_2c = alloc_2c.alloc_extend(
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([8]),
            seq_lens_cpu=torch.tensor([8]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=8,
        )
        base_2c = (t_2c.translate_loc_from_full_to_swa(toks_2c)[::4] // 4).tolist()
        compact_2c = t_2c.translate_swa_pages_for_draft(base_2c)
        self.assertTrue(all(1 <= p <= n_c for p in compact_2c))
        exp_t_swa = t_2c.expand_swa_pages(base_2c)

        pre_2c = self._wire_spec_draft(self._prefill(), t_2c, draft_2c, chunked=False)
        dec_2c = self._wire_spec_draft(self._decode(), t_2c, draft_2c, chunked=False)
        dec_2c.register_decode_request(
            9130,
            [20, 21],
            exp_t_swa,
            None,
            self._bootstrap_info(pre_2c),
            local_draft_swa_blocks=compact_2c,
        )
        self._pump(pre_2c, dec_2c)
        pre_2c.append_send_blocks(
            9130, [1, 2], swa_block_ids=exp_t_swa, draft_swa_block_ids=compact_2c
        )
        pre_2c.finalize_send(9130, aux_index=None)
        self._pump(pre_2c, dec_2c)
        self.assertTrue(dec_2c.maybe_start_pull(9130))
        dec_2c._engine._stats = ([], ["9130"], [])
        dec_2c._draft_engine._stats = ([], ["draft:9130"], [])
        dec_2c._draft_swa_engine._stats = ([], ["draft_swa:9130"], [])
        pre_2c._engine._stats = (["9130"], [], [])
        pre_2c._draft_engine._stats = (["draft:9130"], [], [])
        pre_2c._draft_swa_engine._stats = (["draft_swa:9130"], [], [])
        dec_2c._drain_engine()
        pre_2c._drain_engine()
        self.assertTrue(pre_2c.is_send_done(9130) and dec_2c.is_recv_done(9130))
        self.assertEqual(
            [(a[3], a[4]) for a, _ in dec_2c._draft_swa_engine.started],
            [(compact_2c, compact_2c)],
        )

    def test_heterogeneous_spec_draft_reshard_push_across_topologies(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        # Unequal TP: tp_p=4 -> tp_d=2 with mixed dtype Target fp8 (8 total heads) + Draft bf16
        p_t = _pool(swa_layers=(1, 2, 4, 5), head_num=2, dtype=torch.float8_e4m3fn)
        d_t = _pool(swa_layers=(1, 2, 4, 5), head_num=4, dtype=torch.float8_e4m3fn)
        p_swa = p_t.expand_swa_pages([5, 6])
        d_swa = d_t.expand_swa_pages([8, 9])

        # Test all 3 topologies at unequal TP: Case 2a, Case 2b (M_D=3 > M_T=2), Case 2c (mixed)
        p_2a = TpuFusedMHATokenToKVPool(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        d_2a = TpuFusedMHATokenToKVPool(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        p_2b = TpuUnifiedSWAKVPool(
            size=80,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=(0,)),
            swa=AttnLayerSpec(head_num=1, head_dim=64, layer_ids=(1, 2, 3)),
            device="cpu",
        )
        d_2b = TpuUnifiedSWAKVPool(
            size=80,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=2, head_dim=64, layer_ids=(0,)),
            swa=AttnLayerSpec(head_num=2, head_dim=64, layer_ids=(1, 2, 3)),
            device="cpu",
        )
        p_2b.swa_group_pages[:, 5] = torch.tensor([11, 12], dtype=torch.int32)
        p_2b.swa_group_pages[:, 6] = torch.tensor([13, 14], dtype=torch.int32)
        d_2b.swa_group_pages[:, 8] = torch.tensor([15, 16], dtype=torch.int32)
        d_2b.swa_group_pages[:, 9] = torch.tensor([17, 18], dtype=torch.int32)
        p_dswa_2b = p_2b.expand_swa_pages([5, 6])
        d_dswa_2b = d_2b.expand_swa_pages([8, 9])

        p_2c = SWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=64,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        d_2c = SWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )

        with patch.dict(sys.modules, _fake_pb2_modules()):
            for case_idx, (label, p_dp, d_dp, p_ds, d_ds) in enumerate(
                (
                    ("2a_fp8_bf16", p_2a, d_2a, [], []),
                    ("2b_unified", p_2b, d_2b, p_dswa_2b, d_dswa_2b),
                    ("2c_mixed", p_2c, d_2c, [1, 2], [3, 4]),
                )
            ):
                for mode_idx, (chunked, swa_only) in enumerate(
                    ((True, False), (False, False), (False, True))
                ):
                    with self.subTest(case=label, chunked=chunked, swa_only=swa_only):
                        ps = []
                        for r in (0, 1):
                            pm = self._prefill(attn_tp_rank=r, attn_tp_size=4)
                            pm.kv_args.total_kv_head_num = 16
                            self._wire_spec_draft(
                                pm,
                                p_t,
                                p_dp,
                                chunked=chunked,
                                draft_total_heads=4,
                                draft_swa_total_heads=4,
                                target_total_heads=16,
                            )
                            ps.append(pm)
                        dec = self._decode(attn_tp_rank=0, attn_tp_size=2)
                        dec.kv_args.total_kv_head_num = 16
                        self._wire_spec_draft(
                            dec,
                            d_t,
                            d_dp,
                            chunked=chunked,
                            draft_total_heads=4,
                            draft_swa_total_heads=4,
                            target_total_heads=16,
                        )
                        room = 9300 + case_idx * 10 + mode_idx
                        dec.register_decode_request(
                            room,
                            [] if swa_only else [20, 21],
                            d_swa,
                            None,
                            [
                                {"rank_ip": "127.0.0.1", "rank_port": p.rank_port}
                                for p in ps
                            ],
                            **(
                                {"local_draft_swa_blocks": d_ds}
                                if d_ds
                                else {}
                            ),
                        )
                        self._pump(*ps, dec)
                        for p in ps:
                            if chunked and not swa_only:
                                p.append_send_blocks(room, [1])
                                p._announce_reshard_chunk(room, 0, [1], total_pages=2)
                        self._pump(*ps, dec)
                        for p in ps:
                            if chunked and not swa_only:
                                p.append_send_blocks(
                                    room,
                                    [2],
                                    swa_block_ids=p_swa,
                                    draft_swa_block_ids=p_ds or None,
                                )
                            else:
                                p.append_send_blocks(
                                    room,
                                    [1, 2],
                                    swa_block_ids=p_swa,
                                    draft_swa_block_ids=p_ds or None,
                                )
                            p.finalize_send(room, aux_index=None)
                        self._pump(*ps, dec)

                        d_pushes = [
                            (a[2], a[3])
                            for a, _ in ps[0]._draft_engine.pushes
                            if a[4] == 0
                        ]
                        if label == "2a_fp8_bf16":
                            self.assertEqual(
                                d_pushes,
                                []
                                if swa_only
                                else (
                                    [([1], [20]), ([2], [21])]
                                    if chunked
                                    else [([1, 2], [20, 21])]
                                ),
                            )
                        elif label == "2b_unified":
                            self.assertEqual(
                                d_pushes,
                                [(p_ds, d_ds)]
                                if swa_only
                                else (
                                    [([1], [20]), ([2] + p_ds, [21] + d_ds)]
                                    if chunked
                                    else [([1, 2] + p_ds, [20, 21] + d_ds)]
                                ),
                            )
                        else:
                            self.assertEqual(
                                d_pushes,
                                []
                                if swa_only
                                else (
                                    [([1], [20]), ([2], [21])]
                                    if chunked
                                    else [([1, 2], [20, 21])]
                                ),
                            )
                            ds_pushes = [
                                (a[2], a[3])
                                for a, _ in ps[0]._draft_swa_engine.pushes
                                if a[4] == 0
                            ]
                            self.assertEqual(ds_pushes, [(p_ds, d_ds)])


if __name__ == "__main__":
    unittest.main()
