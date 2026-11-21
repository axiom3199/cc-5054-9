"""编码预算档案：宽裕档与两套饥饿档（两眼权重互换）。

每个档案给出每段共享预算和左眼权重；swapA 与 swapB 的区别仅是
两眼预算对调。分配在各眼额度内沿实测 RD 曲线贪心升级，
额度连最低档都放不下时托底到最低质量并记录“压底”事件——
这正是微小编号/低对比标志被 JPEG 抹平的直接原因。
"""

import numpy as np

from . import allocator
from . import codec
from . import config as C

PROFILES = {
    # 宽裕档：两眼均分充足预算，标志应全部可辨认
    "generous": {"budget": C.SEGMENT_BUDGET, "left_weight": 0.5},
    # 饥饿档 A：偏袒左眼，右眼纹理区域与右眼标志首先被压坏
    "starve_swapA": {"budget": C.STARVATION_BUDGET, "left_weight": 0.75},
    # 饥饿档 B：仅把两眼预算对调，其余完全一致，用于 A/B 归因
    "starve_swapB": {"budget": C.STARVATION_BUDGET, "left_weight": 0.25},
}


def _allocate_eye(models, quota, difficulty):
    """单眼在额度内沿实测 RD 贪心；额度不足最低档时托底并标记。"""
    q = np.full((C.ROWS, C.COLS), C.PROBE_QUALITIES[0], dtype=np.int64)
    sizes = np.zeros((C.ROWS, C.COLS))
    mses = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            sizes[r, c], mses[r, c] = allocator._lookup(
                models[r][c], C.PROBE_QUALITIES[0])
    spent = float(sizes.sum())
    floored = spent > quota
    quota = max(quota, spent)  # 托底：码流必须能生成
    while True:
        best = None
        for r in range(C.ROWS):
            for c in range(C.COLS):
                nq = allocator._next_quality(models[r][c], q[r, c])
                if nq is None:
                    continue
                s0, m0 = allocator._lookup(models[r][c], q[r, c])
                s1, m1 = allocator._lookup(models[r][c], nq)
                d = s1 - s0
                if spent + d > quota:
                    continue
                ratio = (m0 - m1) / max(d, 1e-9)
                key = (ratio, float(difficulty[r, c]))
                if best is None or key > best[0]:
                    best = (key, r, c, nq, d, s1, m1)
        if best is None:
            break
        _, r, c, nq, d, s1, m1 = best
        q[r, c] = nq
        sizes[r, c], mses[r, c] = s1, m1
        spent += d
    return q, sizes, mses, floored


def encode_profile(data, models, profile_name):
    """按档案逐段真实编码两眼，返回含解码重建帧的完整结果。"""
    spec = PROFILES[profile_name]
    budget, wl = spec["budget"], spec["left_weight"]
    result = {"profile": profile_name, "budget": budget,
              "left_weight": wl, "frames": [],
              "packets": [[], []], "floored_frames": []}
    for f in range(C.N_FRAMES):
        ml = models[0 * C.N_FRAMES + f]
        mr = models[1 * C.N_FRAMES + f]
        quota_l = budget * wl
        quota_r = budget * (1.0 - wl)
        q_l, _, _, floor_l = _allocate_eye(
            ml, quota_l, data["difficulty"][0][f])
        q_r, _, _, floor_r = _allocate_eye(
            mr, quota_r, data["difficulty"][1][f])
        pk_l, rec_l, sz_l, mse_l = codec.encode_with_qualities(
            data["images"][0][f], q_l)
        pk_r, rec_r, sz_r, mse_r = codec.encode_with_qualities(
            data["images"][1][f], q_r)
        result["packets"][0].append(pk_l)
        result["packets"][1].append(pk_r)
        result["frames"].append({
            "frame": f, "q": [q_l, q_r],
            "actual_size": [sz_l, sz_r], "actual_mse": [mse_l, mse_r],
            "recon": [rec_l, rec_r],
            "quota": [quota_l, quota_r],
            "spent": [float(sz_l.sum()), float(sz_r.sum())],
        })
        if floor_l or floor_r:
            result["floored_frames"].append(
                {"frame": f, "left_floored": floor_l,
                 "right_floored": floor_r})
    totals = [sum(fr["spent"][e] for fr in result["frames"]) for e in range(2)]
    result["bytes_total"] = [int(totals[0]), int(totals[1])]
    return result
