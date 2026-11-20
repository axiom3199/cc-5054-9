"""端到端流水线：两遍联合分配 -> 真实编码 -> 实际字节反馈 -> 解码 -> 汇总。

每个同步时间段（一帧）：
  第一遍  左眼在共享预算的一半额度内按 RD 边际贪心；
          约束模式下，对双眼均可见的对应格点，左眼质量封顶在
          “右眼最低质量 MSE - 失真差上限”，保证右眼停在最低档也不越界；
          单眼独有/遮挡格点不受对应约束，只承担自己的失真代价。
  真实编码左眼，得到实际包大小。
  反馈    右眼可用额度 = 总预算 - 左眼实际花费。
  第二遍  右眼在反馈额度内贪心；约束模式先对已越界的对应格点做
          “配对补齐”（强制升到差值达标的最低档），再做常规边际升级，
          普通升级若会造成越界则拒绝。
  真实编码右眼并解码。
"""

import os

import numpy as np

from . import allocator
from . import codec
from . import config as C
from . import container


def _corresponding_left(row, col_r, corr):
    """由右眼格点反查对应左眼列，无对应返回 -1。"""
    for cL in range(C.COLS):
        if int(corr[row, cL]) == col_r:
            return cL
    return -1


def _allocate_left(mode, ml, mr, vis_l, vis_r, corr, quota, hard_l):
    if mode == "equal":
        q_l, _, _ = allocator._equal_allocate(ml, quota)
        return q_l

    if mode == "constrained":
        def allow(r, c, new_mse):
            cR = int(corr[r, c])
            if cR < 0 or not (vis_l[r, c] and vis_r[r, cR]):
                return True  # 单眼独有/遮挡区域不参与对应约束
            m_right_min = allocator._lookup(
                mr[r][cR], C.PROBE_QUALITIES[0])[1]
            return new_mse >= m_right_min - C.DISTORTION_DIFF_LIMIT
    else:
        allow = None
    q_l, _, _ = allocator.greedy_single_eye(
        ml, quota, allow=allow,
        difficulty=None if mode == "equal" else hard_l)
    return q_l


def _allocate_right(mode, mr, q_l_fixed, ml, vis_l, vis_r, corr, quota,
                    hard_r):
    q_r = np.full((C.ROWS, C.COLS), C.PROBE_QUALITIES[0], dtype=np.int64)
    size_r = np.zeros((C.ROWS, C.COLS))
    mse_r = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            size_r[r, c], mse_r[r, c] = allocator._lookup(
                mr[r][c], q_r[r, c])
    base_r = float(size_r.sum())
    quota = max(float(quota), base_r)  # 托底：至少能以最低质量生成码流
    spent = base_r

    def left_mse_of(r, c):
        cL = _corresponding_left(r, c, corr)
        if cL < 0 or not (vis_l[r, cL] and vis_r[r, c]):
            return None
        return allocator._lookup(ml[r][cL], q_l_fixed[r, cL])[1]

    if mode == "constrained":
        # 配对补齐：优先把已经越界的双可见对应格点拉升到达标
        for r in range(C.ROWS):
            for c in range(C.COLS):
                m_left = left_mse_of(r, c)
                if m_left is None:
                    continue
                while abs(mse_r[r, c] - m_left) > C.DISTORTION_DIFF_LIMIT:
                    nq = allocator._next_quality(mr[r][c], q_r[r, c])
                    if nq is None:
                        break
                    s1, m1 = allocator._lookup(mr[r][c], nq)
                    extra = s1 - size_r[r, c]
                    if spent + extra > quota:
                        break
                    spent += extra
                    q_r[r, c] = nq
                    size_r[r, c], mse_r[r, c] = s1, m1

        def allow(r, c, new_mse):
            m_left = left_mse_of(r, c)
            if m_left is None:
                return True
            return abs(new_mse - m_left) <= C.DISTORTION_DIFF_LIMIT
    else:
        allow = None

    # 常规 RD 边际贪心（在补齐后的质量基础上继续升级）
    while True:
        best = None
        for r in range(C.ROWS):
            for c in range(C.COLS):
                nq = allocator._next_quality(mr[r][c], q_r[r, c])
                if nq is None:
                    continue
                s0, m0 = allocator._lookup(mr[r][c], q_r[r, c])
                s1, m1 = allocator._lookup(mr[r][c], nq)
                d = s1 - s0
                if spent + d > quota:
                    continue
                if allow is not None and not allow(r, c, m1):
                    continue
                ratio = (m0 - m1) / max(d, 1e-9)
                key = (ratio, float(hard_r[r, c]))
                if best is None or key > best[0]:
                    best = (key, r, c, nq, d, s1, m1)
        if best is None:
            break
        _, r, c, nq, d, s1, m1 = best
        q_r[r, c] = nq
        size_r[r, c], mse_r[r, c] = s1, m1
        spent += d
    return q_r


def _encode_strategy(mode, data, models):
    corr = data["corr"]
    result = {"frames": [], "packets": [[], []]}
    for f in range(C.N_FRAMES):
        ml = models[0 * C.N_FRAMES + f]
        mr = models[1 * C.N_FRAMES + f]
        vis_l = data["visible"][0][f]
        vis_r = data["visible"][1][f]
        hard_l = data["difficulty"][0][f]
        hard_r = data["difficulty"][1][f]
        half = C.SEGMENT_BUDGET / 2.0

        # 第一遍 + 左眼真实编码
        q_l = _allocate_left(mode, ml, mr, vis_l, vis_r, corr, half, hard_l)
        pk_l, rec_l, sz_l, mse_l = codec.encode_with_qualities(
            data["images"][0][f], q_l)
        actual_l = float(sz_l.sum())

        # 反馈：右眼额度 = 总预算 - 左眼实际包大小
        quota_r = allocator.get_quota(mode, C.SEGMENT_BUDGET, actual_l)
        if mode == "equal":
            q_r, _, _ = allocator._equal_allocate(mr, quota_r)
        else:
            q_r = _allocate_right(mode, mr, q_l, ml, vis_l, vis_r,
                                  corr, quota_r, hard_r)

        pk_r, rec_r, sz_r, mse_r = codec.encode_with_qualities(
            data["images"][1][f], q_r)

        result["packets"][0].append(pk_l)
        result["packets"][1].append(pk_r)
        result["frames"].append({
            "frame": f,
            "q": [q_l, q_r],
            "actual_size": [sz_l, sz_r],
            "actual_mse": [mse_l, mse_r],
            "recon": [rec_l, rec_r],
            "quota_r_actual": max(quota_r, 0.0),
            "spent_l_actual": actual_l,
        })
    return result


def summarize(result, data):
    total = np.zeros(2)
    vis_mse = [[], []]
    violations = []
    corr = data["corr"]
    for fr in result["frames"]:
        f = fr["frame"]
        for eye in range(2):
            total[eye] += fr["actual_size"][eye].sum()
            vis = data["visible"][eye][f]
            vis_mse[eye].append(fr["actual_mse"][eye][vis].mean()
                                if vis.any() else np.nan)
        for r in range(C.ROWS):
            for cL in range(C.COLS):
                cR = int(corr[r, cL])
                if cR < 0:
                    continue
                vl = data["visible"][0][f][r, cL]
                vr = data["visible"][1][f][r, cR]
                if vl and vr:
                    d = abs(float(fr["actual_mse"][0][r, cL])
                            - float(fr["actual_mse"][1][r, cR]))
                    if d > C.DISTORTION_DIFF_LIMIT:
                        violations.append({
                            "frame": f, "row": r, "col_l": cL, "col_r": cR,
                            "diff": d,
                            "mse_l": float(fr["actual_mse"][0][r, cL]),
                            "mse_r": float(fr["actual_mse"][1][r, cR]),
                        })
    return {
        "bytes_l": int(total[0]),
        "bytes_r": int(total[1]),
        "bytes_total": int(total.sum()),
        "mean_mse_l": float(np.nanmean(vis_mse[0])),
        "mean_mse_r": float(np.nanmean(vis_mse[1])),
        "mean_mse": float(np.nanmean(vis_mse[0] + vis_mse[1])),
        "n_violations": len(violations),
        "violations": violations,
    }


def run_all(data, models, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    strategies = {}
    for mode in ("equal", "adaptive", "constrained"):
        result = _encode_strategy(mode, data, models)
        for eye, suffix in ((0, "left"), (1, "right")):
            path = os.path.join(out_dir, f"{mode}_{suffix}.stb1")
            n, total = container.write_stream(path, eye, result["packets"][eye])
            result[f"path_{suffix}"] = path
            result[f"packets_{suffix}"] = n
            result[f"bytes_{suffix}_file"] = total
        result["summary"] = summarize(result, data)
        strategies[mode] = result
    return strategies
