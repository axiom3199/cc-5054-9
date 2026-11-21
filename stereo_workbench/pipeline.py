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


def _corresponding_left(corr, row, col_r):
    """右眼格点反查对应左眼列，无对应返回 -1。"""
    for cL in range(C.COLS):
        if int(corr[row, cL]) == col_r:
            return cL
    return -1


def _joint_floors(ml, mr, vis_l, vis_r, corr):
    """约束模式的双眼联合配对托底。

    对每一对双眼均可见的对应格点，枚举左右质量组合，选满足
    |MSE_L − MSE_R| ≤ 失真差上限的最便宜 (qL, qR) 作为双方的质量地板。
    单侧无档可降的死锁（如左右最低档本身就差超上限）在此被联合升级
    一并解除；单眼独有/遮挡格点只取最低档，不参与对应约束。
    返回 (q_l_floor, q_r_floor, floor_cost_l, floor_cost_r, paired)。
    """
    q_l = np.full((C.ROWS, C.COLS), C.PROBE_QUALITIES[0], dtype=np.int64)
    q_r = np.full((C.ROWS, C.COLS), C.PROBE_QUALITIES[0], dtype=np.int64)
    paired = np.zeros((C.ROWS, C.COLS), dtype=bool)
    grid = C.QUALITY_GRID
    for r in range(C.ROWS):
        for cL in range(C.COLS):
            cR = int(corr[r, cL])
            if cR < 0 or not (vis_l[r, cL] and vis_r[r, cR]):
                continue
            paired[r, cL] = True
            best = None
            for ql in grid:
                sl, ml_v = allocator._lookup(ml[r][cL], ql)
                for qr in grid:
                    sr, mr_v = allocator._lookup(mr[r][cR], qr)
                    if abs(ml_v - mr_v) <= C.DISTORTION_DIFF_LIMIT:
                        cost = sl + sr
                        if best is None or cost < best[0]:
                            best = (cost, ql, qr)
            if best is None:
                # 理论上不会发生：左右都到最高档时 MSE 都接近 0，必达标。
                # 兜底取压差最小的组合。
                best_d, best_q = None, (grid[-1], grid[-1])
                for ql in grid:
                    _, ml_v = allocator._lookup(ml[r][cL], ql)
                    for qr in grid:
                        _, mr_v = allocator._lookup(mr[r][cR], qr)
                        d = abs(ml_v - mr_v)
                        if best_d is None or d < best_d:
                            best_d, best_q = d, (ql, qr)
                best = (0, best_q[0], best_q[1])
            _, ql, qr = best
            # 左眼地板记在左列；右眼地板记在对应右列（多对一取更严的高档）
            q_l[r, cL] = max(q_l[r, cL], ql)
            q_r[r, cR] = max(q_r[r, cR], qr)
    cost_l = sum(allocator._lookup(ml[r][c], q_l[r, c])[0]
                 for r in range(C.ROWS) for c in range(C.COLS))
    cost_r = sum(allocator._lookup(mr[r][c], q_r[r, c])[0]
                 for r in range(C.ROWS) for c in range(C.COLS))
    return q_l, q_r, cost_l, cost_r, paired


def _allocate_left(mode, ml, mr, vis_l, vis_r, corr, quota, hard_l,
                   floors=None, cap_mse=None):
    if mode == "equal":
        q_l, _, _ = allocator._equal_allocate(ml, quota)
        return q_l

    if mode == "constrained":
        # 联合地板保证对应格点有可行解；天花板保证左眼不会好过右眼到顶后
        # 仍追不上的水平。左眼只在 [地板, 天花板] 之间按 RD 边际贪心。
        def allow(r, c, new_mse):
            if cap_mse is None:
                return True
            cap = cap_mse[r, c]
            if not np.isfinite(cap):
                return True
            return new_mse >= cap
        q_l, _, _ = allocator.greedy_single_eye(
            ml, quota, allow=allow, difficulty=hard_l, init_q=floors)
        return q_l

    q_l, _, _ = allocator.greedy_single_eye(
        ml, quota, allow=None,
        difficulty=None if mode == "equal" else hard_l)
    return q_l


def _allocate_right(mode, mr, q_l_fixed, ml, vis_l, vis_r, corr, quota,
                    hard_r, floors=None):
    if mode == "constrained":
        init_q = floors
    else:
        init_q = np.full((C.ROWS, C.COLS), C.PROBE_QUALITIES[0],
                         dtype=np.int64)
    q_r = init_q.astype(np.int64).copy()
    size_r = np.zeros((C.ROWS, C.COLS))
    mse_r = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            size_r[r, c], mse_r[r, c] = allocator._lookup(
                mr[r][c], q_r[r, c])
    base_r = float(size_r.sum())
    quota = max(float(quota), base_r)  # 托底：至少能以地板/最低质量生成码流
    spent = base_r

    def left_mse_of(r, c):
        cL = _corresponding_left(corr, r, c)
        if cL < 0 or not (vis_l[r, cL] and vis_r[r, c]):
            return None
        return allocator._lookup(ml[r][cL], q_l_fixed[r, cL])[1]

    if mode == "constrained":
        # 配对补齐（循环）：左眼贪心后 MSE 通常比联合地板更低，右眼需跟着
        # 升级到与左眼实际 MSE 压差达标。反复扫描直到一轮内无可升级项；
        # 额度不足时跳过该格点，保留越界并由报告记录。
        while True:
            progressed = False
            for r in range(C.ROWS):
                for c in range(C.COLS):
                    m_left = left_mse_of(r, c)
                    if m_left is None:
                        continue
                    if abs(mse_r[r, c] - m_left) <= C.DISTORTION_DIFF_LIMIT:
                        continue
                    target_q, best_diff = None, None
                    for cand in C.QUALITY_GRID:
                        if cand <= q_r[r, c]:
                            continue
                        _, cand_mse = allocator._lookup(mr[r][c], cand)
                        if abs(cand_mse - m_left) <= C.DISTORTION_DIFF_LIMIT:
                            target_q = cand
                            break
                        d = abs(cand_mse - m_left)
                        if best_diff is None or d < best_diff:
                            best_diff, target_q = d, cand
                    if target_q is None or target_q <= q_r[r, c]:
                        continue
                    s1, m1 = allocator._lookup(mr[r][c], target_q)
                    extra = s1 - size_r[r, c]
                    if spent + extra > quota:
                        continue
                    spent += extra
                    q_r[r, c] = target_q
                    size_r[r, c], mse_r[r, c] = s1, m1
                    progressed = True
            if not progressed:
                break

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


def _repair_constrained_real(img_l, img_r, q_l, q_r, mse_l, mse_r, size_r,
                             vis_l, vis_r, corr, total_budget):
    """用真实编码 MSE 修复模型偏差导致的残余越界（预算中性）。

    分配阶段依据插值 RD 模型保证达标，但真实 JPEG 的格点 MSE 与模型
    可能有几个单位的偏差，使个别对应对在真实码流上略超上限。这里直接
    重编违例右眼格点逐档升级，直到真实压差达标；升级只在总预算内进行，
    因此不会突破每帧共享预算。返回更新后的 (q_r, packets_r, rec_r,
    size_r, mse_r)；无违例时原样返回。
    """
    def real_pair_diff(r, cL, cR):
        return abs(float(mse_l[r, cL]) - float(mse_r[r, cR]))

    def has_violation():
        for r in range(C.ROWS):
            for cL in range(C.COLS):
                cR = int(corr[r, cL])
                if cR < 0 or not (vis_l[r, cL] and vis_r[r, cR]):
                    continue
                if real_pair_diff(r, cL, cR) > C.DISTORTION_DIFF_LIMIT:
                    return True
        return False

    if not has_violation():
        return q_r, None, None, size_r, mse_r

    tiles_r = codec.split_frame(img_r)
    packets_r = [[None] * C.COLS for _ in range(C.ROWS)]
    rec_tiles = [[None] * C.COLS for _ in range(C.ROWS)]
    for r in range(C.ROWS):
        for c in range(C.COLS):
            pkt = codec.encode_tile(tiles_r[r][c], int(q_r[r, c]))
            packets_r[r][c] = pkt
            rec_tiles[r][c] = codec.decode_tile(pkt)
    spent = float(size_r.sum())
    grid = C.QUALITY_GRID

    # 多对一共用右眼列：该列需满足所有对应左眼的压差，取最严要求
    while has_violation():
        progressed = False
        for r in range(C.ROWS):
            for c in range(C.COLS):
                cL = _corresponding_left(corr, r, c)
                if cL < 0 or not (vis_l[r, cL] and vis_r[r, c]):
                    continue
                if real_pair_diff(r, cL, c) <= C.DISTORTION_DIFF_LIMIT:
                    continue
                idx = grid.index(int(q_r[r, c]))
                if idx >= len(grid) - 1:
                    continue
                nq = grid[idx + 1]
                pkt = codec.encode_tile(tiles_r[r][c], nq)
                extra = len(pkt) - int(size_r[r, c])
                if spent + extra > total_budget:
                    continue
                rec = codec.decode_tile(pkt)
                base = tiles_r[r][c].astype(np.float64)
                nmse = float(np.mean(
                    (rec.astype(np.float64) - base) ** 2))
                spent += extra
                q_r[r, c] = nq
                packets_r[r][c] = pkt
                rec_tiles[r][c] = rec
                size_r[r, c] = len(pkt)
                mse_r[r, c] = nmse
                progressed = True
        if not progressed:
            break
    return q_r, packets_r, codec.build_frame(rec_tiles), size_r, mse_r


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

        # 约束模式：先求双眼联合配对地板，保证每对双可见格点都存在
        # 满足失真差上限的可行 (qL, qR)，再在地板之上分两遍贪心。
        floor_l = floor_r = None
        cap_l = None
        if mode == "constrained":
            floor_l, floor_r, cost_l, cost_r, _ = _joint_floors(
                ml, mr, vis_l, vis_r, corr)
            # 左眼额度至少要装下它那一侧的联合地板
            left_quota = max(half, cost_l)
            # 左眼天花板：右眼即使到最高档也有其 MSE 下界，左眼不允许
            # 编码得比“右眼最好 MSE − 失真差上限”还要好，否则该对在
            # 右眼到顶后仍会越界（右眼纹理天然更难编码的情形）。
            cap_l = np.full((C.ROWS, C.COLS), np.inf)
            for r in range(C.ROWS):
                for cL in range(C.COLS):
                    cR = int(corr[r, cL])
                    if cR < 0 or not (vis_l[r, cL] and vis_r[r, cR]):
                        continue
                    m_right_best = allocator._lookup(
                        mr[r][cR], C.QUALITY_GRID[-1])[1]
                    cap_l[r, cL] = max(
                        0.0, m_right_best - C.DISTORTION_DIFF_LIMIT)
        else:
            left_quota = half

        # 第一遍 + 左眼真实编码
        q_l = _allocate_left(mode, ml, mr, vis_l, vis_r, corr,
                             left_quota, hard_l, floors=floor_l,
                             cap_mse=cap_l)
        pk_l, rec_l, sz_l, mse_l = codec.encode_with_qualities(
            data["images"][0][f], q_l)
        actual_l = float(sz_l.sum())

        # 反馈：右眼额度 = 总预算 - 左眼实际包大小
        quota_r = allocator.get_quota(mode, C.SEGMENT_BUDGET, actual_l)
        if mode == "equal":
            q_r, _, _ = allocator._equal_allocate(mr, quota_r)
        else:
            q_r = _allocate_right(mode, mr, q_l, ml, vis_l, vis_r,
                                  corr, quota_r, hard_r, floors=floor_r)

        pk_r, rec_r, sz_r, mse_r = codec.encode_with_qualities(
            data["images"][1][f], q_r)

        if mode == "constrained":
            # 真实码流回查：用真实 MSE 修复模型偏差造成的残余越界
            q_r, fix_pk, fix_rec, sz_r, mse_r = _repair_constrained_real(
                data["images"][0][f], data["images"][1][f],
                q_l, q_r, mse_l, mse_r, sz_r,
                vis_l, vis_r, corr, C.SEGMENT_BUDGET)
            if fix_pk is not None:
                pk_r, rec_r = fix_pk, fix_rec

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
