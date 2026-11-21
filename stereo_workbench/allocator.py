"""联合字节预算器的分配原语。

预算器读取：
  * 每只眼的格点可见性（栏杆遮挡状态）；
  * 按棋盘深度建立的左右几何对应（由 scene.correspondence_map 给出）；
  * 实测的速率-失真模型与局部困难程度（梯度）。

本模块提供质量档位插值、额度反馈公式、统一质量分配和单眼 RD 贪心；
双眼对应的失真差约束在 pipeline 的两遍分配中实施
（左眼配对封顶 + 右眼配对补齐）。
"""

import numpy as np

from . import config as C

MIN_Q = C.PROBE_QUALITIES[0]


def _lookup(model, q):
    """在实测 RD 探针点之间线性插值，返回 (字节, MSE)。"""
    qs, sizes, mses = model
    return (float(np.interp(float(q), qs, sizes)),
            float(np.interp(float(q), qs, mses)))


def _next_quality(model, q):
    """返回比当前质量高一档的探针质量，已是最高则返回 None。"""
    qs = model[0]
    idx = int(np.searchsorted(qs, q, side="right"))
    return int(qs[idx]) if idx < len(qs) else None


def get_quota(mode, budget, other_spent):
    """当前眼可用额度。

    adaptive / constrained：另一眼的实际包大小直接反馈为剩余额度；
    equal：两眼始终平均切预算（无反馈）。
    """
    if other_spent is None or mode == "equal":
        return budget / 2.0
    return float(budget - other_spent)


def _equal_allocate(models, quota):
    """所有格点统一取能装进额度的最高质量档（平均分配策略）。"""
    chosen = MIN_Q
    for cand in C.QUALITY_GRID:
        total = sum(_lookup(models[r][c], cand)[0]
                    for r in range(C.ROWS) for c in range(C.COLS))
        if total <= quota:
            chosen = cand
    q = np.full((C.ROWS, C.COLS), chosen, dtype=np.int64)
    sizes = np.zeros((C.ROWS, C.COLS))
    mses = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            sizes[r, c], mses[r, c] = _lookup(models[r][c], chosen)
    return q, sizes, mses


def greedy_single_eye(models, quota, allow=None, difficulty=None, init_q=None):
    """单眼在给定额度内按 RD 边际性价比贪心升级。

    性价比 = MSE 下降 / 增加字节，本质上由局部困难程度决定：
    纹理/边缘越密的格点（difficulty 越大），升档带来的失真收益越高。
    difficulty 可选，用于边际比并列时按局部困难程度稳定排序。
    allow(r, c, new_mse) 可选回调，返回 False 时拒绝该格点升级，
    用于注入双眼对应失真差约束。
    init_q 可选，给定每格点的起始质量（如联合配对托底），默认最低档。
    """
    if init_q is None:
        init_q = np.full((C.ROWS, C.COLS), MIN_Q, dtype=np.int64)
    q = init_q.astype(np.int64).copy()
    sizes = np.zeros((C.ROWS, C.COLS))
    mses = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            sizes[r, c], mses[r, c] = _lookup(models[r][c], q[r, c])
    spent = float(sizes.sum())
    while True:
        best = None
        for r in range(C.ROWS):
            for c in range(C.COLS):
                nq = _next_quality(models[r][c], q[r, c])
                if nq is None:
                    continue
                s0, m0 = _lookup(models[r][c], q[r, c])
                s1, m1 = _lookup(models[r][c], nq)
                d = s1 - s0
                if spent + d > quota:
                    continue
                if allow is not None and not allow(r, c, m1):
                    continue
                ratio = (m0 - m1) / max(d, 1e-9)
                hard = float(difficulty[r, c]) if difficulty is not None else 0.0
                key = (ratio, hard)
                if best is None or key > best[0]:
                    best = (key, r, c, nq, d, s1, m1)
        if best is None:
            break
        _, r, c, nq, d, s1, m1 = best
        q[r, c] = nq
        sizes[r, c], mses[r, c] = s1, m1
        spent += d
    return q, sizes, mses
