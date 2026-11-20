"""合成双目场景：一块从栏杆后经过的棋盘格。

校正后的平行光轴相机，同一世界纹理在左右图像中行坐标相同、
仅存在与深度成反比的水平视差。前景栏杆比棋盘更靠近相机，
因此同一块棋盘纹理会在运动过程中从双眼可见变为单眼可见。
"""

import numpy as np

from . import config as C

CX = C.WIDTH / 2.0
CY = C.HEIGHT / 2.0
BOARD_Y0 = -130.0
BOARD_H = 260.0
RAIL_GRAY = 110


def _world_x_at_depth(u, eye_x, z):
    """图像列 u 反投影到深度 z 处的世界 X。"""
    return eye_x + (u - CX) * z / C.FOCAL


def _image_u(world_x, eye_x, z):
    """世界点投影到图像列。"""
    return C.FOCAL * (world_x - eye_x) / z + CX


def board_x0(frame_idx):
    return C.X0_START + C.X0_SPEED * frame_idx


def render_eye(frame_idx, eye):
    """渲染单眼一帧，返回 (灰度图, 栏杆遮挡掩码, 棋盘覆盖掩码)。"""
    eye_x = C.EYE_X[eye]
    x0 = board_x0(frame_idx)
    board_x1 = x0 + C.N_CELLS * C.CELL_WORLD

    us = np.arange(C.WIDTH, dtype=np.float64)
    vs = np.arange(C.HEIGHT, dtype=np.float64)

    x_board = _world_x_at_depth(us, eye_x, C.BOARD_Z)
    y_world = (vs - CY) * C.BOARD_Z / C.FOCAL

    on_board = (x_board >= x0) & (x_board < board_x1)
    on_board_y = (y_world >= BOARD_Y0) & (y_world < BOARD_Y0 + BOARD_H)

    col_idx = np.floor((x_board - x0) / C.CELL_WORLD).astype(np.int64)
    row_idx = np.floor((y_world - BOARD_Y0) / C.CELL_WORLD).astype(np.int64)
    parity = (col_idx[None, :] + row_idx[:, None]) % 2
    checker = np.where(parity == 0, C.LIGHT_GRAY, C.DARK_GRAY)

    img = np.full((C.HEIGHT, C.WIDTH), C.BG_GRAY, dtype=np.uint8)
    board_mask = on_board[None, :] & on_board_y[:, None]
    img[board_mask] = checker[board_mask].astype(np.uint8)

    x_rail = _world_x_at_depth(us, eye_x, C.RAIL_Z)
    rail_col = np.zeros(C.WIDTH, dtype=bool)
    for x1, x2 in C.RAIL_BARS:
        rail_col |= (x_rail >= x1) & (x_rail < x2)
    rail_mask = np.repeat(rail_col[None, :], C.HEIGHT, axis=0)
    img[rail_mask] = RAIL_GRAY
    return img, rail_mask, board_mask


def tile_visibility(rail_mask, board_mask):
    """每个格点的棋盘覆盖率与栏杆遮挡率，visible 取两者阈值判定。"""
    vis = np.zeros((C.ROWS, C.COLS), dtype=bool)
    cover = np.zeros((C.ROWS, C.COLS))
    occl = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            block = np.s_[r * C.TILE:(r + 1) * C.TILE,
                          c * C.TILE:(c + 1) * C.TILE]
            n = C.TILE * C.TILE
            cover[r, c] = board_mask[block].mean()
            occl[r, c] = rail_mask[block].mean()
            vis[r, c] = cover[r, c] >= 0.5 and occl[r, c] <= 0.5
    return vis, cover, occl


def correspondence_map():
    """左眼格点中心 -> 右眼格点列（按棋盘深度做几何对应）。

    行在校正后不变；超出右图范围或列不一致时返回 -1。
    视差 d = f * B / Z，与深度成反比，这正是预算器读取的几何对应。
    """
    disparity = C.FOCAL * C.BASELINE / C.BOARD_Z
    corr = np.full((C.ROWS, C.COLS), -1, dtype=np.int64)
    for r in range(C.ROWS):
        for cL in range(C.COLS):
            uL = cL * C.TILE + C.TILE / 2.0
            uR = uL - disparity
            cR = int(np.floor(uR / C.TILE))
            if 0 <= cR < C.COLS:
                corr[r, cL] = cR
    return corr


def tile_difficulty(img):
    """局部困难程度：格点内平均梯度幅度（纹理/边缘越密越难编码）。"""
    gx = np.abs(np.diff(img.astype(np.float64), axis=1, prepend=img[:, :1]))
    gy = np.abs(np.diff(img.astype(np.float64), axis=0, prepend=img[:1, :]))
    grad = gx + gy
    diff = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            diff[r, c] = grad[r * C.TILE:(r + 1) * C.TILE,
                              c * C.TILE:(c + 1) * C.TILE].mean()
    return diff


def cell_visibility(frame_idx, eye):
    """每个世界棋盘格在该眼的可见比例（未被栏杆遮挡的投影占比）。"""
    eye_x = C.EYE_X[eye]
    x0 = board_x0(frame_idx)
    out = np.zeros(C.N_CELLS)
    for cell_id in range(C.N_CELLS):
        wx0 = x0 + cell_id * C.CELL_WORLD
        wx1 = wx0 + C.CELL_WORLD
        u0 = _image_u(wx0, eye_x, C.BOARD_Z)
        u1 = _image_u(wx1, eye_x, C.BOARD_Z)
        ia, ib = int(np.ceil(min(u0, u1))), int(np.floor(max(u0, u1)))
        ia, ib = max(ia, 0), min(ib, C.WIDTH - 1)
        if ib < ia:
            continue
        us = np.arange(ia, ib + 1)
        x_rail = _world_x_at_depth(us, eye_x, C.RAIL_Z)
        blocked = np.zeros(us.shape, dtype=bool)
        for rx1, rx2 in C.RAIL_BARS:
            blocked |= (x_rail >= rx1) & (x_rail < rx2)
        out[cell_id] = 1.0 - blocked.mean()
    return out


def build_scene():
    """生成整段双目场景及预算器需要的全部辅助量。"""
    data = {
        "images": [[None] * C.N_FRAMES for _ in range(2)],
        "rail": [[None] * C.N_FRAMES for _ in range(2)],
        "visible": [[None] * C.N_FRAMES for _ in range(2)],
        "cover": [[None] * C.N_FRAMES for _ in range(2)],
        "occl": [[None] * C.N_FRAMES for _ in range(2)],
        "difficulty": [[None] * C.N_FRAMES for _ in range(2)],
        "corr": correspondence_map(),
        "cell_vis": [[None] * C.N_FRAMES for _ in range(2)],
    }
    for f in range(C.N_FRAMES):
        for eye in range(2):
            img, rail, board = render_eye(f, eye)
            vis, cover, occl = tile_visibility(rail, board)
            data["images"][eye][f] = img
            data["rail"][eye][f] = rail
            data["visible"][eye][f] = vis
            data["cover"][eye][f] = cover
            data["occl"][eye][f] = occl
            data["difficulty"][eye][f] = tile_difficulty(img)
            data["cell_vis"][eye][f] = cell_visibility(f, eye)
    return data


def find_transition_cell(data):
    """找一块从双眼可见变为单眼可见的棋盘格，返回 (cell_id, 帧号列表)。"""
    states = []
    for cid in range(C.N_CELLS):
        seq = []
        for f in range(C.N_FRAMES):
            l = data["cell_vis"][0][f][cid] > 0.5
            r = data["cell_vis"][1][f][cid] > 0.5
            seq.append((l, r))
        both = [i for i, s in enumerate(seq) if s == (True, True)]
        mono = [i for i, s in enumerate(seq) if s in ((True, False), (False, True))]
        if both and mono and mono[0] > both[0]:
            states.append((cid, both[0], mono[0]))
    if not states:
        return None
    cid = states[0][0]
    frames = list(range(C.N_FRAMES))
    return cid, frames
