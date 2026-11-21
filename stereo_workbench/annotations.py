"""场景标注资产：单眼微小编号、双眼低对比标志、右眼专属纹理。

所有标注都“贴”在棋盘深度（BOARD_Z）上，因此与棋盘共享同一套投影与
栏杆遮挡规则：标注在某眼是否可见完全由立体几何决定，而不是按眼别硬写。
每个标注同时导出像素级掩码，供码流回读时区分“编码丢细节”与
“另一眼在几何上本来就看不到”。
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config as C

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fonts", "NotoSansSC-Regular.otf")


def world_per_pixel(z=C.BOARD_Z):
    """一个图像像素在深度 z 处对应的世界尺寸。"""
    return z / C.FOCAL


def image_u(world_x, eye_x, z=C.BOARD_Z):
    return C.FOCAL * (world_x - eye_x) / z + C.WIDTH / 2.0


def image_v(world_y, z=C.BOARD_Z):
    return C.HEIGHT / 2.0 + C.FOCAL * world_y / z


def rail_blocks(us, eye_x):
    """给定图像列数组，返回各列是否被前景栏杆遮挡（栏杆贯穿全高）。"""
    x_rail = eye_x + (us - C.WIDTH / 2.0) * C.RAIL_Z / C.FOCAL
    blocked = np.zeros(us.shape, dtype=bool)
    for x1, x2 in C.RAIL_BARS:
        blocked |= (x_rail >= x1) & (x_rail < x2)
    return blocked


def eye_position(eye, swap=False):
    """某只眼睛的世界 X；交换相机时左右位置对调。"""
    positions = (C.EYE_X[1], C.EYE_X[0]) if swap else C.EYE_X
    return positions[eye]


def _load_tiny_font():
    # PIL 内置位图字体，数字约 6x10 像素，正好构成“微小编号”
    return ImageFont.load_default()


def make_micro_patch(text="7"):
    """单眼微小编号贴图层。返回 (灰度贴片, 有效像素掩码)。"""
    font = _load_tiny_font()
    canvas = Image.new("L", (C.MICRO_W, C.MICRO_H), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ox = (C.MICRO_W - tw) // 2 - bbox[0]
    oy = (C.MICRO_H - th) // 2 - bbox[1]
    draw.text((ox, oy), text, fill=255, font=font)
    patch = np.asarray(canvas, dtype=np.uint8)
    mask = patch > 120
    # 亮底深字，保证在棋盘亮/暗格上都有反差
    patch = np.where(mask, 30, C.LIGHT_GRAY).astype(np.uint8)
    return patch, mask


def make_soft_patch():
    """双眼共有低对比标志：浅灰标牌上的低振幅圆环加十字。

    自带标牌底色，保证无论落在棋盘亮格还是暗格上，
    标志与底图的灰度差都只有 SOFT_AMPLITUDE。
    """
    w, h = C.SOFT_W, C.SOFT_H
    patch = np.full((h, w), C.SOFT_BASE, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ring = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                  - min(h, w) * 0.32) <= 1.4
    cross = (np.abs(xx - cx) <= 0.9) | (np.abs(yy - cy) <= 0.9)
    mark = ring | cross
    patch[mark] = np.clip(C.SOFT_BASE - C.SOFT_AMPLITUDE, 0, 255)
    # 整个贴片都是标牌（掩码全 True），前景用更深的标记位单独区分
    mask = np.ones((h, w), dtype=bool)
    return patch, mark


def make_right_texture(size=48, seed=20261121):
    """仅右眼叠加的确定性密纹（共同可见区域内提高右眼纹理复杂度）。"""
    rng = np.random.default_rng(seed)
    base = C.LIGHT_GRAY
    noise = rng.normal(0, C.RIGHT_TEXTURE_AMP / 2.0, size=(size, size))
    # 叠加细网格，保证即使噪纹被量化，边缘仍贡献编码困难度
    noise[::3, :] += C.RIGHT_TEXTURE_AMP / 1.5
    noise[:, ::3] -= C.RIGHT_TEXTURE_AMP / 1.5
    patch = np.clip(base + noise, 0, 255).astype(np.uint8)
    mask = np.ones((size, size), dtype=bool)
    return patch, mask


def _window(flags):
    """布尔序列 -> 连续 True 时段 [[start, end], ...]，帧号含端点。"""
    out, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append([start, i - 1])
            start = None
    if start is not None:
        out.append([start, len(flags) - 1])
    return out


def annotation_projection(annot, frame_idx, eye, swap=False):
    """计算标注在某眼某帧的投影盒与逐列遮挡，返回 dict 或 None（出画）。

    move_with_board=True 的标注随棋盘平移，因此会在不同帧滑入/滑出
    固定在世界坐标中的前景栏杆，产生随时间变化的单眼遮挡窗口。
    swap=True 时左右相机位置对调，遮挡关系随之整体反转。
    """
    eye_x = eye_position(eye, swap)
    wx = annot["wx"]
    if annot.get("move_with_board"):
        wx += C.X0_SPEED * frame_idx
    wpx, hpx = annot["w"], annot["h"]
    u_c = image_u(wx, eye_x)
    v_c = image_v(annot["wy"])
    u0 = int(round(u_c - wpx / 2.0))
    v0 = int(round(v_c - hpx / 2.0))
    if u0 >= C.WIDTH or u0 + wpx <= 0 or v0 >= C.HEIGHT or v0 + hpx <= 0:
        return None
    cols = np.arange(max(u0, 0), min(u0 + wpx, C.WIDTH))
    blocked = rail_blocks(cols.astype(np.float64), eye_x)
    return {"u0": u0, "v0": v0, "cols": cols,
            "blocked_ratio": float(blocked.mean()) if len(cols) else 1.0}


def annot_visibility(annot, frame_idx, eye, patch_mask, swap=False):
    """标注有效像素中未被栏杆遮挡且落在画面内的比例。"""
    proj = annotation_projection(annot, frame_idx, eye, swap)
    if proj is None:
        return 0.0
    u0, v0, cols = proj["u0"], proj["v0"], proj["cols"]
    blocked = rail_blocks(cols.astype(np.float64),
                          eye_position(eye, swap))
    vis_cols = set(cols[~blocked].tolist())
    total, seen = 0, 0
    for j in range(annot["w"]):
        u = u0 + j
        if u < 0 or u >= C.WIDTH:
            continue
        for i in range(annot["h"]):
            v = v0 + i
            if v < 0 or v >= C.HEIGHT or not patch_mask[i, j]:
                continue
            total += 1
            if u in vis_cols:
                seen += 1
    return seen / total if total else 0.0


def choose_micro_position(patch_mask, min_run=3, min_both=2):
    """扫描起始世界 X，找让编号先双眼可见、后仅单眼可见的运动轨迹。

    编号随棋盘平移（move_with_board），会在中段滑过固定栏杆：
    要求至少 min_both 帧双眼可见、至少 min_run 个连续帧仅一眼可见。
    布点完全由几何决定，交换相机后同一搜索把单眼窗口翻到另一侧。
    """
    best = None
    for wx0 in range(-220, 120, 2):
        annot = {"id": "micro", "wx": float(wx0), "wy": C.MICRO_WORLD_Y,
                 "w": C.MICRO_W, "h": C.MICRO_H, "move_with_board": True}
        r_vis = [annot_visibility(annot, f, 1, patch_mask)
                 >= C.ANNOT_VIS_RATIO for f in range(C.N_FRAMES)]
        l_vis = [annot_visibility(annot, f, 0, patch_mask)
                 >= C.ANNOT_VIS_RATIO for f in range(C.N_FRAMES)]
        both_n = sum(rv and lv for rv, lv in zip(r_vis, l_vis))
        if both_n < min_both:
            continue
        run = best_run = 0
        mono_eye = 0
        for rv, lv in zip(r_vis, l_vis):
            if rv != lv:
                run += 1
                if run > best_run:
                    best_run = run
                    mono_eye = 1 if rv else 0
            else:
                run = 0
        if best_run < min_run:
            continue
        # 优先更长的单眼连续段，其次更多双眼帧，最后取离画面中心近的点
        score = (best_run, both_n, -abs(wx0), -mono_eye)
        if best is None or score > best[0]:
            best = (score, float(wx0))
    if best is None:
        return float(C.MICRO_WORLD_X)
    return best[1]


def choose_soft_position(patch_mask, min_run=10):
    """低对比标志落在共同可见区：两眼贴片全程无栏杆遮挡（可见度=1）。"""
    best = None
    for wx in range(-120, 130, 1):
        annot = {"id": "soft", "wx": float(wx), "wy": C.SOFT_WORLD_Y,
                 "w": C.SOFT_W, "h": C.SOFT_H}
        ratios = [
            min(annot_visibility(annot, f, 0, patch_mask),
                annot_visibility(annot, f, 1, patch_mask))
            for f in range(C.N_FRAMES)]
        good = sum(r >= 0.999 for r in ratios)
        if good >= min_run and (best is None or good > best[0]):
            best = (good, float(wx))
    return best[1] if best else float(C.SOFT_WORLD_X)


def choose_right_texture_position(patch_mask, min_run=10):
    """右眼专属纹理必须位于几何共同可见区：两眼投影都不被栏杆遮挡。

    它只在右眼素材中绘制（eyes=(1,)），但几何上左眼对应位置同样可见，
    这样它对预算器的意义纯粹是“右眼纹理复杂度被单独提高”。
    """
    best = None
    for wx in range(-120, 130, 1):
        annot = {"id": "rtex", "wx": float(wx), "wy": C.SOFT_WORLD_Y,
                 "w": patch_mask.shape[1], "h": patch_mask.shape[0]}
        good = sum(
            annot_visibility(annot, f, 0, patch_mask) >= 0.999
            and annot_visibility(annot, f, 1, patch_mask) >= 0.999
            for f in range(C.N_FRAMES))
        if good >= min_run and (best is None or good > best[0]):
            best = (good, float(wx))
    return best[1] if best else 0.0


def build_annotations():
    """构造全部标注资产与自动布点，返回标注列表（含贴片/掩码/世界坐标）。

    mask 为绘制掩码（贴到场景上的像素）；score_mask 为评分时区分
    标志前景与局部背景的掩码（soft 只把圆环十字当前景）。
    """
    micro_patch, micro_mask = make_micro_patch()
    soft_patch, soft_mark = make_soft_patch()
    tex_patch, tex_mask = make_right_texture()
    micro_wx = choose_micro_position(micro_mask)
    soft_wx = choose_soft_position(np.ones_like(soft_mark))
    tex_wx = choose_right_texture_position(tex_mask)
    return [
        {"id": "micro", "kind": "mono_micro", "wx": micro_wx,
         "wy": C.MICRO_WORLD_Y, "w": C.MICRO_W, "h": C.MICRO_H,
         "patch": micro_patch, "mask": micro_mask,
         "draw_mask": micro_mask, "score_mask": micro_mask,
         "eyes": (0, 1), "move_with_board": True},
        {"id": "soft", "kind": "shared_low_contrast", "wx": soft_wx,
         "wy": C.SOFT_WORLD_Y, "w": C.SOFT_W, "h": C.SOFT_H,
         "patch": soft_patch, "mask": soft_mark,
         "draw_mask": np.ones_like(soft_mark), "score_mask": soft_mark,
         "eyes": (0, 1), "move_with_board": False,
         "stamp_full_rect": True},
        {"id": "rtex", "kind": "right_only_texture",
         "wx": tex_wx, "wy": C.SOFT_WORLD_Y + 72.0,
         "w": tex_patch.shape[1], "h": tex_patch.shape[0],
         "patch": tex_patch, "mask": tex_mask,
         "draw_mask": tex_mask, "score_mask": tex_mask,
         "eyes": (1,), "move_with_board": False},
    ]


def visibility_windows(annots, swap=False):
    """从几何计算每个标注在每眼的可见比例与连续可见时段。

    右眼专属纹理（eyes=(1,)）的左眼条目标记 geometric_only，
    表示该区域在几何上存在但素材不会在左眼绘制它——
    这与微编号的“几何遮挡”是两类不同的“看不到”。
    swap=True 时所有窗口必须随相机对调整体反转。
    """
    out = {}
    for a in annots:
        entry = {"id": a["id"], "kind": a["kind"],
                 "world_x": a["wx"], "world_y": a["wy"],
                 "size": [a["w"], a["h"]], "stamped_eyes": list(a["eyes"]),
                 "swap": swap, "eyes": {}}
        for eye in range(2):
            ratios = [annot_visibility(a, f, eye, a["draw_mask"], swap)
                      for f in range(C.N_FRAMES)]
            entry["eyes"][eye] = {
                "ratios": [round(x, 3) for x in ratios],
                "visible_frames": [f for f, x in enumerate(ratios)
                                   if x >= C.ANNOT_VIS_RATIO],
                "windows": _window([x >= C.ANNOT_VIS_RATIO for x in ratios]),
                "stamped": eye in a["eyes"],
                "geometric_only": eye not in a["eyes"],
            }
        out[a["id"]] = entry
    return out
