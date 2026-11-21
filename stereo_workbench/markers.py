"""标志检测与立体字幕对齐器。

对齐器只信任从“实际码流解码出的图像”里重新检出的标志：
  * 微小编号、低对比标志各自给出可辨认度评分（0..1）；
  * 在右眼搜索窗内与左眼标志做模板匹配，实测水平视差；
  * 评分过低或视差与几何真值超差时拒绝自动放字幕，并写明拒绝原因；
  * 通过后按实测视差把字幕落在双眼共用的字幕安全区内。

作者可更换参考物：reference_id 指定用哪个标志做匹配；
预算压坏仅有的可匹配标志时，换一个仍然完好的标志即可恢复自动对齐。
"""

import numpy as np

from . import annotations as A
from . import config as C


def _patch_bbox(annot, frame_idx, eye, swap=False):
    proj = A.annotation_projection(annot, frame_idx, eye, swap)
    if proj is None:
        return None
    u0, v0 = proj["u0"], proj["v0"]
    return (max(u0, 0), max(v0, 0),
            min(u0 + annot["w"], C.WIDTH), min(v0 + annot["h"], C.HEIGHT))


def detect_score(frame, annot, frame_idx, eye, source_frame=None,
                 swap=False):
    """从解码帧里重检测一个标注，返回可辨认度评分与局部对比度。

    评分由两个可解释分量组成：
      形态 NCC：解码区域与“源帧同区域”的归一化互相关，
                笔画/圆环形态是否保住（需要 source_frame 给出未编码基准）；
      幅度保留率：标注前景与局部背景的灰度差相对源帧的比例，
                  低对比标志在强压缩下首先被量化抹平，该分量先掉。
    没有源帧基准时退化为仅用幅度分量。
    """
    bbox = _patch_bbox(annot, frame_idx, eye, swap)
    if bbox is None:
        return {"score": 0.0, "contrast": 0.0, "ncc": 0.0,
                "amplitude_kept": 0.0, "present": False}
    x0, y0, x1, y1 = bbox
    region = frame[y0:y1, x0:x1].astype(np.float64)
    if region.size == 0:
        return {"score": 0.0, "contrast": 0.0, "ncc": 0.0,
                "amplitude_kept": 0.0, "present": False}
    contrast = float(region.std())

    proj = A.annotation_projection(annot, frame_idx, eye, swap)
    full_u0, full_v0 = proj["u0"], proj["v0"]
    pmask = annot["score_mask"]
    px0, py0 = x0 - full_u0, y0 - full_v0
    tmask = pmask[py0:py0 + region.shape[0], px0:px0 + region.shape[1]]

    ncc = 0.0
    if source_frame is not None:
        src_region = source_frame[y0:y1, x0:x1].astype(np.float64)
        a = region - region.mean()
        b = src_region - src_region.mean()
        denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        if denom > 1e-9:
            ncc = max(0.0, float((a * b).sum() / denom))

    # 幅度保留率：解码区域与源区域各自“前景-局部背景”均值差之比。
    # 用源帧而非贴片模板做基准，自动适配棋盘底图，避免稀疏笔画被背景稀释。
    amplitude_kept = 0.0
    if tmask.any() and (~tmask).any() and source_frame is not None:
        src_region = source_frame[y0:y1, x0:x1].astype(np.float64)
        diff_dec = abs(region[tmask].mean() - region[~tmask].mean())
        diff_src = abs(src_region[tmask].mean() - src_region[~tmask].mean())
        amplitude_kept = min(diff_dec / max(diff_src, 1e-9), 1.0)
    # 形态为主（0.7）、幅度为辅（0.3）：低对比标志幅度天然小，
    # 只要形态还在就应可读；强压缩会同时摧毁两者。
    score = (0.7 * ncc + 0.3 * amplitude_kept
             if source_frame is not None else amplitude_kept)
    return {"score": float(score), "contrast": contrast,
            "ncc": float(ncc), "amplitude_kept": float(amplitude_kept),
            "present": True}


def geometric_disparity(annot, swap=False):
    """标注在棋盘深度上的理论水平视差（左眼列 - 右眼列）。"""
    eye_l = C.EYE_X[1] if swap else C.EYE_X[0]
    eye_r = C.EYE_X[0] if swap else C.EYE_X[1]
    return C.FOCAL * (eye_r - eye_l) / C.BOARD_Z


def estimate_disparity(frame_l, frame_r, annot, frame_idx, swap=False):
    """在右眼搜索窗内匹配左眼标志，实测视差（左列 - 右列）。

    只在标志笔画像素（score_mask）上计算 NCC，避免大面积均匀
    标牌底色把相关系数稀释到 0。以几何真值为中心整像素搜索，
    再用抛物线拟合做亚像素细化。
    """
    bbox_l = _patch_bbox(annot, frame_idx, 0, swap)
    if bbox_l is None:
        return None
    x0, y0, x1, y1 = bbox_l
    templ = frame_l[y0:y1, x0:x1].astype(np.float64)
    proj = A.annotation_projection(annot, frame_idx, 0, swap)
    px0, py0 = x0 - proj["u0"], y0 - proj["v0"]
    pmask = annot["score_mask"][py0:py0 + templ.shape[0],
                                 px0:px0 + templ.shape[1]]
    truth = geometric_disparity(annot, swap)
    expected_r = x0 - truth
    tol = C.DISPARITY_SEARCH_HALF
    samples = []
    for dx in range(-tol, tol + 1):
        rx = int(round(expected_r)) + dx
        if rx < 0 or rx + templ.shape[1] > C.WIDTH:
            continue
        cand = frame_r[y0:y1, rx:rx + templ.shape[1]].astype(np.float64)
        if cand.shape != templ.shape or pmask.sum() < 3:
            continue
        a = templ[pmask] - templ[pmask].mean()
        b = cand[pmask] - cand[pmask].mean()
        denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
        if denom > 1e-9:
            ncc = float((a * b).sum() / denom)
        else:
            # 笔画为恒定灰度（方差为 0）：均值完全重合给满分，
            # 差几个灰度级按 10 级尺度线性衰减
            diff = float(abs(templ[pmask].mean() - cand[pmask].mean()))
            ncc = float(np.clip(1.0 - diff / 10.0, 0.0, 1.0))
        samples.append((dx, rx, ncc))
    if not samples:
        return {"measured": None, "truth": float(truth), "error": None,
                "ncc": 0.0}
    peak = max(samples, key=lambda t: t[2])
    idx = samples.index(peak)
    dx_p, rx_p, ncc_p = peak
    # 抛物线亚像素细化
    sub = 0.0
    if 0 < idx < len(samples) - 1:
        n0, n1, n2 = samples[idx - 1][2], ncc_p, samples[idx + 1][2]
        denom2 = 2.0 * (n0 - 2.0 * n1 + n2)
        if abs(denom2) > 1e-9:
            sub = float(np.clip((n0 - n2) / denom2, -1.0, 1.0))
    measured = float(x0 - (rx_p + sub))
    return {"measured": measured, "truth": float(truth),
            "error": abs(measured - float(truth)), "ncc": float(ncc_p)}


def in_safe_box(u, v):
    x0, y0, x1, y1 = C.SAFE_BOX
    return x0 <= u <= x1 and y0 <= v <= y1


def place_subtitle(frame_l, frame_r, annots, frame_idx,
                   reference_id="soft", swap=False,
                   source_l=None, source_r=None):
    """双眼共用字幕安全区上的自动对齐决策（基于解码图像）。

    返回决策 dict：accepted / reason / 参考标志评分 / 实测视差 /
    左右字幕落点。拒绝时仍给出建议落点并标记不可自动投放。
    """
    ref = next((a for a in annots if a["id"] == reference_id), None)
    decision = {"frame": frame_idx, "reference": reference_id,
                "accepted": False, "reasons": [], "safe_box": list(C.SAFE_BOX)}
    if ref is None:
        decision["reasons"].append(f"参考物 {reference_id} 不存在，作者可更换参考物")
        return decision

    det_l = detect_score(frame_l, ref, frame_idx, 0, source_l, swap)
    det_r = detect_score(frame_r, ref, frame_idx, 1, source_r, swap)
    decision["detect_left"] = det_l
    decision["detect_right"] = det_r

    # 几何可见性闸门：某眼在该帧天然看不到参考物时直接拒绝，
    # 与“编码压坏”严格区分（可见性来自场景真值掩码而非解码结果）
    geo_l = A.annot_visibility(ref, frame_idx, 0, ref["draw_mask"], swap)
    geo_r = A.annot_visibility(ref, frame_idx, 1, ref["draw_mask"], swap)
    decision["geo_visible"] = [round(geo_l, 3), round(geo_r, 3)]
    if geo_l < C.ANNOT_VIS_RATIO:
        decision["reasons"].append(
            f"参考物在左眼几何遮挡（可见比例 {geo_l:.2f}），非编码损失")
    if geo_r < C.ANNOT_VIS_RATIO:
        decision["reasons"].append(
            f"参考物在右眼几何遮挡（可见比例 {geo_r:.2f}），非编码损失")

    if det_l["score"] < C.MARKER_SCORE_MIN:
        decision["reasons"].append(
            f"左眼参考标志评分 {det_l['score']:.2f} < {C.MARKER_SCORE_MIN}（疑似编码压坏）")
    if det_r["score"] < C.MARKER_SCORE_MIN:
        decision["reasons"].append(
            f"右眼参考标志评分 {det_r['score']:.2f} < {C.MARKER_SCORE_MIN}（疑似编码压坏）")

    disp = estimate_disparity(frame_l, frame_r, ref, frame_idx, swap)
    decision["disparity"] = disp
    if disp is None or disp["measured"] is None:
        decision["reasons"].append("右眼搜索窗内找不到可匹配标志")
    else:
        if disp["ncc"] < C.DISPARITY_NCC_MIN - 1e-9:
            decision["reasons"].append(
                f"匹配峰 NCC {disp['ncc']:.2f} < {C.DISPARITY_NCC_MIN}，"
                "右眼对应标志不可信（疑似被编码压坏）")
        if disp["error"] > C.DISPARITY_TOL_PX:
            decision["reasons"].append(
                f"实测视差 {disp['measured']:.1f}px 与几何真值 "
                f"{disp['truth']:.1f}px 误差 {disp['error']:.1f}px 超差")

    # 字幕落于安全区下沿居中，左右共用同一行坐标（校正后行不变）
    sx0, sy0, sx1, sy1 = C.SAFE_BOX
    base_u = (sx0 + sx1) // 2
    sub_v = sy1 - 14
    if disp is not None and disp["measured"] is not None:
        u_l = base_u
        u_r = int(round(base_u - disp["measured"]))
    else:
        u_l, u_r = base_u, base_u
    decision["subtitle_xy_left"] = [u_l, sub_v]
    decision["subtitle_xy_right"] = [u_r, sub_v]
    if not (in_safe_box(u_l, sub_v) and in_safe_box(u_r, sub_v)):
        decision["reasons"].append("字幕落点越出双眼共用安全区")
    decision["accepted"] = not decision["reasons"]
    return decision
