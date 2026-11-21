"""景深布景与编码练习的端到端编排。

所有判断都基于“实际编码 -> 写盘 -> 独立解码 -> 重读码流”的图像，
不引用编码侧内存里的重建帧，确保练习结论来自真实输出而非模型插值。
"""

import json
import os

import numpy as np
from PIL import Image

from . import annotations as A
from . import codec
from . import config as C
from . import container
from . import markers
from . import profiles
from . import scene


def _save_gray(path, img):
    Image.fromarray(img, mode="L").save(path)


def export_assets(data, out_dir):
    """导出左右源片 PNG、单眼可见掩码 NPZ、对应关系与可见时段 JSON。"""
    asset_dir = os.path.join(out_dir, "assets")
    os.makedirs(asset_dir, exist_ok=True)
    paths = {"frames": [[], []], "masks_npz": [], "visibility_json": ""}
    for eye, name in ((0, "left"), (1, "right")):
        eye_dir = os.path.join(asset_dir, name)
        os.makedirs(eye_dir, exist_ok=True)
        for f in range(C.N_FRAMES):
            p = os.path.join(eye_dir, f"{name}_f{f:02d}.png")
            _save_gray(p, data["images"][eye][f])
            paths["frames"][eye].append(p)

    # 仅单眼可见掩码：micro 的几何真值（该眼可见且另一眼不可见）
    mono = {
        "left_only": np.zeros((C.N_FRAMES, C.HEIGHT, C.WIDTH), dtype=np.uint8),
        "right_only": np.zeros((C.N_FRAMES, C.HEIGHT, C.WIDTH), dtype=np.uint8),
        "both_visible_left": np.zeros((C.N_FRAMES, C.HEIGHT, C.WIDTH),
                                      dtype=np.uint8),
        "both_visible_right": np.zeros((C.N_FRAMES, C.HEIGHT, C.WIDTH),
                                       dtype=np.uint8),
    }
    for f in range(C.N_FRAMES):
        ml = data["annot_visible"][0]["micro"][f]
        mr = data["annot_visible"][1]["micro"][f]
        mono["left_only"][f] = (ml & ~mr).astype(np.uint8)
        mono["right_only"][f] = (mr & ~ml).astype(np.uint8)
        mono["both_visible_left"][f] = data["both_visible"][0][f].repeat(
            C.TILE, axis=0).repeat(C.TILE, axis=1).astype(np.uint8)
        mono["both_visible_right"][f] = data["both_visible"][1][f].repeat(
            C.TILE, axis=0).repeat(C.TILE, axis=1).astype(np.uint8)
    mask_path = os.path.join(asset_dir, "mono_visibility_masks.npz")
    np.savez_compressed(mask_path, **mono)
    paths["masks_npz"] = mask_path

    windows = A.visibility_windows(data["annots"], swap=data["swap"])
    vis_payload = {
        "swap": data["swap"],
        "safe_box": list(C.SAFE_BOX),
        "annotations": windows,
        "correspondence": data["corr"].tolist(),
        "frames": [{"frame": f, "t_ms": round(1000.0 * f / C.FPS)}
                   for f in range(C.N_FRAMES)],
    }
    vis_path = os.path.join(asset_dir, "visibility_windows.json")
    with open(vis_path, "w", encoding="utf-8") as fh:
        json.dump(vis_payload, fh, ensure_ascii=False, indent=2)
    paths["visibility_json"] = vis_path
    return paths, vis_payload


def write_and_reread(out_dir, tag, result):
    """把两路码流写成 STB1，再像第三方一样从文件独立解码回来。"""
    stream_dir = os.path.join(out_dir, "streams")
    os.makedirs(stream_dir, exist_ok=True)
    files = []
    for eye, name in ((0, "left"), (1, "right")):
        p = os.path.join(stream_dir, f"{tag}_{name}.stb1")
        container.write_stream(p, eye, result["packets"][eye])
        files.append(p)
    decoded = []
    for p in files:
        _, frames, _ = container.read_stream(p)
        recon = [np.zeros((C.HEIGHT, C.WIDTH), dtype=np.uint8)
                 for _ in range(C.N_FRAMES)]
        for rec in frames:
            tile = codec.decode_tile(rec["payload"])
            r, c, f = rec["row"], rec["col"], rec["frame"]
            recon[f][r * C.TILE:(r + 1) * C.TILE,
                     c * C.TILE:(c + 1) * C.TILE] = tile
        decoded.append(recon)
    return files, decoded


def marker_audit(data, decoded, annots, swap):
    """对每个标注、每只眼、每帧从解码图像重检评分，并对照几何可见性。"""
    audit = {}
    for a in annots:
        per_eye = {}
        for eye in range(2):
            rows = []
            for f in range(C.N_FRAMES):
                if eye not in a["eyes"]:
                    det = {"score": 0.0, "contrast": 0.0, "ncc": 0.0,
                           "present": False}
                else:
                    det = markers.detect_score(
                        decoded[eye][f], a, f, eye,
                        source_frame=data["images"][eye][f], swap=swap)
                geo = A.annot_visibility(a, f, eye, a["draw_mask"], swap)
                rows.append({
                    "frame": f,
                    "geo_visible_ratio": round(geo, 3),
                    "geo_visible": geo >= C.ANNOT_VIS_RATIO,
                    "stamped": eye in a["eyes"],
                    "score": round(det["score"], 3),
                    "contrast": round(det["contrast"], 2),
                    "ncc": round(det["ncc"], 3),
                })
            per_eye[eye] = rows
        audit[a["id"]] = {"kind": a["kind"], "eyes": per_eye}
    return audit


def subtitle_run(data, decoded, annots, swap, reference_id="soft"):
    """逐帧运行字幕对齐器，记录落位与拒绝原因。"""
    decisions = []
    for f in range(C.N_FRAMES):
        d = markers.place_subtitle(
            decoded[0][f], decoded[1][f], annots, f,
            reference_id=reference_id, swap=swap,
            source_l=data["images"][0][f], source_r=data["images"][1][f])
        decisions.append(d)
    return decisions


def _classify_micro(audit_entry):
    """把微编号在每帧每眼的现象归类为四种互斥原因。"""
    out = []
    rows_l = {r["frame"]: r for r in audit_entry["eyes"][0]}
    rows_r = {r["frame"]: r for r in audit_entry["eyes"][1]}
    for f in range(C.N_FRAMES):
        for eye, row in ((0, rows_l[f]), (1, rows_r[f])):
            if not row["stamped"]:
                cause = "not_an_asset_eye"
            elif not row["geo_visible"]:
                cause = "natural_occlusion"
            elif row["score"] < C.MARKER_SCORE_MIN:
                cause = "coding_loss"
            else:
                cause = "readable"
            out.append({"frame": f, "eye": eye, "cause": cause,
                        "score": row["score"],
                        "geo_visible": row["geo_visible"]})
    return out


def budget_swap_experiment(data, models, out_dir, tag_prefix, swap):
    """三套预算档案真实编码 + 重读码流 + 标志审计 + 字幕决策。"""
    runs = {}
    annots = data["annots"]
    for profile_name in profiles.PROFILES:
        result = profiles.encode_profile(data, models, profile_name)
        tag = f"{tag_prefix}_{profile_name}"
        files, decoded = write_and_reread(out_dir, tag, result)
        audit = marker_audit(data, decoded, annots, swap)
        decisions = subtitle_run(data, decoded, annots, swap,
                                 reference_id="soft")
        # 作者更换参考物：软标志被压坏时改用微编号（仅在它双眼可见的帧）
        decisions_micro = subtitle_run(data, decoded, annots, swap,
                                       reference_id="micro")
        runs[profile_name] = {
            "files": files,
            "bytes_total": result["bytes_total"],
            "floored_frames": result["floored_frames"],
            "frame_sizes": [
                [fr["actual_size"][0].tolist(),
                 fr["actual_size"][1].tolist()]
                for fr in result["frames"]],
            "frame_qualities": [
                [fr["q"][0].tolist(), fr["q"][1].tolist()]
                for fr in result["frames"]],
            "audit": audit,
            "micro_classification": _classify_micro(audit["micro"]),
            "subtitle_by_soft": decisions,
            "subtitle_by_micro": decisions_micro,
            "n_subtitle_rejected_soft": sum(
                not d["accepted"] for d in decisions),
            "n_subtitle_rejected_micro": sum(
                not d["accepted"] for d in decisions_micro),
        }
    return runs


def _swap_attribution(runs):
    """对调 A/B 两套饥饿档后，逐眼逐格归因：分配不当还是天然遮挡。"""
    a, b = runs["starve_swapA"], runs["starve_swapB"]
    notes = []
    for eye, name in ((0, "left"), (1, "right")):
        bytes_a = a["bytes_total"][eye]
        bytes_b = b["bytes_total"][eye]
        notes.append({
            "eye": name,
            "bytes_swapA": bytes_a,
            "bytes_swapB": bytes_b,
            "favored_in": "swapA" if bytes_a > bytes_b else "swapB",
        })
    # 微编号：同一帧在两档之间评分变化，且几何可见性不变 => 分配问题
    micro_compare = []
    for row_a in a["micro_classification"]:
        row_b = next(x for x in b["micro_classification"]
                     if x["frame"] == row_a["frame"] and x["eye"] == row_a["eye"])
        if row_a["cause"] == "natural_occlusion" or row_b["cause"] == "natural_occlusion":
            verdict = "natural_occlusion_unchanged"
        elif row_a["cause"] != row_b["cause"] or row_a["score"] != row_b["score"]:
            verdict = "budget_allocation_effect"
        else:
            verdict = "no_change"
        micro_compare.append({
            "frame": row_a["frame"], "eye": row_a["eye"],
            "cause_A": row_a["cause"], "cause_B": row_b["cause"],
            "score_A": row_a["score"], "score_B": row_b["score"],
            "verdict": verdict,
        })
    return {"eye_bytes": notes, "micro_compare": micro_compare}


def run(out_dir, swap=False):
    """完整练习：素材 -> 三档编码 -> 码流重读 -> 字幕/标志/互换归因。"""
    os.makedirs(out_dir, exist_ok=True)
    annots = A.build_annotations()
    data = scene.build_scene(swap=swap, annots=annots)
    asset_paths, vis_payload = export_assets(data, out_dir)
    models = codec.build_rd_models(data)
    tag = "swap" if swap else "normal"
    runs = budget_swap_experiment(data, models, out_dir, tag, swap)
    attribution = _swap_attribution(runs)

    # 相机互换反转校验在 driver 层做（需要 normal+swap 两份结果）
    report = {
        "camera_swap": swap,
        "assets": {"visibility": asset_paths["visibility_json"],
                   "masks": asset_paths["masks_npz"]},
        "visibility_windows": vis_payload,
        "runs": {
            name: {k: v for k, v in run.items() if k != "files"}
            for name, run in runs.items()
        },
        "stream_files": {name: run["files"] for name, run in runs.items()},
        "budget_swap_attribution": attribution,
    }
    rp = os.path.join(out_dir, f"exercise_report_{tag}.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return report, data, models, runs


def reversal_check(report_normal, report_swap):
    """互换相机后，单眼窗口必须左右反转，视差真值符号必须相反。"""
    checks = []
    wn = report_normal["visibility_windows"]["annotations"]["micro"]["eyes"]
    ws = report_swap["visibility_windows"]["annotations"]["micro"]["eyes"]
    checks.append({
        "name": "微编号单眼窗口左右反转",
        "ok": wn[0]["windows"] == ws[1]["windows"]
        and wn[1]["windows"] == ws[0]["windows"],
        "normal_L": wn[0]["windows"], "normal_R": wn[1]["windows"],
        "swap_L": ws[0]["windows"], "swap_R": ws[1]["windows"],
    })
    cn = np.array(report_normal["visibility_windows"]["correspondence"],
                  dtype=int)
    cs = np.array(report_swap["visibility_windows"]["correspondence"],
                  dtype=int)
    cols = np.arange(C.COLS)
    dn = cn - cols[None, :]
    ds = cs - cols[None, :]
    valid = (cn >= 0) & (cs >= 0)
    sign_ok = bool(np.all(np.sign(dn[valid]) == -np.sign(ds[valid])))
    checks.append({"name": "对应映射列偏移方向反转", "ok": sign_ok,
                   "normal_offsets": dn[0].tolist(),
                   "swap_offsets": ds[0].tolist()})
    return {"all_ok": all(c["ok"] for c in checks), "checks": checks}
