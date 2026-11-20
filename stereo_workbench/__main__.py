"""双目影像联合编码工作台 —— 演示入口。

流程：
  1. 生成栏杆后经过的棋盘双目场景（含双眼可见 -> 单眼可见的纹理）；
  2. 对真实 JPEG 格点探针编码，建立实测速率-失真模型；
  3. 三种预算策略逐段分配，左眼实际包大小反馈为右眼可用额度；
  4. 导出两路真实 STB1 码流并解码重建；
  5. 从两个生成文件独立核对时间戳、帧数、总量；
  6. 输出对比报告，并安排/定位“平均误差下降但对应失真差越界”的情形；
  7. 图形窗口（有显示则交互，否则导出快照）。
"""

import json
import os
import sys

import numpy as np

from . import allocator
from . import codec
from . import config as C
from . import container
from . import decode_check
from . import pipeline
from . import scene
from . import viewer

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "stereo_workbench", "output")


def _print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main(interactive=False):
    os.makedirs(OUT, exist_ok=True)
    _print_header("1. 生成双目场景：棋盘从栏杆后经过")
    data = scene.build_scene()
    trans = scene.find_transition_cell(data)
    cid, _ = trans
    print(f"过渡棋盘格 cell_id={cid} 的可见性（L/R，1=可见）：")
    for f in range(C.N_FRAMES):
        l = data["cell_vis"][0][f][cid]
        r = data["cell_vis"][1][f][cid]
        tag = "双眼" if l > 0.5 and r > 0.5 else ("单眼" if l > 0.5 or r > 0.5
                                                  else "遮挡")
        print(f"  帧{f:2d} t={1000*f/C.FPS:5.0f}ms  L={l:.2f} R={r:.2f}  {tag}")

    _print_header("2. 真实 JPEG 探针编码，建立实测 RD 模型")
    models = codec.build_rd_models(data)
    print(f"已对 2眼 x {C.N_FRAMES}帧 x {C.ROWS}x{C.COLS} 格点 x "
          f"{len(C.PROBE_QUALITIES)} 质量档完成真实编解码。")

    _print_header("3-4. 三种联合预算策略 -> 真实编码/解码 -> 导出两路码流")
    strategies = pipeline.run_all(data, models, OUT)
    for mode, res in strategies.items():
        s = res["summary"]
        print(f"\n[{viewer.MODE_LABEL[mode]}] {mode}")
        print(f"  文件: {os.path.basename(res['path_left'])}, "
              f"{os.path.basename(res['path_right'])}")
        print(f"  实际总量 {s['bytes_total']}B "
              f"(左眼 {s['bytes_l']}B / 右眼 {s['bytes_r']}B)，"
              f"共享预算/段 {C.SEGMENT_BUDGET}B x {C.N_FRAMES} 段")
        print(f"  可见区域平均 MSE: 左 {s['mean_mse_l']:.2f}  "
              f"右 {s['mean_mse_r']:.2f}  双眼合计 {s['mean_mse']:.2f}")
        print(f"  对应失真差越界对数: {s['n_violations']} "
              f"(上限 {C.DISTORTION_DIFF_LIMIT})")

    _print_header("5. 从两个生成文件独立核对（时间戳/帧数/总量）")
    verify = container.verify_streams(
        strategies["adaptive"]["path_left"],
        strategies["adaptive"]["path_right"])
    for chk in verify["checks"]:
        print(f"  [{'OK' if chk['ok'] else 'FAIL'}] {chk['name']}: {chk['detail']}")
    print(f"  两文件载荷合计 {verify['combined_total_bytes']}B，"
          f"全部通过={verify['all_ok']}")

    _print_header("6. 平均分配 vs 按困难度调配，及越界情形")
    eq = strategies["equal"]["summary"]
    ad = strategies["adaptive"]["summary"]
    cs = strategies["constrained"]["summary"]
    drop = eq["mean_mse"] - ad["mean_mse"]
    print(f"  平均分配   平均MSE={eq['mean_mse']:.2f} 越界={eq['n_violations']}")
    print(f"  困难度调配 平均MSE={ad['mean_mse']:.2f} 越界={ad['n_violations']}")
    print(f"  约束调配   平均MSE={cs['mean_mse']:.2f} 越界={cs['n_violations']}")
    print(f"  -> 困难度调配相对平均分配，平均误差下降 {drop:.2f} MSE"
          f"（{eq['mean_mse']:.2f} -> {ad['mean_mse']:.2f}）")
    if ad["n_violations"]:
        v0 = ad["violations"][0]
        print("  -> 已安排出越界情形（无约束调配）：")
        print(f"     帧{v0['frame']} 格点 L({v0['row']},{v0['col_l']}) / "
              f"R({v0['row']},{v0['col_r']})  "
              f"MSE {v0['mse_l']:.2f} vs {v0['mse_r']:.2f}，"
              f"差 {v0['diff']:.2f} > {C.DISTORTION_DIFF_LIMIT}")
        print("     约束策略将成对格点绑定升级，消除越界（见 constrained 结果）。")
    else:
        print("  （当前预算下无自然越界，见下方定向构造检查）")

    # 反馈机制示例：打印一帧左右眼额度与实际花费
    fr = strategies["adaptive"]["frames"][6]
    print("\n  实际包大小反馈示例（帧6）：")
    print(f"    左眼实际花费 {int(fr['actual_size'][0].sum())}B，"
          f"右眼可用额度=预算-左眼={int(fr['quota_r_actual'])}B，"
          f"右眼实际花费 {int(fr['actual_size'][1].sum())}B")

    _print_header("6b. 从两个生成文件独立解码重建（脱离编码侧内存）")
    decode_ok = True
    for eye, path in ((0, strategies["adaptive"]["path_left"]),
                      (1, strategies["adaptive"]["path_right"])):
        header, dec_frames, footer = decode_check.decode_stream(path)
        # 与编码侧内存中的重建帧逐像素比对，必须完全一致
        same = all(np.array_equal(dec_frames[f],
                                  strategies["adaptive"]["frames"][f]
                                  ["recon"][eye])
                   for f in range(C.N_FRAMES))
        orig = np.stack(data["images"][eye])
        mse = float(np.mean(
            (np.stack(dec_frames).astype(np.float64)
             - orig.astype(np.float64)) ** 2))
        decode_ok &= same
        print(f"  {'左眼' if eye == 0 else '右眼'}: 解码 {len(dec_frames)} 帧，"
              f"与编码侧重建逐像素一致={same}，对原图整段 MSE={mse:.2f}")
    print(f"  独立解码核对：{'全部通过' if decode_ok else '失败'}")

    report = {
        "config": {"width": C.WIDTH, "height": C.HEIGHT, "tile": C.TILE,
                   "n_frames": C.N_FRAMES, "fps": C.FPS,
                   "budget_per_segment": C.SEGMENT_BUDGET,
                   "distortion_diff_limit": C.DISTORTION_DIFF_LIMIT,
                   "qualities": list(C.PROBE_QUALITIES)},
        "transition_cell": {"cell_id": cid,
                            "visibility": [
                                {"frame": f,
                                 "left": float(data["cell_vis"][0][f][cid]),
                                 "right": float(data["cell_vis"][1][f][cid])}
                                for f in range(C.N_FRAMES)]},
        "summaries": {m: strategies[m]["summary"] for m in strategies},
        "violations_adaptive": ad["violations"],
        "verify": verify,
        "decode_roundtrip_ok": decode_ok,
        "files": {m: {"left": strategies[m]["path_left"],
                      "right": strategies[m]["path_right"]}
                  for m in strategies},
    }
    rp = os.path.join(OUT, "report.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {rp}")

    _print_header("7. 图形窗口")
    if interactive and os.environ.get("DISPLAY"):
        v = viewer.Viewer(data, strategies, verify)
        v.show()
    else:
        snaps = viewer.make_snapshots(data, strategies, verify, OUT)
        print("  当前无显示环境，已导出窗口快照：")
        for p in snaps:
            print(f"    {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(interactive="--interactive" in sys.argv))
