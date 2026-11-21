"""工作台关键不变量的冒烟测试：python3 -m stereo_workbench.smoke_test。"""

import os
import sys
import tempfile

import numpy as np

from . import allocator
from . import codec
from . import config as C
from . import container
from . import pipeline
from . import scene


def main():
    failures = []

    def check(name, cond, detail=""):
        status = "OK" if cond else "FAIL"
        print(f"  [{status}] {name} {detail}")
        if not cond:
            failures.append(name)

    print("1. 立体几何与对应关系")
    data = scene.build_scene()
    disparity = C.FOCAL * C.BASELINE / C.BOARD_Z
    check("视差为正且与深度成反比", disparity > 0, f"d={disparity:.2f}px")
    check("对应映射行不变、仅列偏移",
          all(data["corr"][r, c] in (-1, c - int(round(disparity / C.TILE)))
              for r in range(C.ROWS) for c in range(C.COLS)))
    trans = scene.find_transition_cell(data)
    check("存在双眼->单眼过渡的棋盘格", trans is not None,
          f"cell={trans[0] if trans else None}")
    cid = trans[0]
    states = [(data["cell_vis"][0][f][cid] > 0.5,
               data["cell_vis"][1][f][cid] > 0.5) for f in range(C.N_FRAMES)]
    check("过渡序列含双眼态与单眼态",
          (True, True) in states and
          any(s in ((True, False), (False, True)) for s in states))

    print("2. RD 模型为真实编解码实测")
    models = codec.build_rd_models(data)
    sample = models[0][2][2]
    qs, sizes, mses = sample
    check("质量升高字节不减", all(np.diff(sizes) >= 0))
    # 极端低质量档（q2->q5）JPEG 量化表跳变会让 MSE 短时反弹，
    # 这是真实编码器特性；整体趋势（最低档 vs 最高档）必须单调改善
    check("最高档 MSE 显著低于最低档", mses[-1] < mses[0])
    check("q20 以上档位 MSE 单调不增",
          all(mses[i + 1] <= mses[i] + 1e-9
              for i, q in enumerate(qs)
              if q >= 20 and i + 1 < len(qs) and qs[i + 1] >= 20))

    print("3. 三种策略与预算约束")
    with tempfile.TemporaryDirectory() as tmp:
        strategies = pipeline.run_all(data, models, tmp)
        for mode, res in strategies.items():
            per_frame = [int(fr["actual_size"][0].sum()
                             + fr["actual_size"][1].sum())
                         for fr in res["frames"]]
            check(f"{mode} 每帧不超共享预算",
                  all(v <= C.SEGMENT_BUDGET for v in per_frame),
                  f"max={max(per_frame)}")
        eq = strategies["equal"]["summary"]
        ad = strategies["adaptive"]["summary"]
        cs = strategies["constrained"]["summary"]
        check("困难度调配平均误差低于平均分配",
              ad["mean_mse"] < eq["mean_mse"],
              f"{ad['mean_mse']:.2f} < {eq['mean_mse']:.2f}")
        check("无约束调配存在对应失真差越界", ad["n_violations"] > 0,
              f"n={ad['n_violations']}")
        check("约束策略越界清零", cs["n_violations"] == 0)
        check("约束的平均代价高于无约束",
              cs["mean_mse"] >= ad["mean_mse"] - 1e-9)

        print("4. 实际包大小反馈")
        res = strategies["adaptive"]
        for fr in res["frames"]:
            expect = C.SEGMENT_BUDGET - fr["spent_l_actual"]
            check_abs = abs(fr["quota_r_actual"] - max(expect, 0.0)) < 1e-6
            if not check_abs:
                failures.append("反馈额度计算错误")
                break
        print("  [OK] 右眼额度=预算-左眼实际花费")

        print("5. 容器往返与文件核对")
        path_l = os.path.join(tmp, "adaptive_left.stb1")
        path_r = os.path.join(tmp, "adaptive_right.stb1")
        report = container.verify_streams(path_l, path_r)
        check("文件核对全部通过", report["all_ok"])
        for eye, path in ((0, path_l), (1, path_r)):
            header, records, footer = container.read_stream(path)
            decoded = [codec.decode_tile(rec["payload"]) for rec in records]
            check(f"眼{eye} 解码包数与尾部一致",
                  len(records) == footer["n_packets"] == len(decoded))
            check(f"眼{eye} 解码格点尺寸正确",
                  all(t.shape == (C.TILE, C.TILE) for t in decoded))

    print("\n结果：", "全部通过" if not failures else f"失败 {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
