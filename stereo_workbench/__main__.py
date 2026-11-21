"""双目景深布景与编码练习工作台 —— 命令行入口。

子命令：
  assets     生成左右源片、对应关系、单眼可见掩码与可见时段
  exercise   三档预算真实编码，重读码流检查字幕落位/拒绝原因与标志评分
  swap       交换相机位置再生成一次，并校验遮挡与视差整体反转
  preview    导出自包含立体点选 HTML
  inspect    回传争议区域坐标，重读两套码流做区域级归因
  demo       原始三策略 RD 预算演示（保留）
"""

import argparse
import json
import os

from . import annotations as A
from . import config as C
from . import exercises
from . import preview

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(ROOT, "output")


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"  写入 {path}")


def cmd_assets(args):
    from . import scene
    out = os.path.join(args.out, "assets_run")
    os.makedirs(out, exist_ok=True)
    annots = A.build_annotations()
    data = scene.build_scene(swap=args.swap, annots=annots)
    paths, vis = exercises.export_assets(data, out)
    preview.build_preview(data, out, tag="swap" if args.swap else "normal")
    print("标注资产：")
    for a in annots:
        print(f"  {a['id']:<5} {a['kind']:<22} 世界X={a['wx']:.1f} "
              f"尺寸={a['w']}x{a['h']} 绘制眼={a['eyes']}")
    print("可见时段：")
    for aid, entry in vis["annotations"].items():
        for eye in range(2):
            e = entry["eyes"][eye]
            tag = "几何仅参考(不绘制)" if e["geometric_only"] else ""
            print(f"  {aid:<5} 眼{eye} 窗口={e['windows']} {tag}")
    print(f"源片目录 {os.path.join(out, 'assets')}")
    print(f"掩码包   {paths['masks_npz']}")
    return 0


def cmd_exercise(args):
    out = os.path.join(args.out, "exercise_run")
    report, data, models, runs = exercises.run(out, swap=False)
    _print_exercise_summary(report)
    return 0


def cmd_swap(args):
    out = os.path.join(args.out, "exercise_run")
    os.makedirs(out, exist_ok=True)
    report_n, data_n, _, runs_n = exercises.run(out, swap=False)
    report_s, data_s, _, runs_s = exercises.run(out, swap=True)
    reversal = exercises.reversal_check(report_n, report_s)
    _write_json(os.path.join(out, "camera_reversal_check.json"), reversal)
    print("\n相机互换反转校验：")
    for chk in reversal["checks"]:
        print(f"  [{'OK' if chk['ok'] else 'FAIL'}] {chk['name']}")
        if "normal_L" in chk:
            print(f"       正常 L={chk['normal_L']} R={chk['normal_R']}")
            print(f"       互换 L={chk['swap_L']} R={chk['swap_R']}")
    # 互换后微编号在饥饿档下的可读性必须来自重新编码，而非复制旧结果
    print("\n互换后微编号评分（generous 档，重读码流）：")
    for eye in range(2):
        rows = report_s["runs"]["generous"]["audit"]["micro"]["eyes"][eye]
        print(f"  眼{eye}: " + " ".join(
            f"f{r['frame']}={r['score']:.2f}/{'V' if r['geo_visible'] else 'X'}"
            for r in rows))
    return 0 if reversal["all_ok"] else 1


def cmd_preview(args):
    from . import scene
    out = os.path.join(args.out, "preview_run")
    os.makedirs(out, exist_ok=True)
    annots = A.build_annotations()
    data = scene.build_scene(swap=args.swap, annots=annots)
    path = preview.build_preview(data, out,
                                 tag="swap" if args.swap else "normal")
    print(f"立体点选预览：{path}")
    print("浏览器打开后拖拽框选，把 JSON 粘给 inspect 子命令。")
    return 0


def cmd_inspect(args):
    out = os.path.join(args.out, "exercise_run")
    region = json.loads(args.region)
    required = {"frame", "eye", "x0", "y0", "x1", "y1"}
    missing = required - set(region)
    if missing:
        raise SystemExit(f"区域 JSON 缺少字段：{sorted(missing)}")
    # 复用已生成的练习产物（如不存在则先生成）
    report, data, models, runs = exercises.run(out, swap=False)
    result = preview.inspect_region(data, runs, region)
    rp = os.path.join(out, "region_inspection.json")
    _write_json(rp, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _print_exercise_summary(report):
    print("\n各预算档案（总量为重读容器得到的实际字节）：")
    for name, run in report["runs"].items():
        print(f"  {name:<14} L={run['bytes_total'][0]:>7}B "
              f"R={run['bytes_total'][1]:>7}B  "
              f"字幕拒绝(软标志)={run['n_subtitle_rejected_soft']} "
              f"字幕拒绝(微编号)={run['n_subtitle_rejected_micro']} "
              f"压底帧={len(run['floored_frames'])}")
    print("\n微编号四分类（每帧每眼）：")
    for name, run in report["runs"].items():
        causes = {}
        for row in run["micro_classification"]:
            causes.setdefault(row["cause"], 0)
            causes[row["cause"]] += 1
        print(f"  {name:<14} {causes}")
    print("\n软标志拒绝原因示例（starve_swapA 前 3 条拒绝）：")
    n = 0
    for d in report["runs"]["starve_swapA"]["subtitle_by_soft"]:
        if not d["accepted"]:
            print(f"  帧{d['frame']}: " + "；".join(d["reasons"]))
            n += 1
            if n >= 3:
                break
    print("\n预算对调归因（微编号）：")
    for row in report["budget_swap_attribution"]["micro_compare"]:
        if row["verdict"] != "no_change":
            print(f"  帧{row['frame']} 眼{row['eye']} "
                  f"A={row['cause_A']}({row['score_A']}) "
                  f"B={row['cause_B']}({row['score_B']}) -> {row['verdict']}")


def cmd_demo(args):
    from . import pipeline, viewer, container, decode_check
    from . import codec, scene
    out = os.path.join(args.out, "demo_run")
    os.makedirs(out, exist_ok=True)
    data = scene.build_scene()
    models = codec.build_rd_models(data)
    strategies = pipeline.run_all(data, models, out)
    verify = container.verify_streams(
        strategies["adaptive"]["path_left"],
        strategies["adaptive"]["path_right"])
    print("三策略演示完成，容器核对 all_ok =", verify["all_ok"])
    for mode, res in strategies.items():
        s = res["summary"]
        print(f"  {mode:<12} 总{s['bytes_total']}B "
              f"MSE={s['mean_mse']:.2f} 越界={s['n_violations']}")
    viewer.make_snapshots(data, strategies, verify, out)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3 -m stereo_workbench",
        description="双目景深布景与编码预算练习工作台")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出根目录")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("assets", help="生成源片/掩码/对应关系/可见时段")
    p.add_argument("--swap", action="store_true", help="交换相机位置")
    p.set_defaults(func=cmd_assets)

    p = sub.add_parser("exercise", help="三档预算编码并重读码流检查")
    p.set_defaults(func=cmd_exercise)

    p = sub.add_parser("swap", help="交换相机重新生成并校验反转")
    p.set_defaults(func=cmd_swap)

    p = sub.add_parser("preview", help="导出立体点选 HTML")
    p.add_argument("--swap", action="store_true")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("inspect", help="回传争议区域 JSON 做归因")
    p.add_argument("region", help='如 \'{"frame":7,"eye":1,...}\'')
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("demo", help="原始三策略 RD 预算演示")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
