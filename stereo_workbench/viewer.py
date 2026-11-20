"""图形窗口：沿选中格点查看每眼重建、遮挡状态和花掉的字节。

有显示环境时打开可交互窗口（点击任意子图选择格点、方向键换帧、
按策略切换）；无显示环境（headless）时用 Agg 后端导出等价快照 PNG。
"""

import os

import numpy as np

import matplotlib
try:
    import tkinter  # noqa: F401
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from . import config as C

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(_FONT_PATH):
    font_manager.fontManager.addfont(_FONT_PATH)
    _FONT_NAME = font_manager.FontProperties(fname=_FONT_PATH).get_name()
    plt.rcParams["font.family"] = _FONT_NAME
    plt.rcParams["font.monospace"] = [_FONT_NAME, "DejaVu Sans Mono"]
    plt.rcParams["font.sans-serif"] = [_FONT_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODE_LABEL = {"equal": "平均分配", "adaptive": "困难度调配(无约束)",
              "constrained": "困难度调配(带对应约束)"}


class Viewer:
    def __init__(self, data, strategies, verify_report):
        self.data = data
        self.strategies = strategies
        self.verify = verify_report
        self.modes = list(strategies.keys())
        self.mode = self.modes[0]
        self.frame = 0
        self.sel = (C.ROWS // 2, C.COLS // 2)
        self.fig, self.axes = plt.subplots(2, 4, figsize=(15, 8))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()

    # ---------- 交互 ----------
    def _on_click(self, event):
        if event.inaxes is None:
            return
        ax_idx = list(self.axes.flat).index(event.inaxes)
        if ax_idx < 2 and event.xdata is not None and event.ydata is not None:
            c = min(int(event.xdata // C.TILE), C.COLS - 1)
            r = min(int(event.ydata // C.TILE), C.ROWS - 1)
            self.sel = (r, c)
            self._draw()

    def _on_key(self, event):
        if event.key == "right":
            self.frame = (self.frame + 1) % C.N_FRAMES
        elif event.key == "left":
            self.frame = (self.frame - 1) % C.N_FRAMES
        elif event.key in ("up", "down"):
            i = self.modes.index(self.mode)
            step = 1 if event.key == "down" else -1
            self.mode = self.modes[(i + step) % len(self.modes)]
        else:
            return
        self._draw()

    # ---------- 绘制 ----------
    def _draw(self):
        f = self.frame
        r, c = self.sel
        res = self.strategies[self.mode]
        fr = res["frames"][f]
        corr_c = int(self.data["corr"][r, c])
        for ax in self.axes.flat:
            ax.clear()
            ax.axis("off")

        self._eye_panel(0, 0, r, c, f, fr)
        self._eye_panel(1, 1, r, corr_c, f, fr)
        self._occlusion_panel(2, f, r, c, corr_c)
        self._byte_panel(3, f)
        self._detail_panel(4, r, c, corr_c, f, fr)
        self._timeline_panel(5, r, c, corr_c)
        self._compare_panel(6)
        self._verify_panel(7)

        self.fig.suptitle(
            f"策略：{MODE_LABEL[self.mode]}   帧 {f}/{C.N_FRAMES - 1}"
            f"   时间戳 {round(1000 * f / C.FPS)} ms   "
            f"选中格点 L({r},{c}) -> R({r},{corr_c})   "
            "(点击左/右图选格点，←→换帧，↑↓切策略)",
            fontsize=12)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.fig.canvas.draw_idle()

    def _eye_panel(self, eye, ax_idx, r, c, f, fr):
        ax = self.axes.flat[ax_idx]
        ax.imshow(fr["recon"][eye], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{'左眼' if eye == 0 else '右眼'} 重建")
        for rr in range(C.ROWS):
            for cc in range(C.COLS):
                vis = self.data["visible"][eye][f][rr, cc]
                edge = "lime" if vis else "red"
                ax.add_patch(Rectangle(
                    (cc * C.TILE, rr * C.TILE), C.TILE, C.TILE,
                    fill=False, edgecolor=edge, lw=0.7, alpha=0.8))
        ax.add_patch(Rectangle((c * C.TILE, r * C.TILE), C.TILE, C.TILE,
                               fill=False, edgecolor="yellow", lw=2.5))

    def _occlusion_panel(self, ax_idx, f, r, c, corr_c):
        ax = self.axes.flat[ax_idx]
        combo = np.zeros((C.ROWS, C.COLS, 3))
        vl = self.data["visible"][0][f]
        vr = self.data["visible"][1][f]
        for rr in range(C.ROWS):
            for cc in range(C.COLS):
                cr = int(self.data["corr"][rr, cc])
                l = vl[rr, cc]
                rvis = vr[rr, cr] if cr >= 0 else False
                if l and rvis:
                    combo[rr, cc] = (0.2, 0.8, 0.2)       # 双眼可见
                elif l or rvis:
                    combo[rr, cc] = (1.0, 0.65, 0.0)      # 单眼可见
                else:
                    combo[rr, cc] = (0.7, 0.2, 0.2)       # 双眼遮挡
        ax.imshow(combo)
        ax.set_title("可见性：绿=双眼 橙=单眼 红=遮挡")
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                               edgecolor="white", lw=2))

    def _byte_panel(self, ax_idx, f):
        ax = self.axes.flat[ax_idx]
        fr = self.strategies[self.mode]["frames"][f]
        spend = np.concatenate([fr["actual_size"][0].flatten(),
                                fr["actual_size"][1].flatten()])
        grid = np.zeros((C.ROWS, C.COLS * 2))
        grid[:, :C.COLS] = fr["actual_size"][0]
        grid[:, C.COLS:] = fr["actual_size"][1]
        im = ax.imshow(grid, cmap="viridis")
        ax.set_title(f"各格点实际字节 (本帧合计 {int(spend.sum())} / "
                     f"预算 {C.SEGMENT_BUDGET})")
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(C.COLS * 2) - 0.5, minor=True)
        ax.grid(which="minor", color="white", lw=0.5)

    def _detail_panel(self, ax_idx, r, c, corr_c, f, fr):
        ax = self.axes.flat[ax_idx]
        ax.axis("off")
        lines = [f"选中格点  L({r},{c}) -> R({r},{corr_c})"]
        for eye, (rr, cc) in enumerate(((r, c), (r, corr_c))):
            if cc < 0:
                lines.append(f"{'左眼' if eye == 0 else '右眼'}: 超出对应范围")
                continue
            vis = self.data["visible"][eye][f][rr, cc]
            q = fr["q"][eye][rr, cc]
            spent = int(fr["actual_size"][eye][rr, cc])
            mse = fr["actual_mse"][eye][rr, cc]
            occ = self.data["occl"][eye][f][rr, cc]
            hard = self.data["difficulty"][eye][f][rr, cc]
            lines.append(
                f"{'左眼' if eye == 0 else '右眼'} 可见={vis} 遮挡率={occ:.2f}")
            lines.append(f"   质量档={q}  实际包={spent}B  实测MSE={mse:.2f}"
                         f"  困难度={hard:.1f}")
        if corr_c >= 0:
            d = abs(fr["actual_mse"][0][r, c] - fr["actual_mse"][1][r, corr_c])
            tag = "越界!" if d > C.DISTORTION_DIFF_LIMIT else "达标"
            lines.append(f"对应失真差={d:.2f} (上限{C.DISTORTION_DIFF_LIMIT}) {tag}")
        ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=10, transform=ax.transAxes)

    def _timeline_panel(self, ax_idx, r, c, corr_c):
        ax = self.axes.flat[ax_idx]
        frames = np.arange(C.N_FRAMES)
        for eye, cc in ((0, c), (1, corr_c)):
            if cc < 0:
                continue
            spent = [self.strategies[self.mode]["frames"][f]
                     ["actual_size"][eye][r, cc] for f in frames]
            ax.plot(frames, spent, marker="o",
                    label=f"{'左眼' if eye == 0 else '右眼'}字节")
        ax.set_title("选中格点跨时间段花费")
        ax.set_xlabel("帧（同步时间段）")
        ax.set_ylabel("实际字节")
        ax.legend(fontsize=8)

    def _compare_panel(self, ax_idx):
        ax = self.axes.flat[ax_idx]
        ax.axis("off")
        lines = ["三策略总量对比（真实编码）"]
        for m in self.modes:
            s = self.strategies[m]["summary"]
            lines.append(
                f"{MODE_LABEL[m]}:")
            lines.append(
                f"  总 {s['bytes_total']}B (L{s['bytes_l']}/R{s['bytes_r']})"
                f"  平均MSE={s['mean_mse']:.2f}  越界对数={s['n_violations']}")
        ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9, transform=ax.transAxes)

    def _verify_panel(self, ax_idx):
        ax = self.axes.flat[ax_idx]
        ax.axis("off")
        v = self.verify
        lines = ["两个生成文件独立核对"]
        for side in ("left", "right"):
            h = v[side]["header"]
            ft = v[side]["footer"]
            lines.append(f"{side}: 眼{h['eye']} 帧{h['n_frames']} "
                         f"包{ft['n_packets']} 载荷{ft['total_payload_bytes']}B")
        lines.append(f"合计 {v['combined_total_bytes']}B")
        for chk in v["checks"]:
            lines.append(f"[{'OK' if chk['ok'] else 'FAIL'}] {chk['name']}")
        ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=8.5, transform=ax.transAxes)

    def show(self):
        plt.show()

    def snapshot(self, path):
        self.fig.savefig(path, dpi=110, bbox_inches="tight")
        return path


def make_snapshots(data, strategies, verify_report, out_dir,
                   frames=(0, 5, 9)):
    """headless 环境下为几个关键帧导出窗口快照。"""
    paths = []
    v = Viewer(data, strategies, verify_report)
    for f in frames:
        v.frame = f
        v.mode = "adaptive"
        v._draw()
        p = os.path.join(out_dir, f"window_adaptive_f{f}.png")
        v.snapshot(p)
        paths.append(p)
    # 约束策略同帧对照
    v.mode = "constrained"
    v.frame = frames[1]
    v._draw()
    p = os.path.join(out_dir, f"window_constrained_f{frames[1]}.png")
    v.snapshot(p)
    paths.append(p)
    plt.close(v.fig)
    return paths
