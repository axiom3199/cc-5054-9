"""立体预览点选：自包含 HTML + 争议区域回查。

无显示环境下无法弹窗口，因此导出一个内嵌 base64 图像的 HTML：
创作者在左右任一帧上拖出矩形争议区域，页面给出坐标 JSON；
把坐标粘回 `inspect-region` 子命令即可对两套预算码流做区域级回查，
区分误差来自预算分配还是天然遮挡。
"""

import base64
import io
import json
import os

import numpy as np
from PIL import Image

from . import annotations as A
from . import config as C

HTML_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>立体预览点选 - {tag}</title>
<style>
body{{font-family:sans-serif;margin:18px;background:#1b1b1b;color:#eee}}
.row{{display:flex;gap:18px;flex-wrap:wrap}}
figure{{margin:0}}
canvas{{background:#000;cursor:crosshair;border:1px solid #555}}
.safe{{outline:2px solid #39d353;outline-offset:-2px}}
pre{{background:#111;padding:10px;border-radius:6px;white-space:pre-wrap}}
button{{padding:6px 14px;margin-right:8px}}
</style></head><body>
<h2>立体预览（{tag}）帧选择 <select id="frame"></select></h2>
<p>绿色框为双眼共用字幕安全区。在任一帧上拖拽框选争议区域，
松开后把下方 JSON 粘贴给 <code>inspect-region</code> 命令。</p>
<div class="row">
<figure><figcaption>左眼</figcaption><canvas id="L" width="{w}" height="{h}"></canvas></figure>
<figure><figcaption>右眼</figcaption><canvas id="R" width="{w}" height="{h}"></canvas></figure>
</div>
<p><button onclick="clearSel()">清除框选</button>
<button onclick="copyJson()">复制坐标 JSON</button></p>
<pre id="out">（尚未框选）</pre>
<script>
const frames = {frames};
const safe = {safe};
const sel = document.getElementById('frame');
frames.forEach((_, i) => sel.add(new Option('帧 ' + i, i)));
let rect = null, drag = null;
function draw(){{
  for (const [eye, cv] of [['L', 'L'], ['R', 'R']].entries()) {{
    const c = document.getElementById(cv);
    const ctx = c.getContext('2d');
    const img = new Image();
    img.onload = () => {{
      ctx.drawImage(img, 0, 0);
      ctx.strokeStyle = '#39d353'; ctx.lineWidth = 1.5;
      ctx.strokeRect(safe[0], safe[1], safe[2]-safe[0], safe[3]-safe[1]);
      if (rect) {{
        ctx.strokeStyle = '#ffd60a'; ctx.lineWidth = 2;
        ctx.strokeRect(rect.x0, rect.y0, rect.x1-rect.x0, rect.y1-rect.y0);
      }}
    }};
    img.src = frames[sel.value][eye];
  }}
}}
function start(e){{
  const r = e.target.getBoundingClientRect();
  drag = {{x: e.clientX-r.left, y: e.clientY-r.top, eye: e.target.id}};
}}
function end(e){{
  if (!drag) return;
  const r = e.target.getBoundingClientRect();
  const x = e.clientX-r.left, y = e.clientY-r.top;
  rect = {{eye: drag.eye, x0: Math.round(Math.min(drag.x,x)),
           y0: Math.round(Math.min(drag.y,y)),
           x1: Math.round(Math.max(drag.x,x)),
           y1: Math.round(Math.max(drag.y,y)), frame: Number(sel.value)}};
  drag = null; emit(); draw();
}}
function emit(){{
  document.getElementById('out').textContent = JSON.stringify(rect);
}}
function clearSel(){{ rect = null; document.getElementById('out').textContent='（尚未框选）'; draw(); }}
function copyJson(){{ navigator.clipboard.writeText(document.getElementById('out').textContent); }}
for (const id of ['L','R']) {{
  const c = document.getElementById(id);
  c.addEventListener('mousedown', start);
  c.addEventListener('mouseup', end);
}}
sel.addEventListener('change', draw);
draw();
</script></body></html>"""


def _png_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(
        buf.getvalue()).decode("ascii")


def build_preview(data, out_dir, tag="normal"):
    """导出内嵌左右源片与安全区的自包含点选 HTML。"""
    frames = [[_png_b64(data["images"][0][f]),
               _png_b64(data["images"][1][f])]
              for f in range(C.N_FRAMES)]
    html = HTML_TEMPLATE.format(
        frames=json.dumps(frames), safe=json.dumps(C.SAFE_BOX),
        w=C.WIDTH, h=C.HEIGHT, tag=tag)
    path = os.path.join(out_dir, f"stereo_preview_{tag}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def inspect_region(data, runs, region):
    """对创作者点选的区域做两套饥饿档区域级归因。

    region: {frame, eye, x0, y0, x1, y1}
    输出区域在 A/B 档的 MSE、字节、涉及标注的几何可见性与检测评分，
    据此判断是预算分配差异还是天然遮挡。
    """
    f = region["frame"]
    eye = region["eye"]
    x0, y0, x1, y1 = (region["x0"], region["y0"],
                      region["x1"], region["y1"])
    sl = (slice(y0, y1), slice(x0, x1))
    base = data["images"][eye][f][sl].astype(np.float64)
    out = {"region": region, "profiles": {}, "annotations": []}
    for name in ("starve_swapA", "starve_swapB", "generous"):
        run = runs[name]
        # 从落盘容器重新解码，保证看的是实际输出
        from . import container, codec
        path = run["files"][eye]
        _, records, _ = container.read_stream(path)
        recon = np.zeros((C.HEIGHT, C.WIDTH), dtype=np.uint8)
        for rec in records:
            if rec["frame"] != f:
                continue
            tile = codec.decode_tile(rec["payload"])
            r, c = rec["row"], rec["col"]
            recon[r * C.TILE:(r + 1) * C.TILE,
                  c * C.TILE:(c + 1) * C.TILE] = tile
        region_mse = float(np.mean((recon[sl].astype(np.float64) - base) ** 2))
        # 区域覆盖格点的实际字节（编码帧记录与落盘码流同源）
        r0, r1 = y0 // C.TILE, (y1 - 1) // C.TILE
        c0, c1 = x0 // C.TILE, (x1 - 1) // C.TILE
        bytes_in = 0
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                bytes_in += int(run["frame_sizes"][f][eye][rr][cc])
        out["profiles"][name] = {"region_mse": round(region_mse, 3),
                                 "region_bytes": bytes_in}
    for a in data["annots"]:
        geo = A.annot_visibility(a, f, eye, a["draw_mask"],
                                 data.get("swap", False))
        entry = {"id": a["id"], "geo_visible_ratio": round(geo, 3),
                 "stamped_in_eye": eye in a["eyes"]}
        if eye in a["eyes"]:
            from . import markers
            entry["scores"] = {
                name: round(markers.detect_score(
                    _decode_frame(runs[name]["files"][eye], f),
                    a, f, eye,
                    source_frame=data["images"][eye][f],
                    swap=data.get("swap", False))["score"], 3)
                for name in runs}
        out["annotations"].append(entry)
    a_mse = out["profiles"]["starve_swapA"]["region_mse"]
    b_mse = out["profiles"]["starve_swapB"]["region_mse"]
    if abs(a_mse - b_mse) < 0.5:
        out["verdict"] = "两档误差接近：若该眼几何不可见则为天然遮挡，否则与预算无关"
    else:
        worse = "swapA" if a_mse > b_mse else "swapB"
        out["verdict"] = (f"两档误差差异显著（{worse} 更差）："
                          "误差随预算对调移动，属于分配不当而非天然遮挡")
    return out


def _decode_frame(path, frame_idx):
    from . import codec, container
    _, records, _ = container.read_stream(path)
    frame = np.zeros((C.HEIGHT, C.WIDTH), dtype=np.uint8)
    for rec in records:
        if rec["frame"] != frame_idx:
            continue
        tile = codec.decode_tile(rec["payload"])
        r, c = rec["row"], rec["col"]
        frame[r * C.TILE:(r + 1) * C.TILE,
              c * C.TILE:(c + 1) * C.TILE] = tile
    return frame
