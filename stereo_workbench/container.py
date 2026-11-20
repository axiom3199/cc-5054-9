"""STB1 双目码流容器：把两路真实 JPEG 包写成两个文件，并支持独立核对。

文件布局（小端）：
  magic "STB1" | version u8 | eye u8 | fps u16 | n_frames u16
  tile u8 | width u16 | height u16
  然后逐帧逐格点记录：
    timestamp_ms u32 | frame u16 | row u8 | col u8 | payload_len u32 | JPEG 字节
  文件尾：total_packets u32 | total_payload_bytes u64 | magic "END1"
"""

import struct

from . import config as C

HEADER = struct.Struct("<4sBBHHHHH")
META = struct.Struct("<IHBBI")
FOOTER = struct.Struct("=IQ4s")  # u32 包数 + u64 总字节 + END1，禁用对齐


def write_stream(path, eye, packets_by_frame):
    """packets_by_frame[f][r][c] 为真实 JPEG 字节。返回 (包数, 总载荷字节)。"""
    n_packets = 0
    total = 0
    with open(path, "wb") as fh:
        fh.write(HEADER.pack(C.CONTAINER_MAGIC, 1, eye, C.FPS, C.N_FRAMES,
                             C.TILE, C.WIDTH, C.HEIGHT))
        for f, grid in enumerate(packets_by_frame):
            ts = round(1000.0 * f / C.FPS)
            for r in range(C.ROWS):
                for c in range(C.COLS):
                    pkt = grid[r][c]
                    fh.write(META.pack(ts, f, r, c, len(pkt)))
                    fh.write(pkt)
                    n_packets += 1
                    total += len(pkt)
        fh.write(FOOTER.pack(int(n_packets), int(total), b"END1"))
    return int(n_packets), int(total)


def read_stream(path):
    """独立读取容器，返回头部信息、记录列表与尾部汇总，不依赖编码侧内存。"""
    with open(path, "rb") as fh:
        raw = fh.read()
    offset = 0
    magic, version, eye, fps, n_frames, tile, width, height = HEADER.unpack_from(
        raw, offset)
    offset += HEADER.size
    if magic != C.CONTAINER_MAGIC:
        raise ValueError(f"非法魔标: {magic!r}")
    expected = n_frames * C.ROWS * C.COLS
    records = []
    for _ in range(expected):
        ts, f, r, c, plen = META.unpack_from(raw, offset)
        offset += META.size
        payload = raw[offset:offset + plen]
        offset += plen
        records.append({"ts": ts, "frame": f, "row": r, "col": c,
                        "payload": payload})
    n_packets, total, end_magic = FOOTER.unpack_from(raw, offset)
    if end_magic != b"END1":
        raise ValueError("尾部魔标损坏")
    header = {"version": version, "eye": eye, "fps": fps,
              "n_frames": n_frames, "tile": tile, "width": width,
              "height": height}
    footer = {"n_packets": n_packets, "total_payload_bytes": total}
    return header, records, footer


def verify_streams(path_l, path_r):
    """从两个生成文件核对时间戳、帧数及最终总量。"""
    hl, rec_l, foot_l = read_stream(path_l)
    hr, rec_r, foot_r = read_stream(path_r)
    report = {"left": {"header": hl, "footer": foot_l},
              "right": {"header": hr, "footer": foot_r}, "checks": []}

    def check(name, ok, detail):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    check("左眼魔标/版本", hl["version"] == 1, str(hl))
    check("右眼魔标/版本", hr["version"] == 1, str(hr))
    check("帧数一致", hl["n_frames"] == hr["n_frames"] == C.N_FRAMES,
          f"{hl['n_frames']} vs {hr['n_frames']} vs {C.N_FRAMES}")
    check("眼别标记", hl["eye"] == 0 and hr["eye"] == 1,
          f"left={hl['eye']} right={hr['eye']}")
    check("包数=帧数x格点数",
          foot_l["n_packets"] == foot_r["n_packets"]
          == C.N_FRAMES * C.ROWS * C.COLS,
          f"{foot_l['n_packets']}, {foot_r['n_packets']}")
    check("载荷总量与尾部一致",
          foot_l["total_payload_bytes"] == sum(len(x["payload"]) for x in rec_l)
          and foot_r["total_payload_bytes"] == sum(
              len(x["payload"]) for x in rec_r),
          f"{foot_l['total_payload_bytes']} / {foot_r['total_payload_bytes']}")

    ts_l = {(x["frame"], x["row"], x["col"]): x["ts"] for x in rec_l}
    ts_r = {(x["frame"], x["row"], x["col"]): x["ts"] for x in rec_r}
    synced = all(ts_l[k] == ts_r[k] for k in ts_l) and ts_l.keys() == ts_r.keys()
    expected_ts = {k: round(1000.0 * k[0] / C.FPS) for k in ts_l}
    check("左右时间戳严格同步", synced, "同一帧/格点时间戳相等")
    check("时间戳等于帧号/FPS", ts_l == expected_ts, str(sorted(set(ts_l.values()))))

    frames_l = sorted({x["frame"] for x in rec_l})
    check("帧号连续 0..N-1", frames_l == list(range(C.N_FRAMES)), str(frames_l))
    report["combined_total_bytes"] = (foot_l["total_payload_bytes"]
                                      + foot_r["total_payload_bytes"])
    report["all_ok"] = all(c["ok"] for c in report["checks"])
    return report
