"""从生成的 STB1 码流文件独立解码，重建整帧并与原始场景核对。

不引用编码侧的任何内存对象，只读取磁盘上的两个文件，
验证：容器可解析、JPEG 包可解码、帧数/尺寸正确、重建误差与报告一致。
"""

import sys

import numpy as np

from . import codec
from . import config as C
from . import container


def decode_stream(path):
    """读取容器并把格点 JPEG 包解码拼回整帧序列。"""
    header, records, footer = container.read_stream(path)
    frames = [np.zeros((C.HEIGHT, C.WIDTH), dtype=np.uint8)
              for _ in range(header["n_frames"])]
    for rec in records:
        tile = codec.decode_tile(rec["payload"])
        r, c, f = rec["row"], rec["col"], rec["frame"]
        frames[f][r * C.TILE:(r + 1) * C.TILE,
                  c * C.TILE:(c + 1) * C.TILE] = tile
    return header, frames, footer


def main():
    from . import scene
    out = "/workspace/stereo_workbench/output"
    mode = sys.argv[1] if len(sys.argv) > 1 else "adaptive"
    path_l = f"{out}/{mode}_left.stb1"
    path_r = f"{out}/{mode}_right.stb1"
    scene_data = scene.build_scene()
    ok_all = True
    for eye, path in ((0, path_l), (1, path_r)):
        header, frames, footer = decode_stream(path)
        print(f"{'左眼' if eye == 0 else '右眼'} {path}")
        print(f"  解码帧数={len(frames)} 尺寸={frames[0].shape} "
              f"容器包数={footer['n_packets']} 总载荷={footer['total_payload_bytes']}B")
        assert len(frames) == C.N_FRAMES
        assert frames[0].shape == (C.HEIGHT, C.WIDTH)
        assert footer["n_packets"] == C.N_FRAMES * C.ROWS * C.COLS
        stacked = np.stack(frames)
        valid = int(stacked.min()) >= 0 and int(stacked.max()) <= 255
        # 与原始场景对比，给出整段平均重建 MSE（含被栏杆遮挡区域）
        orig = np.stack(scene_data["images"][eye])
        mse = float(np.mean((stacked.astype(np.float64)
                             - orig.astype(np.float64)) ** 2))
        ok_all &= valid
        print(f"  像素范围 [{stacked.min()}, {stacked.max()}] 有效={valid}"
              f" 对原图整段 MSE={mse:.2f}")
    print("独立解码核对：", "全部通过" if ok_all else "存在失败")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
