"""两路真实码流的格点级 JPEG 编解码与速率-失真模型。

每只眼睛的每一帧被切成独立 JPEG 格点，真实调用编码器产生真实包；
包大小（字节）与解码失真（MSE）均为实测值，预算器只在实测点之间插值。
"""

import io

import numpy as np
from PIL import Image

from . import config as C


def encode_tile(tile, quality):
    """编码一个格点，返回真实 JPEG 字节包。"""
    im = Image.fromarray(tile, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), subsampling=0)
    return buf.getvalue()


def decode_tile(packet):
    """解码 JPEG 包，重建格点灰度数组。"""
    return np.asarray(Image.open(io.BytesIO(packet)).convert("L"),
                      dtype=np.uint8)


def build_frame(tiles):
    """把格点数组拼回整帧。"""
    rows = []
    for r in range(C.ROWS):
        rows.append(np.hstack(tiles[r]))
    return np.vstack(rows)


def split_frame(img):
    return [[img[r * C.TILE:(r + 1) * C.TILE,
                 c * C.TILE:(c + 1) * C.TILE]
             for c in range(C.COLS)] for r in range(C.ROWS)]


def probe_tile(tile):
    """在探针质量点上实测 (字节数, MSE)，返回字典 quality -> (bytes, mse)。"""
    out = {}
    base = tile.astype(np.float64)
    for q in C.PROBE_QUALITIES:
        pkt = encode_tile(tile, q)
        rec = decode_tile(pkt).astype(np.float64)
        out[q] = (len(pkt), float(np.mean((rec - base) ** 2)))
    return out


def model_curve(probe):
    """把探针点整理成可插值的速率-失真曲线（质量升序）。"""
    qs = sorted(probe)
    sizes = np.array([probe[q][0] for q in qs], dtype=np.float64)
    mses = np.array([probe[q][1] for q in qs], dtype=np.float64)
    return np.array(qs, dtype=np.float64), sizes, mses


def build_rd_models(data):
    """对全部眼/帧/格点做探针编码，得到实测 RD 模型。"""
    models = [[[None] * C.COLS for _ in range(C.ROWS)]
              for _ in range(C.N_FRAMES * 2)]
    for eye in range(2):
        for f in range(C.N_FRAMES):
            tiles = split_frame(data["images"][eye][f])
            for r in range(C.ROWS):
                for c in range(C.COLS):
                    models[eye * C.N_FRAMES + f][r][c] = model_curve(
                        probe_tile(tiles[r][c]))
    return models


def encode_with_qualities(img, qualities):
    """按给定格点质量矩阵真实编码整帧。

    返回 packets[r][c]、重建整帧、格点实测字节与 MSE 矩阵。
    """
    tiles = split_frame(img)
    packets = [[None] * C.COLS for _ in range(C.ROWS)]
    rec_tiles = [[None] * C.COLS for _ in range(C.ROWS)]
    sizes = np.zeros((C.ROWS, C.COLS))
    mses = np.zeros((C.ROWS, C.COLS))
    for r in range(C.ROWS):
        for c in range(C.COLS):
            pkt = encode_tile(tiles[r][c], int(qualities[r, c]))
            rec = decode_tile(pkt)
            packets[r][c] = pkt
            rec_tiles[r][c] = rec
            sizes[r, c] = len(pkt)
            base = tiles[r][c].astype(np.float64)
            mses[r, c] = float(np.mean((rec.astype(np.float64) - base) ** 2))
    return packets, build_frame(rec_tiles), sizes, mses
