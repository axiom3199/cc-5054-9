"""双目联合编码工作台的全局几何与编码参数。"""

# 图像（每只眼睛一帧）
WIDTH = 320
HEIGHT = 256
TILE = 64
COLS = WIDTH // TILE
ROWS = HEIGHT // TILE

# 时间分段（同步时间段：左右眼逐帧一一对应）
N_FRAMES = 12
FPS = 6

# 校正后的平行光轴立体相机（像素单位）
BASELINE = 40.0
FOCAL = 220.0
EYE_X = (-BASELINE / 2.0, BASELINE / 2.0)  # 左眼、右眼的相机 x

# 棋盘：世界坐标 x 从 X0 开始向右平移，深度 Z 固定
BOARD_Z = 180.0
CELL_WORLD = 26.0
N_CELLS = 14
X0_START = -200.0
X0_SPEED = 16.0  # 每帧世界单位，向左向右移动由符号控制

# 前景栏杆（比棋盘更靠近相机），世界坐标下的竖条区间 [x1, x2]
RAIL_Z = 90.0
RAIL_BARS = [(-70.0, -46.0), (18.0, 42.0), (106.0, 130.0)]

# 背景亮度与棋盘灰度
BG_GRAY = 24
DARK_GRAY = 40
LIGHT_GRAY = 215

# 每段（帧）两只眼睛共享的字节预算。
# 定在两眼 q20 基线成本之上约 15%，预算器必须在格点与眼别之间取舍。
SEGMENT_BUDGET = 24000

# JPEG 探针质量点（速率-失真模型的采样点）
PROBE_QUALITIES = (20, 35, 55, 80)
# 编码时允许的实际质量档位（从探针点中选）
QUALITY_GRID = PROBE_QUALITIES

# 对应格点的失真差上限（MSE），两眼均可见时生效
DISTORTION_DIFF_LIMIT = 12.0

# 容器格式魔标与版本
CONTAINER_MAGIC = b"STB1"
