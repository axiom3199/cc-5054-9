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
# 高于两眼 q2 基线成本，预算器必须在格点与眼别之间取舍，
# 并给约束策略的配对补齐留出升级空间。
SEGMENT_BUDGET = 30000
# 压预算练习使用的紧预算：低于两眼最低档总成本，触发托底并压坏标志
STARVATION_BUDGET = 12000

# JPEG 探针质量点（速率-失真模型的采样点）
# 低档位用于饥饿预算：q2/q5 会真正抹掉微小编号笔画与低对比标志
PROBE_QUALITIES = (2, 5, 8, 12, 20, 35, 55, 80)
# 编码时允许的实际质量档位（从探针点中选）
QUALITY_GRID = PROBE_QUALITIES
# 饥饿预算下允许跌到的最低档（仍然真实编码，只是细节被 JPEG 抹掉）
STARVATION_Q = 2

# 对应格点的失真差上限（MSE），两眼均可见时生效
DISTORTION_DIFF_LIMIT = 12.0

# 容器格式魔标与版本
CONTAINER_MAGIC = b"STB1"

# ---------- 字幕安全区（双眼共用，相对画面的比例框） ----------
SAFE_MARGIN_X = 0.10
SAFE_MARGIN_Y = 0.10
SAFE_BOX = (
    round(SAFE_MARGIN_X * WIDTH),
    round(SAFE_MARGIN_Y * HEIGHT),
    round((1.0 - SAFE_MARGIN_X) * WIDTH),
    round((1.0 - SAFE_MARGIN_Y) * HEIGHT),
)  # (x0, y0, x1, y1)

# ---------- 标注资产（渲染在棋盘深度的世界纹理） ----------
# 单眼微小编号：只让右眼在中段帧看到（栏杆对左眼形成天然遮挡）
MICRO_W = 13
MICRO_H = 11
# 双眼共有低对比标志：低振幅，预算吃紧时最先被 JPEG 量化抹平
SOFT_W = 40
SOFT_H = 22
SOFT_AMPLITUDE = 14          # 与本地底图的灰度差（足够低，紧预算下先被抹平）
SOFT_BASE = LIGHT_GRAY       # 画在亮格上
# 标志可辨认度评分阈值：高于它才允许自动放字幕
MARKER_SCORE_MIN = 0.55
# 视差估计允许的最大误差（像素），超过则拒绝自动放字幕
DISPARITY_TOL_PX = 3.5
# 视差搜索窗半宽（像素），需覆盖整像素取整与亚像素细化前的初始偏差
DISPARITY_SEARCH_HALF = 8
# 视差匹配 NCC 下限：峰太弱意味着右眼找不到可信的对应标志
DISPARITY_NCC_MIN = 0.25

# 纹理练习：共同可见段中单独给右眼叠加的确定性噪纹幅度
RIGHT_TEXTURE_AMP = 34

# 标注几何（世界坐标，深度固定为 BOARD_Z）
# 微小编号放在栏杆条 2 的右缘附近，运动中产生天然单眼窗口
MICRO_WORLD_X = 30.0
MICRO_WORLD_Y = 24.0
# 低对比标志放在画面中部偏上、始终处于共同可见范围
SOFT_WORLD_X = -34.0
SOFT_WORLD_Y = -70.0

# 标注可见判定阈值：一个标注在某眼的可见像素占比超过它才算“该眼可见”
ANNOT_VIS_RATIO = 0.6
