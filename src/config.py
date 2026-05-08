# 配置项，如路径、参数等
import warnings
import matplotlib.pyplot as plt

# =============================================================================
# 全局设置
# =============================================================================
warnings.filterwarnings("ignore")  # 忽略 librosa 读音频时的一些格式/警告（不影响训练）

# 解决 matplotlib 中文与负号显示问题（需要系统安装 SimHei 字体）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 数据配置
# =============================================================================
TrainJson = "train_info.json"  # 训练集 JSON
TestJson = "test_info.json"    # 测试集 JSON

# BaseDatasetDir 必须是“包含 DR1/DR2/... 的那个目录”
BaseDatasetDir = "DATA/TRAIN"