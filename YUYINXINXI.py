import os
import json
import ast
import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

# =============================================================================
# 全局设置
# =============================================================================
warnings.filterwarnings("ignore")  # 忽略 librosa 读音频时的一些格式/警告（不影响训练）

# 解决 matplotlib 中文与负号显示问题（需要系统安装 SimHei 字体）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# =============================================================================
# 1) 配置项（你原来的 TrainJson / TestJson 写反了，这里修复）
# =============================================================================
TrainJson = "train_info.json"  # 训练集 JSON
TestJson = "test_info.json"    # 测试集 JSON

# BaseDatasetDir 必须是“包含 DR1/DR2/... 的那个目录”
# 例如：DATA/TRAIN 目录下直接有 DR1、DR2... 文件夹，则这样写是对的
BaseDatasetDir = "DATA/TRAIN"


# =============================================================================
# 2) 音频路径解析：专门修复你遇到的两类问题
#    - JSON 中是：SI1027_.wav（小写 wav + 尾随下划线）
#    - 实际文件是：.WAV（大写），且可能会去掉尾随下划线
# =============================================================================
def resolve_audio_path(base_dir: str, rel_path: str) -> Optional[str]:
    """
    将 JSON 中的相对路径解析为实际存在的音频文件路径。

    参数
    ----
    base_dir : str
        音频数据根目录，要求 base_dir 下直接包含 DR1、DR2... 文件夹
    rel_path : str
        JSON 中的 filepath，如 "DR1/FCJF0/SI1027_.wav"

    返回
    ----
    str 或 None
        找到则返回真实存在的路径；找不到返回 None
    """
    # 统一斜杠，避免 Windows/Linux 混用
    rel_path = rel_path.replace("\\", "/").lstrip("/")

    # 拼接为完整路径
    full = os.path.normpath(os.path.join(base_dir, rel_path))

    # 用 splitext 拆分后缀，比 replace 更安全（不会误替换文件名中间的内容）
    stem, ext = os.path.splitext(full)  # ext 例如 ".wav"

    candidates: List[str] = []

    # 1) 原样尝试（有时文件就真的叫 .wav）
    candidates.append(full)

    # 2) 尝试后缀大小写（你的数据是 .WAV）
    if ext:
        candidates.append(stem + ext.upper())
        candidates.append(stem + ext.lower())
    else:
        candidates.append(full + ".WAV")
        candidates.append(full + ".wav")

    # 3) 如果文件名以 '_' 结尾，尝试去掉 '_'（TIMIT 有些标注会这样）
    if stem.endswith("_"):
        stem2 = stem[:-1]
        if ext:
            candidates.append(stem2 + ext.upper())
            candidates.append(stem2 + ext.lower())
        else:
            candidates.append(stem2 + ".WAV")
            candidates.append(stem2 + ".wav")

    # 去重并按顺序查找第一个存在的候选
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p

    return None


# =============================================================================
# 3) 特征提取：MFCC +（可选）去静音 + CMVN
# =============================================================================
def extract_features(
    file_path: str,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,   # 10ms @ 16kHz
    win_length: int = 400,   # 25ms @ 16kHz
    do_trim: bool = True,
    trim_top_db: int = 25,
    do_cmvn: bool = True,
) -> np.ndarray:
    """
    从音频中提取 MFCC 特征，输出形状为 (T, n_mfcc)。

    为什么这里做了额外处理？
    - do_trim: 去掉前后静音，减少静音帧对 GMM 的污染（通常会提升说话人识别效果）
    - do_cmvn: 做倒谱均值方差归一化（CMVN），增强对录音条件变化的鲁棒性
    """
    y, _sr = librosa.load(file_path, sr=sr)

    # 1) 去静音（端点检测的简化版）
    if do_trim:
        y, _ = librosa.effects.trim(y, top_db=trim_top_db)

    # 2) 提取 MFCC: (n_mfcc, T)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    # 3) CMVN（对每个 MFCC 维度做均值方差归一化）
    if do_cmvn:
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)

    # 转成 (T, n_mfcc)，每一行是一帧特征，适配 sklearn GMM
    return mfcc.T


# =============================================================================
# 4) JSON 加载与特征提取
# =============================================================================
def load_data_from_json(json_path: str, base_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    从 JSON 文件加载数据并提取 MFCC 特征。

    JSON 每条应至少包含：
    - filepath: "DR1/FCJF0/SI1027_.wav"
    - speaker_id: "CJF0"

    返回：
    - features: List[np.ndarray]，每个元素形状 (T_i, n_mfcc)
    - labels:   List[str] speaker_id 列表
    """
    features: List[np.ndarray] = []
    labels: List[str] = []

    if not os.path.exists(json_path):
        print(f"[!] 致命错误：找不到 JSON 文件：{json_path}")
        return features, labels

    print(f"[*] 正在解析 {json_path} ...")

    # 读取并解析 JSON：这里保留你原来的“容错”逻辑（兼容非标准 JSON）
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # 兼容 “单个字典对象” 的情况：{...} -> [{...}]
    if content.startswith("{") and content.endswith("}"):
        content = f"[{content}]"

    try:
        data_info = json.loads(content)
    except json.JSONDecodeError:
        # 如果 JSON 不标准（比如单引号），尝试用 literal_eval 兜底
        data_info = ast.literal_eval(content)

    # 统计信息
    ok = 0
    miss = 0
    fail = 0

    # 可选：快速检查 base_dir 是否正确（应包含 DR1 文件夹）
    # 若这里为 False，你几乎一定会“全都找不到音频”
    if not os.path.exists(os.path.join(base_dir, "DR1")):
        print(f"[!] 警告：在 BaseDatasetDir 下未发现 DR1 文件夹：{os.path.abspath(base_dir)}")
        print("    请确认 BaseDatasetDir 是否设置为“直接包含 DR1/DR2/... 的目录”。\n")

    for idx, item in enumerate(data_info):
        rel_path = item.get("filepath") or item.get("file_path") or item.get("file")
        speaker_id = item.get("speaker_id") or item.get("speaker") or item.get("label")

        if not rel_path or not speaker_id:
            continue

        actual_path = resolve_audio_path(base_dir, rel_path)
        if actual_path is None:
            miss += 1
            # 只打印前若干条，避免刷屏；需要更详细可自行调整
            if miss <= 10:
                print(f"   [!] 找不到音频: rel_path={rel_path}")
            continue

        try:
            feat = extract_features(actual_path)
            if feat.size == 0:
                # 极端情况：trim 后音频为空（全静音）
                fail += 1
                continue
            features.append(feat)
            labels.append(str(speaker_id))
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"   [!] 提取特征失败: {actual_path}  错误: {e}")

    print(f"[*] 加载完成：成功 {ok} 条 | 路径缺失 {miss} 条 | 特征失败 {fail} 条\n")
    return features, labels


# =============================================================================
# 5) 训练：每个 speaker 一个 GMM
# =============================================================================
def train_speaker_gmms(
    train_features: List[np.ndarray],
    train_labels: List[str],
    n_components: int = 16,
    covariance_type: str = "diag",
    max_iter: int = 200,
    random_state: int = 42,
) -> Dict[str, GaussianMixture]:
    """
    为每个说话人训练一个 GMM。

    注意：n_components 需要与每个说话人的数据量匹配。
    如果每个说话人语料较少，建议从 4/8 开始试。
    """
    speaker_models: Dict[str, GaussianMixture] = {}
    speakers = np.unique(train_labels)

    print(f"[*] 开始训练：{len(speakers)} 个说话人的 GMM（n_components={n_components}）...")

    for speaker in speakers:
        speaker_feats = [train_features[i] for i, lab in enumerate(train_labels) if lab == speaker]
        X = np.vstack(speaker_feats)  # (sum_T, n_mfcc)

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
        gmm.fit(X)
        speaker_models[speaker] = gmm

    print("[*] 所有 GMM 模型训练完成！\n")
    return speaker_models


# =============================================================================
# 6) 预测与评估
# =============================================================================
def predict_speakers(
    speaker_models: Dict[str, GaussianMixture],
    test_features: List[np.ndarray],
) -> List[str]:
    """
    对每条测试音频：计算其在所有 speaker GMM 下的平均对数似然，取最大者。
    """
    predicted: List[str] = []

    speakers = list(speaker_models.keys())
    for feat in test_features:
        best_score = -np.inf
        best_spk = None

        for spk in speakers:
            score = speaker_models[spk].score(feat)  # 平均 log-likelihood
            if score > best_score:
                best_score = score
                best_spk = spk

        predicted.append(best_spk)

    return predicted


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], title: str = "混淆矩阵 (GMM-MFCC)"):
    """
    绘制混淆矩阵。这里使用 true/pred 的并集作为 labels，避免“测试集中有训练集外 speaker”
    导致矩阵维度不全。
    """
    all_labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=all_labels,
                yticklabels=all_labels)
    plt.title(title, fontsize=16)
    plt.xlabel("预测标签 (Predicted)", fontsize=12)
    plt.ylabel("真实标签 (True)", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_one_mfcc(mfcc_feat: np.ndarray, speaker_id: str, hop_length: int = 160, sr: int = 16000):
    """
    绘制一条样本的 MFCC 特征图。
    注意：这是 MFCC，不是频谱，y 轴代表“倒谱系数索引”。
    """
    plt.figure(figsize=(10, 4))
    # mfcc_feat: (T, n_mfcc) -> specshow 期望 (n_mfcc, T)
    librosa.display.specshow(
        mfcc_feat.T,
        x_axis="time",
        sr=sr,
        hop_length=hop_length
    )
    plt.colorbar()
    plt.title(f"说话人 {speaker_id} 的 MFCC 特征图")
    plt.ylabel("MFCC 系数")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7) 主程序
# =============================================================================
if __name__ == "__main__":
    # 1) 加载训练/测试数据
    train_features, train_labels = load_data_from_json(TrainJson, BaseDatasetDir)
    test_features, test_labels = load_data_from_json(TestJson, BaseDatasetDir)

    if len(train_labels) == 0 or len(test_labels) == 0:
        print("[!] 提取的数据为空。请重点检查：")
        print("    1) BaseDatasetDir 是否指向包含 DR1/DR2/... 的目录")
        print("    2) JSON 的 filepath 是否与目录结构一致")
        print("    3) resolve_audio_path 的规则是否符合你的真实文件名（是否带下划线、是否 .WAV）")
        raise SystemExit(1)

    # 2) 训练 GMM（每说话人一个模型）
    speaker_models = train_speaker_gmms(
        train_features, train_labels,
        n_components=16,          # 若数据少可改 8 或 4
        covariance_type="diag",
        max_iter=200
    )

    # 3) 测试集预测
    print("[*] 开始在测试集上进行预测...")
    predicted_labels = predict_speakers(speaker_models, test_features)

    # 4) 评估准确率
    accuracy = accuracy_score(test_labels, predicted_labels)
    print("=======================================")
    print(f"   测试集最终准确率 (Accuracy): {accuracy * 100:.2f}%")
    print("=======================================\n")

    # 5) 可视化
    print("[*] 正在生成可视化图表...")
    plot_confusion_matrix(test_labels, predicted_labels, title="说话人识别混淆矩阵 (GMM-MFCC)")

    # 画一条样本的 MFCC（如果你想看第 1 条）
    plot_one_mfcc(test_features[0], speaker_id=test_labels[0])