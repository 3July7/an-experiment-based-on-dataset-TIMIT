# 模型训练与推理相关函数
from typing import Dict, List
import numpy as np

from sklearn.mixture import GaussianMixture


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
    """
    speaker_models: Dict[str, GaussianMixture] = {}
    speakers = np.unique(train_labels)

    print(f"[*] 开始训练：{len(speakers)} 个说话人的 GMM（n_components={n_components}）...")

    for speaker in speakers:
        speaker_feats = [train_features[i] for i, lab in enumerate(train_labels) if lab == speaker]
        X = np.vstack(speaker_feats)

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
            score = speaker_models[spk].score(feat)
            if score > best_score:
                best_score = score
                best_spk = spk

        predicted.append(best_spk)

    return predicted