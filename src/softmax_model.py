from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from embedding import utterance_mean_std


@dataclass
class SoftmaxBundle:
    clf: LogisticRegression
    classes: np.ndarray  # speaker labels in order


def train_softmax_logreg(train_features: List[np.ndarray], train_labels: List[str]) -> SoftmaxBundle:
    X = np.vstack([utterance_mean_std(f)[None, :] for f in train_features])  # (N, 2D)
    y = np.array(train_labels)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=1,  # Softmax 训练禁用并行,joblib 在 Windows 上开多进程时创建临时目录/注册资源，路径里含中文会导致问题
    )
    clf.fit(X, y)
    return SoftmaxBundle(clf=clf, classes=clf.classes_)


def predict_softmax(bundle: SoftmaxBundle, test_features: List[np.ndarray]) -> List[str]:
    X = np.vstack([utterance_mean_std(f)[None, :] for f in test_features])
    pred = bundle.clf.predict(X)
    return pred.tolist()