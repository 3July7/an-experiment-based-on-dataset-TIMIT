# 数据加载与预处理相关函数
import os
import json
import ast
from typing import List, Tuple

import numpy as np

from path_utils import resolve_audio_path
from features import extract_features, FeatureType


def load_data_from_json(json_path: str, base_dir: str, feature_type: FeatureType = "mfcc") -> Tuple[List[np.ndarray], List[str]]:
    """
    从 JSON 文件加载数据并提取 MFCC 特征。

    JSON 每条应至少包含：
    - filepath: "DR1/FCJF0/SI1027_.wav"
    - speaker_id: "CJF0"
    """
    features: List[np.ndarray] = []
    labels: List[str] = []

    if not os.path.exists(json_path):
        print(f"[!] 致命错误：找不到 JSON 文件：{json_path}")
        return features, labels

    print(f"[*] 正在解析 {json_path} ...")

    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{") and content.endswith("}"):
        content = f"[{content}]"

    try:
        data_info = json.loads(content)
    except json.JSONDecodeError:
        data_info = ast.literal_eval(content)

    ok = 0
    miss = 0
    fail = 0

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
            if miss <= 10:
                print(f"   [!] 找不到音频: rel_path={rel_path}")
            continue

        try:
            feat = extract_features(actual_path, feature_type=feature_type)
            if feat.size == 0:
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