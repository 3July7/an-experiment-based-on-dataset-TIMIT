# 路径处理相关工具函数
import os
from typing import List, Optional


def resolve_audio_path(base_dir: str, rel_path: str) -> Optional[str]:
    """
    将 JSON 中的相对路径解析为实际存在的音频文件路径。

    - JSON 中可能是：SI1027_.wav（小写 wav + 尾随下划线）
    - 实际文件可能是：.WAV（大写），且可能去掉尾随下划线
    """
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    full = os.path.normpath(os.path.join(base_dir, rel_path))

    stem, ext = os.path.splitext(full)

    candidates: List[str] = []
    candidates.append(full)

    if ext:
        candidates.append(stem + ext.upper())
        candidates.append(stem + ext.lower())
    else:
        candidates.append(full + ".WAV")
        candidates.append(full + ".wav")

    if stem.endswith("_"):
        stem2 = stem[:-1]
        if ext:
            candidates.append(stem2 + ext.upper())
            candidates.append(stem2 + ext.lower())
        else:
            candidates.append(stem2 + ".WAV")
            candidates.append(stem2 + ".wav")

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            return p

    return None