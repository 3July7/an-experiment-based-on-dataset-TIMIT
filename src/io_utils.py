import os, csv
from typing import List, Tuple

def save_confusion_csv(path: str, rows: List[Tuple[str, str, int]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_speaker", "pred_speaker", "count"])
        w.writerows(rows)
    return path