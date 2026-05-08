# src/persistence.py
import os, json
from typing import Any, Dict, Tuple
import joblib

def save_bundle(out_dir: str, model_obj: Any, meta: Dict[str, Any]) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "models.joblib")
    meta_path = os.path.join(out_dir, "meta.json")
    joblib.dump(model_obj, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return model_path, meta_path

def load_bundle(in_dir: str) -> Tuple[Any, Dict[str, Any]]:
    model_path = os.path.join(in_dir, "models.joblib")
    meta_path = os.path.join(in_dir, "meta.json")
    model_obj = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model_obj, meta