from __future__ import annotations
import os, time, csv, json
from dataclasses import dataclass
from typing import List, Dict, Any

from sklearn.metrics import accuracy_score

from config import TrainJson, TestJson, BaseDatasetDir
from data import load_data_from_json
from model import train_speaker_gmms, predict_speakers
from persistence import save_bundle, load_bundle
from confusion_stats import top_confusions
from io_utils import save_confusion_csv
from metrics import worst_k_speakers
from viz import plot_results_bar, plot_worst_speakers_recall


@dataclass
class ExpCfg:
    feature_type: str           # "mfcc" or "mfcc_d_dd"
    name: str                   # display name
    exp_dir_name: str           # folder-safe name
    n_components: int           # GMM K

# out_dir = os.path.join("artifacts", "experiments_gmm_k", exp.exp_dir_name)
def exp_dir(base: str, exp: ExpCfg) -> str:
    return os.path.join(base, exp.exp_dir_name)

# 判断某个实验目录下是否已经存在训练好的模型和 meta 信息（即上次训练的结果），如果存在且 meta 匹配当前配置，则直接加载使用；
# 否则重新训练并覆盖保存。这样你在调参时就不必每次都从头训练，节省时间。
def bundle_exists(d: str) -> bool:
    return os.path.exists(os.path.join(d, "models.joblib")) and os.path.exists(os.path.join(d, "meta.json"))


def train_or_load_model(exp: ExpCfg, base_out_dir: str):
    out = exp_dir(base_out_dir, exp)

    if bundle_exists(out):
        print(f"[*] Found existing bundle, loading: {out}")
        model_obj, meta = load_bundle(out)

        # 可选：防止你改了 K 但复用了旧模型
        if meta.get("n_components") != exp.n_components or meta.get("feature_type") != exp.feature_type:
            print("[!] Bundle meta mismatch (feature_type or n_components). Retraining...")
        else:
            return model_obj, meta, False

    print(f"[*] No bundle found (or mismatch), training: {out}")

    train_features, train_labels = load_data_from_json(
        TrainJson, BaseDatasetDir, feature_type=exp.feature_type
    )

    # 训练模型并返回模型对象（这里是一个 dict，key 是 speaker_id，value 是对应的 GMM 模型）
    model_obj = train_speaker_gmms(
        train_features, train_labels,
        n_components=exp.n_components,
        covariance_type="diag",
        max_iter=200
    )

    # 保存模型和 meta 信息到磁盘，以便下次复用
    meta = {
        "train_json": TrainJson,
        "test_json": TestJson,
        "base_dataset_dir": BaseDatasetDir,
        "feature_type": exp.feature_type,
        "model_type": "gmm",
        "n_components": exp.n_components,
        "name": exp.name,
    }

    save_bundle(out, model_obj, meta)
    return model_obj, meta, True

# 评估函数：输入模型对象和测试数据，输出预测结果、准确率、预测时间等指标
def eval_model(exp: ExpCfg, model_obj):
    test_features, test_labels = load_data_from_json(
        TestJson, BaseDatasetDir, feature_type=exp.feature_type
    )

    t0 = time.perf_counter()
    y_pred = predict_speakers(model_obj, test_features)
    pred_sec = time.perf_counter() - t0

    acc = accuracy_score(test_labels, y_pred)
    return test_labels, y_pred, acc, pred_sec

# 主函数：定义多个实验配置，循环执行训练和评估，保存结果并生成对比图表
def main():
    base_out_dir = "artifacts/experiments_gmm_k"
    plot_dir = os.path.join(base_out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 定义多个实验配置：不同的特征类型（MFCC vs MFCC+Δ+ΔΔ）和 GMM 组件数 K
    exps = [
        ExpCfg("mfcc",      "MFCC + GMM(K=16)",              "MFCC__GMM_K16",      16),
        ExpCfg("mfcc",      "MFCC + GMM(K=8)",               "MFCC__GMM_K8",       8),
        ExpCfg("mfcc",      "MFCC + GMM(K=4)",               "MFCC__GMM_K4",       4),

        ExpCfg("mfcc_d_dd", "MFCC+Δ+ΔΔ + GMM(K=16)",         "MFCC_D_DD__GMM_K16", 16),
        ExpCfg("mfcc_d_dd", "MFCC+Δ+ΔΔ + GMM(K=8)",          "MFCC_D_DD__GMM_K8",  8),
        ExpCfg("mfcc_d_dd", "MFCC+Δ+ΔΔ + GMM(K=4)",          "MFCC_D_DD__GMM_K4",  4),
    ]

    results: List[Dict[str, Any]] = []
    best_for_worst_plot = None  # store best config's worst-20 recall
    
    # 循环执行每个实验配置：训练/加载模型，评估性能，保存结果和混淆信息
    for exp in exps:
        print(f"\n===== Experiment: {exp.name} =====")
        out = exp_dir(base_out_dir, exp)

        model_obj, meta, trained_now = train_or_load_model(exp, base_out_dir)

        y_true, y_pred, acc, pred_sec = eval_model(exp, model_obj)
        print(f"[result] acc={acc*100:.2f}%  pred_time={pred_sec:.2f}s  trained_now={trained_now}")

        # 保存混淆Top-20
        conf = top_confusions(y_true, y_pred, top_k=20)
        conf_path = save_confusion_csv(os.path.join(out, "confusion_top20.csv"), conf)

        # 保存本次评估信息
        eval_info = {
            "accuracy": acc,
            "pred_time_sec": pred_sec,
            "num_test_utts": len(y_true),
            "feature_type": exp.feature_type,
            "n_components": exp.n_components,
        }
        with open(os.path.join(out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(eval_info, f, ensure_ascii=False, indent=2)

        results.append({
            "name": exp.name,
            "feature": exp.feature_type,
            "model": "gmm",
            "n_components": exp.n_components,
            "accuracy": acc,
            "pred_time_sec": pred_sec,
            "exp_dir": out,
            "confusion_top20_csv": conf_path,
        })

        # 选准确率最高的一组来画 worst20 recall
        if best_for_worst_plot is None or acc > best_for_worst_plot["acc"]:
            spk, rec = worst_k_speakers(y_true, y_pred, k=20)
            best_for_worst_plot = {
                "acc": acc,
                "speakers": spk,
                "recall": rec,
                "title": f"Worst-20 Speaker Recall ({exp.name})"
            }

    # 汇总结果表
    csv_path = os.path.join(base_out_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "feature", "model", "n_components", "accuracy", "pred_time_sec", "exp_dir", "confusion_top20_csv"]
        )
        w.writeheader()
        w.writerows(results)

    # 图1：accuracy柱状图
    plot_results_bar(results, save_path=os.path.join(plot_dir, "accuracy_bar.png"), show=False)

    # 图2：最差20 speaker recall（选最好的那组）
    plot_worst_speakers_recall(
        best_for_worst_plot["speakers"],
        best_for_worst_plot["recall"],
        title=best_for_worst_plot["title"],
        save_path=os.path.join(plot_dir, "worst20_recall.png"),
        show=False,
    )

    print(f"\n[*] Done. results: {csv_path}")
    print(f"[*] Plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()