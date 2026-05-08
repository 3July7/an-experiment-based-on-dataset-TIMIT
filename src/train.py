from datetime import datetime

from config import TrainJson, BaseDatasetDir  # 保留你的默认配置
from data import load_data_from_json
from model import train_speaker_gmms
from persistence import save_bundle


def main():
    # 你也可以改成命令行参数；这里先保持简单
    out_dir = "artifacts/gmm_mfcc"

    train_features, train_labels = load_data_from_json(TrainJson, BaseDatasetDir)
    if len(train_labels) == 0:
        raise SystemExit("[!] 训练数据为空：请检查 TrainJson / BaseDatasetDir / 路径解析规则。")

    # 训练参数（建议你把它们当作“实验配置”，保存到 meta 里）
    train_cfg = {
        "n_components": 16,
        "covariance_type": "diag",
        "max_iter": 200,
        "random_state": 42,
    }

    speaker_models = train_speaker_gmms(
        train_features, train_labels,
        n_components=train_cfg["n_components"],
        covariance_type=train_cfg["covariance_type"],
        max_iter=train_cfg["max_iter"],
        random_state=train_cfg["random_state"],
    )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "train_json": TrainJson,
        "base_dataset_dir": BaseDatasetDir,
        "train_cfg": train_cfg,
        # 下面这些是特征侧的“默认值说明”，如果你后续做 CLI 参数化，建议写入真实值
        "feature_cfg": {
            "sr": 16000,
            "n_mfcc": 20,
            "n_fft": 512,
            "hop_length": 160,
            "win_length": 400,
            "do_trim": True,
            "trim_top_db": 25,
            "do_cmvn": True,
        },
        "speakers": sorted(list(speaker_models.keys())),
        "num_speakers": len(speaker_models),
        "num_train_utts": len(train_labels),
    }

    model_path, meta_path = save_bundle(out_dir, speaker_models, meta)
    print("[*] 训练完成并已保存：")
    print(f"    - {model_path}")
    print(f"    - {meta_path}")


if __name__ == "__main__":
    main()