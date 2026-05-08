import os
from sklearn.metrics import accuracy_score

from config import TestJson, BaseDatasetDir
from data import load_data_from_json
from model import predict_speakers
from persistence import load_bundle
from viz import plot_confusion_matrix, plot_one_mfcc


def main():
    bundle_dir = "artifacts/gmm_mfcc"
    plot_dir = os.path.join(bundle_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    speaker_models, meta = load_bundle(bundle_dir)
    print("[*] 已加载模型与元数据：")
    print(f"    speakers={meta.get('num_speakers')}  created_at={meta.get('created_at')}")

    test_features, test_labels = load_data_from_json(TestJson, BaseDatasetDir)
    if len(test_labels) == 0:
        raise SystemExit("[!] 测试数据为空：请检查 TestJson / BaseDatasetDir / 路径解析规则。")

    print("[*] 开始在测试集上进行预测...")
    predicted_labels = predict_speakers(speaker_models, test_features)

    accuracy = accuracy_score(test_labels, predicted_labels)
    print("=======================================")
    print(f"   测试集最终准确率 (Accuracy): {accuracy * 100:.2f}%")
    print("=======================================")

    print("[*] 正在保存可视化图表到磁盘...")

    plot_confusion_matrix(
        test_labels,
        predicted_labels,
        title="说话人识别混淆矩阵 (GMM-MFCC)",
        save_path=os.path.join(plot_dir, "confusion_matrix.png"),
        show=False,
    )

    plot_one_mfcc(
        test_features[0],
        speaker_id=test_labels[0],
        save_path=os.path.join(plot_dir, "mfcc_example_0.png"),
        show=False,
    )

    print(f"[*] 图表已保存到：{plot_dir}")


if __name__ == "__main__":
    main()