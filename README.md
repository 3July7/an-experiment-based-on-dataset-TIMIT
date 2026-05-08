# 第二次实验报告：基于 TIMIT 的说话人识别（GMM + MFCC / MFCC+Δ+ΔΔ）

## 1. 实验目的
本实验面向 **封闭集（closed-set）说话人识别**：给定一段语音，预测其所属说话人（训练集中共 462 类）。实验重点考察说话人识别整体流程与代码实现原理，包括：
1) **数据预处理与特征提取**：对比 MFCC 与 MFCC+Δ+ΔΔ；
2) **模型训练与推理**：使用每说话人一个 GMM 的生成式建模方法；
3) **参数影响分析**：对比 GMM 成分数 `n_components(K)` 对准确率与推理耗时的影响；
4) **错误分析与可视化**：统计常见混淆对、分析最差类别召回率。

---

## 2. 实验环境与依赖
- 语言：Python 3.x
- 依赖库（主要）：
  - numpy
  - librosa（音频读取、MFCC、delta）
  - scikit-learn（GaussianMixture、accuracy_score）
  - matplotlib（绘图）
  - joblib（模型持久化）

---

## 3. 项目结构与各文件作用说明

### 3.1 主要代码文件（src/目录）
- **config.py**：全局配置（训练/测试 json 路径、数据根目录等）。
- **data.py**：数据加载与预处理，从 json 读取 `filepath/speaker_id`，并调用特征提取函数。
- **features.py**：音频特征提取，支持 `mfcc` 与 `mfcc_d_dd`（MFCC+Δ+ΔΔ）。
- **model.py**：GMM 训练与推理（每说话人一个模型；测试时最大似然选择说话人）。
- **experiment.py**：实验主控脚本：配置实验组、训练/加载模型、评估并保存结果、生成可视化。
- **persistence.py**：模型与元信息保存/加载（`models.joblib` + `meta.json`）。
- **confusion_stats.py**：统计最常见混淆对（Top-N 错误对）。
- **metrics.py**：评估指标计算（每 speaker recall、worst-k speaker）。
- **viz.py**：可视化（准确率柱状图、最差 speaker recall 图）。
- **io_utils.py**：通用 I/O（保存 csv 等）。
- **path_utils.py**：路径解析（兼容大小写、下划线等差异，保证能定位音频文件）。

### 3.2 数据与实验结果目录
- **DATA/**：TIMIT 音频数据目录（DR1~DR8）。
- **train_info.json / test_info.json**：训练/测试样本清单（含 `filepath` 与 `speaker_id`）。
- **artifacts/**：实验产物输出根目录。
- **artifacts/experiments_gmm_k/**：本次实验各配置下的模型、评估与图表结果。

---

## 4. 实验整体流程

本实验完整流程如下：

### Step 1：读取训练/测试清单
`data.py: load_data_from_json(json_path, base_dir, feature_type)`  
- 从 `train_info.json` 或 `test_info.json` 中逐条读取：
  - `filepath`：音频相对路径
  - `speaker_id`：标签
- 通过 `resolve_audio_path` 将相对路径解析为真实文件路径。

### Step 2：特征提取
`features.py: extract_features(file_path, feature_type=...) -> (T, D)`  
- 音频读取：`librosa.load(sr=16000)`
- 可选端点裁剪（trim）：去除低能量静音段（减少无信息帧）
- MFCC：得到 `(n_mfcc, T)`，转置后为 `(T, 20)`
- 若 `feature_type="mfcc_d_dd"`：计算 Δ 与 ΔΔ 并与 MFCC 拼接为 `(T, 60)`
- CMVN：逐维标准化（减小通道/增益差异）

最终 `load_data_from_json` 输出：
- `features`: List[np.ndarray]，每条语音一个 `(T, D)`
- `labels`: List[str]，每条语音对应一个 speaker_id

### Step 3：训练模型（每说话人一个 GMM）
`model.py: train_speaker_gmms(train_features, train_labels, n_components=K, ...)`  
- 按 `speaker_id` 对训练样本分组，将同一 speaker 的所有帧特征拼接；
- 为每个 speaker 训练一个 GMM：
\[
p(x|s)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x|\mu_k,\Sigma_k)
\]
- 训练使用 EM 算法（E步算责任度，M步更新参数），迭代至收敛或达到 `max_iter`。

### Step 4：测试推理（最大似然分类）
`model.py: predict_speakers(gmm_models, test_features)`  
对每条测试语音 `X`：
- 对 462 个 speaker 模型分别计算平均对数似然：
\[
score(s)=\frac{1}{T}\sum_t \log p(x_t|s)
\]
- 选择分数最高的 speaker 作为预测结果：
\[
\hat{s}=\arg\max_s score(s)
\]

### Step 5：评估与可视化
- 总体准确率：`accuracy_score(y_true, y_pred)`
- 推理耗时：记录对测试集预测的总耗时
- 错误分析：
  - Top-20 混淆对：`confusion_stats.top_confusions`
  - 最差 20 speaker recall：`metrics.worst_k_speakers`
- 可视化：
  - accuracy 柱状图
  - worst-20 speaker recall 条形图

---

## 5. 方法原理

### 5.1 为什么 MFCC 能用于说话人识别？
MFCC 主要反映语音短时谱的 **包络形状（声道特性）**，与说话人的生理结构（声道长度、口腔形状等）相关，因此具有说话人区分能力。

### 5.2 Δ/ΔΔ 的意义与可能副作用
Δ/ΔΔ 表示特征随时间变化率（动态信息），可能补充语音过渡信息，但也会：
- 提高维度（20→60），增加参数估计难度；
- 对边界与噪声突变更敏感；
- 引入更多与内容/节奏相关的变化，可能扩大类内方差。

### 5.3 为什么 GMM 适合此类实验？
GMM 属于生成式模型，能够建模每个 speaker 的特征概率分布。测试时使用最大似然原则进行分类，流程清晰、可解释性强，适合实验教学与原理理解。

### 5.4 `n_components(K)` 的影响
`K` 越大：
- 模型容量越强，能拟合更复杂的多峰分布；
- 但训练/推理计算量增加，且在数据不足时可能更不稳定。

---

## 6. 实验方案与参数设置

### 6.1 对比维度
- 特征：`mfcc` 与 `mfcc_d_dd`
- GMM 成分数：K = 16 / 8 / 4
- 协方差类型：`diag`
- 最大迭代：`max_iter=200`

### 6.2 评价指标
1) Accuracy（整体准确率）
2) pred_time（测试集推理耗时）
3) Top-20 混淆对（错误类型统计）
4) Worst-20 speaker recall（类别级薄弱点定位）

---

## 7. 实验结果

### 7.1 总体对比（Accuracy / 推理时间）
| 特征 | K | Accuracy | pred_time(s) |
|---|---:|---:|---:|
| MFCC | 16 | **86.90%** | 67.56 |
| MFCC | 8  | 79.76% | 57.34 |
| MFCC | 4  | 64.94% | 52.09 |
| MFCC+Δ+ΔΔ | 16 | 83.12% | **96.46** |
| MFCC+Δ+ΔΔ | 8  | 73.38% | 75.00 |
| MFCC+Δ+ΔΔ | 4  | 59.42% | 65.38 |

**结论：**
- 最佳配置为 **MFCC + GMM(K=16)**，Accuracy = **86.90%**。
- K 从 16 降到 8/4 时准确率明显下降，说明模型容量不足导致欠拟合。
- 加入 Δ/ΔΔ 后推理明显变慢，同时准确率未提升，反而下降。

---

## 8. 结果分析与讨论

### 8.1 为什么 K 越小准确率越低？
K 决定 GMM 的表达能力。说话人 MFCC 分布通常呈多峰结构（不同音素/发音状态），K 太小会欠拟合，导致对数似然区分度下降，因此准确率降低。

### 8.2 为什么 MFCC+Δ+ΔΔ 反而变差？
主要原因可归纳为：
1) 维度升高导致估计更困难，模型更不稳定；  
2) Δ/ΔΔ 对边界与突变敏感，端点裁剪并非完美，差分会放大帧间突变；  
3) 动态信息引入更多内容/节奏因素，可能增加类内变化而非增强类间差异。

### 8.3 为什么 Δ/ΔΔ 推理更慢？
推理需要对每条测试语音对 462 个模型计算 score；计算量近似与 `T × K × D` 成正比。Δ/ΔΔ 使维度 D 由 20 增至 60，因此推理耗时显著增加。

### 8.4 错误分析补充
- **Top-20 混淆对**（`confusion_stats.top_confusions`）：统计最常见 `(true, pred)` 错误类型，便于定位“哪些 speaker 对最像”。
- **Worst-20 recall**（`metrics.worst_k_speakers`）：统计每个 speaker 的 recall，找到表现最差的类别，作为后续改进目标（数据不足、音色相近、噪声影响等）。

---

## 9. 实验结论
1) MFCC + GMM(K=16) 在本实验中取得最高准确率 **86.90%**。  
2) 减小 n_components 会导致欠拟合并显著降低准确率。  
3) 在当前预处理与数据条件下，加入 Δ/ΔΔ 未带来性能提升且显著增加推理时间。  

---

## 10. 改进方向
1) 更稳健的端点检测/VAD，减少差分特征对边界突变的敏感性；  
2) 对 MFCC+Δ+ΔΔ 做降维（PCA）或特征裁剪（clip）以降低噪声影响；  
3) 使用 UBM-GMM / MAP 自适应或更强的判别式模型（SVM、softmax）与嵌入式方法（x-vector）以提升性能与效率。