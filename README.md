# DLinear 销量预测（快速版）

这个项目提供了两个版本的训练脚本：

1. `src/train.py`：baseline（保留不动，便于对比）
2. `src/train_improved.py`：改进版（Huber/MAE + 间歇序列回退 + 分层指标）
3. `src/train_deep.py`：深度学习版（NHiTS/PatchTST + 融合 + 首月准确率）
3. 数据清洗（每个 SKU 仅在自身有效区间补 0，避免全局尾部补 0）
4. 在 notebook 中可视化预测与真实值对比

## 1) 环境准备

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) 数据清洗

```bash
python data/clean_data_v2.py \
  --input ./data/data_all.csv \
  --output ./data/data_cleaned.csv
```

## 3) 训练与回测（Baseline）

```bash
python src/train.py \
  --data ./data/data_cleaned.csv \
  --horizon 12 \
  --input-size 18 \
  --n-windows 1 \
  --max-steps 300 \
  --cv-output ./forecast_results.csv \
  --metrics-output ./metrics.csv
```

## 4) 训练与回测（Improved，推荐）

```bash
python src/train_improved.py \
  --data ./data/data_cleaned.csv \
  --horizon 12 \
  --input-size 18 \
  --n-windows 2 \
  --step-size 3 \
  --loss huber \
  --huber-delta 5.0 \
  --fallback-zero-ratio 0.8 \
  --fallback-nonzero-max 6 \
  --cv-output ./forecast_results_improved.csv \
  --metrics-output ./metrics_improved.csv \
  --segment-metrics-output ./metrics_segments_improved.csv
```

其中改进版会额外输出：

- `yhat_hybrid`：DLinear 与 SeasonalNaive 的融合预测
- `use_fallback`：是否触发回退
- `metrics_segments_improved.csv`：按间歇/高销量/长尾分层的指标

## 5) 推送到 GitHub 并在远程服务器运行

先在 GitHub 新建一个空仓库（例如 `Dlinear`），然后在本地执行：

```bash
git add .
git commit -m "feat: improve data cleaning and dlinear backtest pipeline"
git branch -M main
git remote add origin git@github.com:<your_user>/Dlinear.git
git push -u origin main
```

在服务器上执行：

```bash
git clone git@github.com:<your_user>/Dlinear.git
cd Dlinear
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python data/clean_data_v2.py --input ./data/data_all.csv --output ./data/data_cleaned.csv
python src/train.py --data ./data/data_cleaned.csv --horizon 12 --input-size 18 --n-windows 1
# 或使用改进版
python src/train_improved.py --data ./data/data_cleaned.csv --horizon 12 --input-size 18 --n-windows 2 --step-size 3
# 或使用深度学习版
python src/train_deep.py --data ./data/data_cleaned.csv --models nhits,patchtst --horizon 12 --input-size 24 --n-windows 2 --step-size 3
```

如果你的数据文件未来超过 GitHub 单文件限制（100MB），建议使用 Git LFS。

## 6) 可视化预测 vs 真实值

运行完训练后打开 `draw.ipynb`，它会：

- 自动读取 `data/data_cleaned.csv` 与预测结果文件（可切换 baseline/improved）

## 7) 深度学习版（推荐在中快流件上试）

```bash
python src/train_deep.py \
  --data ./data/data_cleaned.csv \
  --models nhits,patchtst \
  --horizon 12 \
  --input-size 24 \
  --n-windows 2 \
  --step-size 3 \
  --loss huber \
  --huber-delta 5.0 \
  --cv-output ./forecast_results_deep.csv \
  --metrics-output ./metrics_deep.csv
```

`metrics_deep.csv` 除了常规指标外，还包含 `FirstMonthAccuracy`（每个 SKU 预测期第一个月按业务公式计算后取平均）。
- 自动识别预测列（如 `DLinear`）
- 绘制 Top SKU 的「历史 + 真实未来 + 预测未来」曲线
- 绘制全量聚合层面的预测对比与误差指标
