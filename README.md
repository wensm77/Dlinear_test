# DLinear 销量预测（快速版）

这个项目提供了一个仅使用销量 `y` 的快速基线流程：

1. 清洗数据（每个 SKU 仅在自身有效区间补 0，避免全局尾部补 0）
2. 使用 DLinear 进行滚动回测
3. 输出预测结果与评估指标
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

## 3) 训练与回测（DLinear）

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

## 4) 推送到 GitHub 并在远程服务器运行

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
```

如果你的数据文件未来超过 GitHub 单文件限制（100MB），建议使用 Git LFS。

## 5) 可视化预测 vs 真实值

运行完训练后打开 `draw.ipynb`，它会：

- 自动读取 `forecast_results.csv` 与 `data/data_cleaned.csv`
- 自动识别预测列（如 `DLinear`）
- 绘制 Top SKU 的「历史 + 真实未来 + 预测未来」曲线
- 绘制全量聚合层面的预测对比与误差指标
