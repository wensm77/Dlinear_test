import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear


def calc_metrics(df, pred_col):
    y = df["y"].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y - yhat)
    sq_err = (y - yhat) ** 2
    denom = np.abs(y) + np.abs(yhat)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    wape = float(abs_err.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else np.nan
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * abs_err / denom)))
    nonzero_mask = y != 0
    mape_nonzero = float(np.mean(abs_err[nonzero_mask] / np.abs(y[nonzero_mask]))) if nonzero_mask.any() else np.nan

    return {
        "metric": ["MAE", "RMSE", "WAPE", "sMAPE", "MAPE_nonzero"],
        "value": [mae, rmse, wape, smape, mape_nonzero],
    }


def train_and_backtest(
    data_path,
    cv_output_path,
    metrics_output_path,
    horizon,
    input_size,
    n_windows,
    max_steps,
    learning_rate,
):
    print(f"读取数据: {data_path}")
    df = pd.read_csv(data_path)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    min_len = input_size + horizon
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    dropped = len(id_counts) - len(valid_ids)
    df = df[df["unique_id"].isin(valid_ids)].copy()

    print(f"数据行数: {len(df)}")
    print(f"序列数: {df['unique_id'].nunique()} (过滤掉过短序列: {dropped})")
    print(f"时间范围: {df['ds'].min().date()} -> {df['ds'].max().date()}")
    print(f"0 占比: {(df['y'] == 0).mean():.4f}")

    model = DLinear(
        h=horizon,
        input_size=input_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        scaler_type="robust",
        val_check_steps=100,
    )
    nf = NeuralForecast(models=[model], freq="MS")

    print("开始滚动回测...")
    cv_df = nf.cross_validation(df=df, n_windows=n_windows, step_size=horizon)
    pred_col = "DLinear"
    cv_df[pred_col] = cv_df[pred_col].clip(lower=0.0)

    cv_output_path = Path(cv_output_path)
    cv_output_path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(cv_output_path, index=False)

    metrics_df = pd.DataFrame(calc_metrics(cv_df, pred_col))
    metrics_output_path = Path(metrics_output_path)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_output_path, index=False)

    print("回测完成。")
    print(metrics_df.to_string(index=False))
    print(f"回测结果已保存: {cv_output_path}")
    print(f"指标已保存: {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLinear 训练与滚动回测。")
    parser.add_argument("--data", default="./data/data_cleaned.csv", help="清洗后数据路径")
    parser.add_argument("--cv-output", default="./forecast_results.csv", help="回测输出 CSV 路径")
    parser.add_argument("--metrics-output", default="./metrics.csv", help="指标输出 CSV 路径")
    parser.add_argument("--horizon", type=int, default=3, help="预测步长（月）")
    parser.add_argument("--input-size", type=int, default=12, help="历史窗口长度（月）")
    parser.add_argument("--n-windows", type=int, default=4, help="滚动回测窗口数")
    parser.add_argument("--max-steps", type=int, default=300, help="模型训练步数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    args = parser.parse_args()

    train_and_backtest(
        data_path=args.data,
        cv_output_path=args.cv_output,
        metrics_output_path=args.metrics_output,
        horizon=args.horizon,
        input_size=args.input_size,
        n_windows=args.n_windows,
        max_steps=args.max_steps,
        learning_rate=args.lr,
    )
