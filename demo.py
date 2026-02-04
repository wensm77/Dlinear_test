import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def business_acc_vectorized(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    out = np.zeros_like(forecast, dtype=float)
    both_zero = (forecast == 0) & (actual == 0)
    actual_zero_forecast_nonzero = (actual == 0) & (forecast != 0)
    normal = actual != 0

    out[both_zero] = 1.0
    out[actual_zero_forecast_nonzero] = 0.0
    out[normal] = np.maximum(0.0, 1.0 - np.abs(forecast[normal] - actual[normal]) / np.abs(actual[normal]))
    return out


def safe_div(num, den):
    return float(num / den) if den else np.nan


def evaluate_month(df: pd.DataFrame, pred_col: str) -> dict:
    y = df["y"].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    biz_acc = float(business_acc_vectorized(yhat, y).mean()) if len(df) else np.nan

    actual_pos = y > 0
    pred_pos = yhat > 0
    actual_zero = ~actual_pos
    pred_zero = ~pred_pos

    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & actual_zero).sum())
    tn = int((pred_zero & actual_zero).sum())
    fn = int((pred_zero & actual_pos).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if not np.isnan(precision) and not np.isnan(recall) else np.nan

    return {
        "month": df["month"].iloc[0],
        "rows": int(len(df)),
        "sku_count": int(df["unique_id"].nunique()),
        "business_accuracy": biz_acc,
        "cls_accuracy": safe_div(tp + tn, tp + fp + tn + fn),
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def auto_pred_col(df: pd.DataFrame, pred_col: Optional[str]) -> str:
    if pred_col:
        if pred_col not in df.columns:
            raise ValueError(f"pred_col={pred_col} 不存在于输入文件。")
        return pred_col

    for c in ["yhat_two_stage", "PatchTST", "yhat_hybrid", "DLinear", "NBEATS"]:
        if c in df.columns:
            return c
    raise ValueError("未找到默认预测列，请通过 --pred-col 指定。")


def maybe_filter_by_month(df: pd.DataFrame, month_start: Optional[str], month_end: Optional[str]) -> pd.DataFrame:
    if month_start is None and month_end is None:
        return df
    out = df.copy()
    if month_start:
        start = pd.to_datetime(f"{month_start[:4]}-{month_start[4:6]}-01")
        out = out[out["ds"] >= start]
    if month_end:
        end = pd.to_datetime(f"{month_end[:4]}-{month_end[4:6]}-01")
        out = out[out["ds"] <= end]
    return out


def main():
    parser = argparse.ArgumentParser(description="按月份评估业务准确率 + 2stage分类指标。")
    parser.add_argument("--input", default="./forecast_results_patchtst_2stage.csv")
    parser.add_argument("--pred-col", default="yhat_two_stage")
    parser.add_argument("--month-start", default=None, help="YYYYMM，可选")
    parser.add_argument("--month-end", default=None, help="YYYYMM，可选")
    parser.add_argument("--output", default="./patchtst_2stage_monthly_eval.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ds", "cutoff"])
    pred_col = auto_pred_col(df, args.pred_col)
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = maybe_filter_by_month(df, args.month_start, args.month_end)
    if df.empty:
        raise ValueError("筛选后数据为空，请检查 month_start/month_end。")

    df["month"] = df["ds"].dt.strftime("%Y-%m")
    monthly_rows = [evaluate_month(g, pred_col) for _, g in df.groupby("month", sort=True)]
    monthly_df = pd.DataFrame(monthly_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(out_path, index=False)

    print("\n=== monthly evaluation ===")
    print(monthly_df.to_string(index=False))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
