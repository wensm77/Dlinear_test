import argparse
from pathlib import Path
from typing import List, Optional

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


def evaluate_slice(df: pd.DataFrame, pred_col: str) -> dict:
    y = df["y"].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    acc = business_acc_vectorized(yhat, y)

    actual_pos = y > 0
    pred_pos = yhat > 0
    actual_zero = ~actual_pos
    pred_zero = ~pred_pos

    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & actual_zero).sum())
    tn = int((pred_zero & actual_zero).sum())
    fn = int((pred_zero & actual_pos).sum())

    abs_err = np.abs(y - yhat)
    wape = safe_div(abs_err.sum(), np.abs(y).sum())
    volume_bias = safe_div(yhat.sum() - y.sum(), np.abs(y).sum())

    return {
        "rows": int(len(df)),
        "sku_count": int(df["unique_id"].nunique()),
        "overall_acc": float(acc.mean()) if len(acc) else np.nan,
        "pos_acc": float(acc[actual_pos].mean()) if actual_pos.any() else np.nan,
        "zero_acc": float(acc[actual_zero].mean()) if actual_zero.any() else np.nan,
        "wape": wape,
        "business_accuracy": float(1 - wape) if not np.isnan(wape) else np.nan,
        "volume_bias": volume_bias,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "cls_acc": safe_div(tp + tn, tp + fp + tn + fn),
        "precision_pos": safe_div(tp, tp + fp),
        "recall_pos": safe_div(tp, tp + fn),
        "f1_pos": safe_div(2 * tp, 2 * tp + fp + fn),
        "recall_zero": safe_div(tn, tn + fp),
        "precision_zero": safe_div(tn, tn + fn),
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


def threshold_sweep(df: pd.DataFrame, prob_col: str, reg_col: str, grid: List[float]) -> pd.DataFrame:
    y = df["y"].to_numpy(dtype=float)
    p = df[prob_col].to_numpy(dtype=float)
    reg = df[reg_col].to_numpy(dtype=float)
    rows = []
    for t in grid:
        yhat = np.where(p >= t, reg, 0.0)
        tmp = df[["unique_id", "ds", "cutoff", "y"]].copy()
        tmp["yhat"] = yhat
        m = evaluate_slice(tmp, "yhat")
        rows.append({"threshold": t, **m})
    return pd.DataFrame(rows).sort_values("overall_acc", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="2-stage PatchTST 详细评估报告（含混淆矩阵/正样本准确率/整体准确率）。")
    parser.add_argument("--input", default="./forecast_results_patchtst_2stage.csv")
    parser.add_argument("--pred-col", default=None)
    parser.add_argument("--month-start", default=None, help="YYYYMM，可选")
    parser.add_argument("--month-end", default=None, help="YYYYMM，可选")
    parser.add_argument("--output-prefix", default="patchtst_2stage_eval")
    parser.add_argument("--threshold-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()

    df = pd.read_csv(args.input, parse_dates=["ds", "cutoff"])
    pred_col = auto_pred_col(df, args.pred_col)
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = maybe_filter_by_month(df, args.month_start, args.month_end)
    if df.empty:
        raise ValueError("筛选后数据为空，请检查 month_start/month_end。")

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    overall = pd.DataFrame([evaluate_slice(df, pred_col)])
    overall.insert(0, "pred_col", pred_col)
    overall.to_csv(f"{out_prefix}_overall.csv", index=False)

    by_ds_rows = []
    for ds, g in df.groupby("ds", sort=True):
        row = evaluate_slice(g, pred_col)
        row["ds"] = ds.strftime("%Y-%m-%d")
        by_ds_rows.append(row)
    by_ds = pd.DataFrame(by_ds_rows)
    by_ds.to_csv(f"{out_prefix}_by_ds.csv", index=False)

    by_cutoff_rows = []
    for cutoff, g in df.groupby("cutoff", sort=True):
        row = evaluate_slice(g, pred_col)
        row["cutoff"] = cutoff.strftime("%Y-%m-%d")
        by_cutoff_rows.append(row)
    by_cutoff = pd.DataFrame(by_cutoff_rows)
    by_cutoff.to_csv(f"{out_prefix}_by_cutoff.csv", index=False)

    sku_rows = []
    for uid, g in df.groupby("unique_id", sort=False):
        row = evaluate_slice(g, pred_col)
        row["unique_id"] = uid
        sku_rows.append(row)
    sku_diag = pd.DataFrame(sku_rows).sort_values("overall_acc")
    sku_diag.to_csv(f"{out_prefix}_sku_diagnosis.csv", index=False)

    conf = pd.DataFrame(
        {
            "": ["actual_zero", "actual_pos"],
            "pred_zero": [int(overall["tn"].iloc[0]), int(overall["fn"].iloc[0])],
            "pred_pos": [int(overall["fp"].iloc[0]), int(overall["tp"].iloc[0])],
        }
    )
    conf.to_csv(f"{out_prefix}_confusion_overall.csv", index=False)

    if {"p_nonzero", "yhat_reg_only"}.issubset(df.columns):
        grid = [float(x.strip()) for x in args.threshold_grid.split(",") if x.strip()]
        sweep = threshold_sweep(df, "p_nonzero", "yhat_reg_only", grid)
        sweep.to_csv(f"{out_prefix}_threshold_sweep.csv", index=False)
        print("best threshold by overall_acc:")
        print(sweep.head(5)[["threshold", "overall_acc", "pos_acc", "zero_acc", "wape"]].to_string(index=False))

    print("\n=== overall ===")
    print(overall.to_string(index=False))
    print(f"\nsaved: {out_prefix}_overall.csv")
    print(f"saved: {out_prefix}_by_ds.csv")
    print(f"saved: {out_prefix}_by_cutoff.csv")
    print(f"saved: {out_prefix}_sku_diagnosis.csv")
    print(f"saved: {out_prefix}_confusion_overall.csv")


if __name__ == "__main__":
    main()
