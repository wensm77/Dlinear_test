import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE, MSE
from neuralforecast.models import PatchTST


def build_loss(name: str, huber_delta: float):
    n = name.lower()
    if n == "mae":
        return MAE()
    if n == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


def acc_vectorized(f: np.ndarray, a: np.ndarray):
    out = np.zeros_like(f, dtype=float)
    both_zero = (f == 0) & (a == 0)
    bad_zero = (f != 0) & (a == 0)
    normal = a != 0
    out[both_zero] = 1.0
    out[bad_zero] = 0.0
    out[normal] = np.maximum(0.0, 1.0 - np.abs(f[normal] - a[normal]) / np.abs(a[normal]))
    return out


def calc_metrics(df: pd.DataFrame, pred_col: str):
    y = df["y"].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y - yhat)
    sq_err = (y - yhat) ** 2
    denom = np.abs(y) + np.abs(yhat)

    first_idx = df.groupby(["cutoff", "unique_id"])["ds"].idxmin()
    first = df.loc[first_idx]

    return pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "WAPE", "sMAPE", "MeanAccuracy", "FirstDayAccuracy"],
            "value": [
                float(abs_err.mean()),
                float(np.sqrt(sq_err.mean())),
                float(abs_err.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else np.nan,
                float(np.mean(np.where(denom == 0, 0.0, 2 * abs_err / denom))),
                float(acc_vectorized(yhat, y).mean()),
                float(
                    acc_vectorized(
                        first[pred_col].to_numpy(dtype=float),
                        first["y"].to_numpy(dtype=float),
                    ).mean()
                ),
            ],
        }
    )


def main(args):
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    counts = df["unique_id"].value_counts()
    valid_ids = counts[counts >= min_len].index
    dropped = len(counts) - len(valid_ids)
    df = df[df["unique_id"].isin(valid_ids)].copy()

    tokens = (args.input_size - args.patch_len) // args.stride + 1 if args.input_size >= args.patch_len else 0
    print(f"rows={len(df)} uids={df['unique_id'].nunique()} dropped_short={dropped} min_len={min_len}")
    print(f"date={df['ds'].min().date()}->{df['ds'].max().date()} zero_ratio={(df['y']==0).mean():.4f}")
    print(f"PatchTST tokens={tokens} (L={args.input_size}, patch_len={args.patch_len}, stride={args.stride})")

    loss_fn = build_loss(args.loss, args.huber_delta)
    model = PatchTST(
        h=args.horizon,
        input_size=args.input_size,
        patch_len=args.patch_len,
        stride=args.stride,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        scaler_type=args.scaler_type,
        loss=loss_fn,
        valid_loss=loss_fn,
        val_check_steps=args.val_check_steps,
    )

    nf = NeuralForecast(models=[model], freq="D")
    cv_df = nf.cross_validation(df=df, n_windows=args.n_windows, step_size=args.step_size)
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    cv_df["PatchTST"] = cv_df["PatchTST"].clip(lower=0.0)

    out_cv = Path(args.cv_output)
    out_metrics = Path(args.metrics_output)
    out_cv.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(out_cv, index=False)
    metrics_df = calc_metrics(cv_df, "PatchTST")
    metrics_df.to_csv(out_metrics, index=False)

    print("\nmetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nsaved cv: {out_cv}")
    print(f"saved metrics: {out_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTST 日粒度：720天输入，预测未来30天。")
    parser.add_argument("--data", default="./data/daily_midfast_panel.csv")
    parser.add_argument("--cv-output", default="./forecast_results_patchtst_daily.csv")
    parser.add_argument("--metrics-output", default="./metrics_patchtst_daily.csv")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--input-size", type=int, default=720)
    parser.add_argument("--n-windows", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--patch-len", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scaler-type", default="robust", choices=["identity", "standard", "robust"])
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    main(parser.parse_args())
