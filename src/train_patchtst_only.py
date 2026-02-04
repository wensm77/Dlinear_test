import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE, MSE
from neuralforecast.models import PatchTST


def build_loss(loss_name: str, huber_delta: float):
    name = loss_name.lower()
    if name == "mae":
        return MAE()
    if name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


def acc(f, a):
    if f == 0 and a == 0:
        return 1.0
    if f != 0 and a == 0:
        return 0.0
    return max(0.0, 1 - abs(f - a) / a)


def calc_metrics(df: pd.DataFrame, pred_col: str):
    y = df["y"].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y - yhat)
    sq_err = (y - yhat) ** 2
    denom = np.abs(y) + np.abs(yhat)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    abs_y_sum = float(np.abs(y).sum())
    wape = float(abs_err.sum() / abs_y_sum) if abs_y_sum > 0 else np.nan
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * abs_err / denom)))

    # 每个 (cutoff, sku) 预测期第一个月
    first_idx = df.groupby(["cutoff", "unique_id"])["ds"].idxmin()
    first = df.loc[first_idx].copy()
    first["acc"] = first.apply(lambda r: acc(float(r[pred_col]), float(r["y"])), axis=1)
    first_month_acc = float(first["acc"].mean()) if len(first) else np.nan

    return pd.DataFrame(
        {
            "metric": ["MAE", "RMSE", "WAPE", "sMAPE", "FirstMonthAccuracy", "BusinessAccuracy"],
            "value": [
                mae,
                rmse,
                wape,
                smape,
                first_month_acc,
                float(1 - wape) if not np.isnan(wape) else np.nan,
            ],
        }
    )


def main(args):
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    dropped = len(id_counts) - len(valid_ids)
    df = df[df["unique_id"].isin(valid_ids)].copy()

    print(f"rows={len(df)} uids={df['unique_id'].nunique()} dropped_short={dropped} min_len={min_len}")
    print(f"date={df['ds'].min().date()}->{df['ds'].max().date()} zero_ratio={(df['y']==0).mean():.4f}")

    # Patch token count sanity check:
    # tokens = floor((L - patch_len)/stride) + 1
    tokens = (args.input_size - args.patch_len) // args.stride + 1 if args.input_size >= args.patch_len else 0
    print(f"PatchTST window L={args.input_size} patch_len={args.patch_len} stride={args.stride} -> tokens={tokens}")

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

    nf = NeuralForecast(models=[model], freq="MS")
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
    parser = argparse.ArgumentParser(description="仅 PatchTST（月度）训练与回测（不使用外生特征）。")
    parser.add_argument("--data", default="./data/data_cleaned.csv", help="输入数据（NeuralForecast long format）")
    parser.add_argument("--cv-output", default="./forecast_results_patchtst.csv")
    parser.add_argument("--metrics-output", default="./metrics_patchtst.csv")

    # T+1 优先：默认 horizon=1
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=24)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--step-size", type=int, default=1)

    # 对月度短窗口，默认 patch_len/stride 调小，避免 tokens 过少（例如 24,16,8 -> tokens=2）
    parser.add_argument("--patch-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)

    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scaler-type", default="robust", choices=["identity", "standard", "robust"])
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)

    main(parser.parse_args())
