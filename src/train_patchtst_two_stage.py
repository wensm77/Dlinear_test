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


def acc_vectorized(f: np.ndarray, a: np.ndarray):
    out = np.zeros_like(f, dtype=float)
    both_zero = (f == 0) & (a == 0)
    a_zero_f_nonzero = (a == 0) & (f != 0)
    normal = a != 0

    out[both_zero] = 1.0
    out[a_zero_f_nonzero] = 0.0
    out[normal] = np.maximum(0.0, 1.0 - np.abs(f[normal] - a[normal]) / np.abs(a[normal]))
    return out


def metrics_from_pred(df: pd.DataFrame, pred_col: str):
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

    acc_all = float(acc_vectorized(yhat, y).mean())

    first_idx = df.groupby(["cutoff", "unique_id"])["ds"].idxmin()
    first = df.loc[first_idx]
    first_acc = float(acc_vectorized(first[pred_col].to_numpy(dtype=float), first["y"].to_numpy(dtype=float)).mean())

    actual_zero = y == 0
    pred_zero = yhat == 0
    zero_recall = float((pred_zero & actual_zero).sum() / actual_zero.sum()) if actual_zero.any() else np.nan
    zero_precision = float((pred_zero & actual_zero).sum() / pred_zero.sum()) if pred_zero.any() else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "sMAPE": smape,
        "MeanAccuracy": acc_all,
        "FirstMonthAccuracy": first_acc,
        "ZeroRecall": zero_recall,
        "ZeroPrecision": zero_precision,
        "BusinessAccuracy": float(1 - wape) if not np.isnan(wape) else np.nan,
    }


def parse_threshold_grid(grid_str: str):
    vals = []
    for x in grid_str.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    vals = [v for v in vals if 0.0 <= v <= 1.0]
    if not vals:
        raise ValueError("threshold-grid 不能为空，且阈值应在 [0,1]。")
    return sorted(set(vals))


def run_cv_patchtst(df: pd.DataFrame, args, loss_name: str, huber_delta: float):
    loss_fn = build_loss(loss_name, huber_delta)
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
    return nf.cross_validation(df=df, n_windows=args.n_windows, step_size=args.step_size)


def main(args):
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    dropped = len(id_counts) - len(valid_ids)
    df = df[df["unique_id"].isin(valid_ids)].copy()

    tokens = (args.input_size - args.patch_len) // args.stride + 1 if args.input_size >= args.patch_len else 0
    print(f"rows={len(df)} uids={df['unique_id'].nunique()} dropped_short={dropped} min_len={min_len}")
    print(f"date={df['ds'].min().date()}->{df['ds'].max().date()} zero_ratio={(df['y']==0).mean():.4f}")
    print(f"PatchTST tokens={tokens} (L={args.input_size}, patch_len={args.patch_len}, stride={args.stride})")

    # Stage-1: 分类（预测是否非零）
    cls_df = df.copy()
    cls_df["y"] = (cls_df["y"] > 0).astype(float)
    cls_cv = run_cv_patchtst(cls_df, args, loss_name=args.cls_loss, huber_delta=args.huber_delta).rename(
        columns={"y": "y_cls", "PatchTST": "p_nonzero_raw"}
    )

    # Stage-2: 回归（预测量级，log1p 空间）
    reg_df = df.copy()
    reg_df["y"] = np.log1p(reg_df["y"])
    reg_cv = run_cv_patchtst(reg_df, args, loss_name=args.reg_loss, huber_delta=args.huber_delta).rename(
        columns={"y": "y_log", "PatchTST": "yhat_log_raw"}
    )

    key_cols = ["unique_id", "ds", "cutoff"]
    out = cls_cv[key_cols + ["p_nonzero_raw"]].merge(
        reg_cv[key_cols + ["yhat_log_raw"]],
        on=key_cols,
        how="inner",
    )
    out["ds"] = pd.to_datetime(out["ds"])
    out["cutoff"] = pd.to_datetime(out["cutoff"])
    out = out.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")

    out["p_nonzero"] = out["p_nonzero_raw"].clip(lower=0.0, upper=1.0)
    out["yhat_reg_only"] = np.expm1(out["yhat_log_raw"]).clip(lower=0.0)

    grid = parse_threshold_grid(args.threshold_grid)
    best_t = None
    best_acc = -1.0
    for t in grid:
        yhat_t = np.where(out["p_nonzero"].to_numpy() >= t, out["yhat_reg_only"].to_numpy(), 0.0)
        score = float(acc_vectorized(yhat_t, out["y"].to_numpy(dtype=float)).mean())
        if score > best_acc:
            best_acc = score
            best_t = t

    out["yhat_two_stage"] = np.where(out["p_nonzero"] >= best_t, out["yhat_reg_only"], 0.0).astype(float)

    rows = []
    for name, pred_col in [("RegOnly", "yhat_reg_only"), ("TwoStage", "yhat_two_stage")]:
        m = metrics_from_pred(out, pred_col)
        rows.append({"model": name, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df["best_threshold"] = np.where(metrics_df["model"] == "TwoStage", best_t, np.nan)

    out_cv = Path(args.cv_output)
    out_metrics = Path(args.metrics_output)
    out_cv.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_cv, index=False)
    metrics_df.to_csv(out_metrics, index=False)

    print(f"\nselected threshold={best_t:.4f} mean_acc={best_acc:.5f}")
    print("\nmetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nsaved cv: {out_cv}")
    print(f"saved metrics: {out_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTST 两阶段：先分类(y>0)再回归(log1p(y))。")
    parser.add_argument("--data", default="./data/data_cleaned.csv")
    parser.add_argument("--cv-output", default="./forecast_results_patchtst_2stage.csv")
    parser.add_argument("--metrics-output", default="./metrics_patchtst_2stage.csv")

    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=24)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--step-size", type=int, default=1)

    parser.add_argument("--patch-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scaler-type", default="robust", choices=["identity", "standard", "robust"])

    parser.add_argument("--cls-loss", choices=["mse", "mae", "huber"], default="mse")
    parser.add_argument("--reg-loss", choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument(
        "--threshold-grid",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="分类阈值候选，逗号分隔，范围[0,1]。",
    )

    main(parser.parse_args())
