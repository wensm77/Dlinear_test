import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE, MSE
from neuralforecast.models import NHiTS, PatchTST


def build_loss(loss_name: str, huber_delta: float):
    loss_name = loss_name.lower()
    if loss_name == "mae":
        return MAE()
    if loss_name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


def acc(f, a):
    if f == 0 and a == 0:
        return 1.0
    if f != 0 and a == 0:
        return 0.0
    return max(0.0, 1 - abs(f - a) / a)


def calc_metrics(y: np.ndarray, yhat: np.ndarray):
    y = y.astype(float)
    yhat = yhat.astype(float)
    abs_err = np.abs(y - yhat)
    sq_err = (y - yhat) ** 2
    denom = np.abs(y) + np.abs(yhat)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    abs_y_sum = float(np.abs(y).sum())
    wape = float(abs_err.sum() / abs_y_sum) if abs_y_sum > 0 else np.nan
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * abs_err / denom)))
    nz = y != 0
    mape_nonzero = float(np.mean(abs_err[nz] / np.abs(y[nz]))) if nz.any() else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "sMAPE": smape,
        "MAPE_nonzero": mape_nonzero,
        "BusinessAccuracy": float(1 - wape) if not np.isnan(wape) else np.nan,
    }


def add_seasonal_naive(history_df: pd.DataFrame, cv_df: pd.DataFrame, season_length: int = 12):
    out = cv_df.copy()
    out["lag_ds"] = out["ds"] - pd.DateOffset(months=season_length)
    lookup = history_df[["unique_id", "ds", "y"]].rename(columns={"ds": "lag_ds", "y": "SeasonalNaive"})
    out = out.merge(lookup, on=["unique_id", "lag_ds"], how="left")
    out.loc[out["lag_ds"] > out["cutoff"], "SeasonalNaive"] = np.nan
    out["SeasonalNaive"] = out["SeasonalNaive"].fillna(0.0)
    return out.drop(columns=["lag_ds"])


def add_fallback_flags(history_df: pd.DataFrame, cv_df: pd.DataFrame, zero_ratio_threshold: float, nonzero_max: int):
    out = cv_df.copy()
    out["use_fallback"] = False

    hist = history_df[["unique_id", "ds", "y"]].copy()
    hist["is_zero"] = (hist["y"] == 0).astype("int8")
    hist["is_nonzero"] = (hist["y"] > 0).astype("int8")

    for cutoff in sorted(out["cutoff"].dropna().unique()):
        hist_cut = hist[hist["ds"] <= cutoff]
        stats = hist_cut.groupby("unique_id", as_index=False).agg(
            zero_ratio=("is_zero", "mean"),
            nonzero_cnt=("is_nonzero", "sum"),
        )
        stats["is_intermittent"] = (stats["zero_ratio"] >= zero_ratio_threshold) | (
            stats["nonzero_cnt"] <= nonzero_max
        )
        inter_map = stats.set_index("unique_id")["is_intermittent"]
        mask = out["cutoff"] == cutoff
        out.loc[mask, "use_fallback"] = out.loc[mask, "unique_id"].map(inter_map).fillna(False).astype(bool).to_numpy()
    return out


def add_first_month_accuracy(metrics_df: pd.DataFrame, cv_df: pd.DataFrame, pred_cols: dict):
    first_idx = cv_df.groupby(["cutoff", "unique_id"])["ds"].idxmin()
    fm = cv_df.loc[first_idx].copy()
    for model_name, pred_col in pred_cols.items():
        fm[f"{model_name}_acc"] = fm.apply(lambda r: acc(float(r[pred_col]), float(r["y"])), axis=1)
        metrics_df.loc[metrics_df["model"] == model_name, "FirstMonthAccuracy"] = fm[f"{model_name}_acc"].mean()
    return metrics_df


def main(args):
    print(f"读取数据: {args.data}")
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    df = df[df["unique_id"].isin(valid_ids)].copy()
    print(f"序列数: {df['unique_id'].nunique()} | 行数: {len(df)} | 最小长度要求: {min_len}")

    loss_fn = build_loss(args.loss, args.huber_delta)
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    models = []
    if "nhits" in model_names:
        models.append(
            NHiTS(
                h=args.horizon,
                input_size=args.input_size,
                max_steps=args.max_steps,
                learning_rate=args.lr,
                scaler_type="robust",
                loss=loss_fn,
                valid_loss=loss_fn,
                val_check_steps=args.val_check_steps,
            )
        )
    if "patchtst" in model_names:
        models.append(
            PatchTST(
                h=args.horizon,
                input_size=args.input_size,
                max_steps=args.max_steps,
                learning_rate=args.lr,
                scaler_type="robust",
                loss=loss_fn,
                valid_loss=loss_fn,
                val_check_steps=args.val_check_steps,
            )
        )
    if not models:
        raise ValueError("至少指定一个模型：nhits 或 patchtst")

    nf = NeuralForecast(models=models, freq="MS")
    cv_df = nf.cross_validation(df=df, n_windows=args.n_windows, step_size=args.step_size)
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])

    pred_cols = {}
    for col in ["NHiTS", "PatchTST"]:
        if col in cv_df.columns:
            cv_df[col] = cv_df[col].clip(lower=0.0)
            pred_cols[col] = col

    if not pred_cols:
        raise ValueError("未在回测输出中找到深度模型预测列。")

    deep_cols = list(pred_cols.values())
    cv_df["yhat_deep_mean"] = cv_df[deep_cols].mean(axis=1).clip(lower=0.0)
    pred_cols["DeepMean"] = "yhat_deep_mean"

    cv_df = add_seasonal_naive(df, cv_df, season_length=args.season_length)
    cv_df = add_fallback_flags(df, cv_df, args.fallback_zero_ratio, args.fallback_nonzero_max)
    cv_df["yhat_deep_hybrid"] = np.where(cv_df["use_fallback"], cv_df["SeasonalNaive"], cv_df["yhat_deep_mean"])
    cv_df["yhat_deep_hybrid"] = cv_df["yhat_deep_hybrid"].clip(lower=0.0)
    pred_cols["DeepHybrid"] = "yhat_deep_hybrid"

    metrics_rows = []
    for model_name, pred_col in pred_cols.items():
        m = calc_metrics(cv_df["y"].to_numpy(), cv_df[pred_col].to_numpy())
        metrics_rows.append({"model": model_name, **m})
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = add_first_month_accuracy(metrics_df, cv_df, pred_cols)

    cv_path = Path(args.cv_output)
    metrics_path = Path(args.metrics_output)
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(cv_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print("\n总体指标：")
    print(metrics_df.to_string(index=False))
    print(f"\n回退比例: {float(cv_df['use_fallback'].mean()):.4f}")
    print(f"回测输出: {cv_path}")
    print(f"指标输出: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深度学习版本：NHiTS/PatchTST + 间歇回退融合。")
    parser.add_argument("--data", default="./data/data_cleaned.csv")
    parser.add_argument("--models", default="nhits,patchtst", help="逗号分隔: nhits,patchtst")
    parser.add_argument("--cv-output", default="./forecast_results_deep.csv")
    parser.add_argument("--metrics-output", default="./metrics_deep.csv")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=24)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--fallback-zero-ratio", type=float, default=0.8)
    parser.add_argument("--fallback-nonzero-max", type=int, default=6)
    main(parser.parse_args())
