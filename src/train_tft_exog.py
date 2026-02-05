import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE, MSE
from neuralforecast.models import TFT


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
    return max(0.0, 1.0 - abs(f - a) / a)


def calc_metrics(y: np.ndarray, yhat: np.ndarray):
    y = y.astype(float)
    yhat = yhat.astype(float)
    abs_err = np.abs(y - yhat)
    sq_err = (y - yhat) ** 2
    denom = np.abs(y) + np.abs(yhat)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt(sq_err.mean()))
    wape = float(abs_err.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else np.nan
    smape = float(np.mean(np.where(denom == 0, 0.0, 2 * abs_err / denom)))
    mean_acc = float(np.mean([acc(f, a) for f, a in zip(yhat, y)])) if len(y) else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "sMAPE": smape,
        "BusinessAccuracy": float(1 - wape) if not np.isnan(wape) else np.nan,
        "MeanAccuracy": mean_acc,
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
        stats["is_intermittent"] = (stats["zero_ratio"] >= zero_ratio_threshold) | (stats["nonzero_cnt"] <= nonzero_max)
        mask = out["cutoff"] == cutoff
        inter_map = stats.set_index("unique_id")["is_intermittent"]
        out.loc[mask, "use_fallback"] = out.loc[mask, "unique_id"].map(inter_map).fillna(False).astype(bool).to_numpy()
    return out


def add_first_month_accuracy(metrics_df: pd.DataFrame, cv_df: pd.DataFrame, pred_cols: dict):
    first_idx = cv_df.groupby(["cutoff", "unique_id"])["ds"].idxmin()
    first = cv_df.loc[first_idx].copy()
    for model_name, pred_col in pred_cols.items():
        first[f"{model_name}_acc"] = first.apply(lambda r: acc(float(r[pred_col]), float(r["y"])), axis=1)
        metrics_df.loc[metrics_df["model"] == model_name, "FirstMonthAccuracy"] = first[f"{model_name}_acc"].mean()
    return metrics_df


def main(args):
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).copy()

    required_cols = [
        "unique_id",
        "ds",
        "y",
        "flow_fast",
        "month_sin",
        "month_cos",
        "holiday_days_in_month",
        "is_padded_month",
        "invntry_cnt",
        "supply_lt",
        "transportation_lt",
        "prchs_cycl",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_nonzero_6",
    ]
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for c in required_cols:
        if c not in {"unique_id", "ds"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    hist_exog_list = [
        "is_padded_month",
        "invntry_cnt",
        "supply_lt",
        "transportation_lt",
        "prchs_cycl",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_nonzero_6",
    ]
    futr_exog_list = ["month_sin", "month_cos", "holiday_days_in_month", "flow_fast"]

    model_cols = ["unique_id", "ds", "y"] + hist_exog_list + futr_exog_list
    df = df[model_cols].copy()

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    dropped = len(id_counts) - len(valid_ids)
    df = df[df["unique_id"].isin(valid_ids)].copy()

    fill_cols = hist_exog_list + futr_exog_list
    df[fill_cols] = df[fill_cols].fillna(0.0)

    print(f"rows={len(df)} uids={df['unique_id'].nunique()} dropped_short={dropped} min_len={min_len}")
    print(f"date={df['ds'].min().date()}->{df['ds'].max().date()} zero_ratio={(df['y']==0).mean():.4f}")

    loss_fn = build_loss(args.loss, args.huber_delta)
    model = TFT(
        h=args.horizon,
        input_size=args.input_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        scaler_type="robust",
        hist_exog_list=hist_exog_list,
        futr_exog_list=futr_exog_list,
        hidden_size=args.hidden_size,
        loss=loss_fn,
        valid_loss=loss_fn,
        val_check_steps=args.val_check_steps,
    )

    nf = NeuralForecast(models=[model], freq="MS")
    cv_df = nf.cross_validation(df=df, n_windows=args.n_windows, step_size=args.step_size)
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    cv_df["TFT"] = cv_df["TFT"].clip(lower=0.0)

    cv_df = add_seasonal_naive(df, cv_df, season_length=args.season_length)
    cv_df = add_fallback_flags(df, cv_df, args.fallback_zero_ratio, args.fallback_nonzero_max)
    cv_df["yhat_tft_hybrid"] = np.where(cv_df["use_fallback"], cv_df["SeasonalNaive"], cv_df["TFT"])
    cv_df["yhat_tft_hybrid"] = cv_df["yhat_tft_hybrid"].clip(lower=0.0)

    pred_cols = {"TFT": "TFT", "TFTHybrid": "yhat_tft_hybrid"}
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

    print("\nmetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nfallback_ratio={float(cv_df['use_fallback'].mean()):.4f}")
    print(f"saved cv: {cv_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFT + 精选特征（月度）预测。")
    parser.add_argument("--data", default="./data/monthly_feature_panel.csv")
    parser.add_argument("--cv-output", default="./forecast_results_tft.csv")
    parser.add_argument("--metrics-output", default="./metrics_tft.csv")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=24)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--fallback-zero-ratio", type=float, default=0.8)
    parser.add_argument("--fallback-nonzero-max", type=int, default=6)
    main(parser.parse_args())
