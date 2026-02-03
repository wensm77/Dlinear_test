import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE, MSE
from neuralforecast.models import DLinear


def build_loss(loss_name: str, huber_delta: float):
    loss_name = loss_name.lower()
    if loss_name == "mae":
        return MAE()
    if loss_name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


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

    # 防止未来信息泄漏：如果 lag_ds 晚于 cutoff，则该位置不可用。
    out.loc[out["lag_ds"] > out["cutoff"], "SeasonalNaive"] = np.nan
    out["SeasonalNaive"] = out["SeasonalNaive"].fillna(0.0)
    out = out.drop(columns=["lag_ds"])
    return out


def add_fallback_flags(
    history_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    zero_ratio_threshold: float,
    nonzero_max: int,
):
    out = cv_df.copy()
    out["use_fallback"] = False

    hist = history_df[["unique_id", "ds", "y"]].copy()
    hist["is_zero"] = (hist["y"] == 0).astype("int8")
    hist["is_nonzero"] = (hist["y"] > 0).astype("int8")

    cutoffs = sorted(out["cutoff"].dropna().unique())
    for cutoff in cutoffs:
        hist_cut = hist[hist["ds"] <= cutoff]
        stats = hist_cut.groupby("unique_id", as_index=False).agg(
            zero_ratio=("is_zero", "mean"),
            nonzero_cnt=("is_nonzero", "sum"),
        )
        stats["is_intermittent"] = (stats["zero_ratio"] >= zero_ratio_threshold) | (
            stats["nonzero_cnt"] <= nonzero_max
        )
        inter_map = stats.set_index("unique_id")["is_intermittent"]

        cutoff_mask = out["cutoff"] == cutoff
        out.loc[cutoff_mask, "use_fallback"] = (
            out.loc[cutoff_mask, "unique_id"].map(inter_map).fillna(False).astype(bool).to_numpy()
        )

    return out


def build_segment_metrics(cv_df: pd.DataFrame, pred_cols: dict, high_volume_quantile: float = 0.8):
    uid_total = cv_df.groupby("unique_id")["y"].sum()
    volume_threshold = uid_total.quantile(high_volume_quantile)
    high_ids = set(uid_total[uid_total >= volume_threshold].index)

    segment_masks = {
        "all": np.ones(len(cv_df), dtype=bool),
        "intermittent": cv_df["use_fallback"].to_numpy(),
        "regular": (~cv_df["use_fallback"]).to_numpy(),
        "high_volume": cv_df["unique_id"].isin(high_ids).to_numpy(),
        "long_tail": (~cv_df["unique_id"].isin(high_ids)).to_numpy(),
    }

    rows = []
    for model_name, pred_col in pred_cols.items():
        for segment_name, mask in segment_masks.items():
            seg = cv_df[mask]
            if seg.empty:
                continue
            m = calc_metrics(seg["y"].to_numpy(), seg[pred_col].to_numpy())
            rows.append(
                {
                    "model": model_name,
                    "segment": segment_name,
                    "rows": int(len(seg)),
                    "series": int(seg["unique_id"].nunique()),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def train_and_backtest(args):
    print(f"读取数据: {args.data}")
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    dropped = len(id_counts) - len(valid_ids)
    df = df[df["unique_id"].isin(valid_ids)].copy()

    print(f"数据行数: {len(df)}")
    print(f"序列数: {df['unique_id'].nunique()} (过滤过短序列: {dropped})")
    print(f"最小长度要求: {min_len}")
    print(f"时间范围: {df['ds'].min().date()} -> {df['ds'].max().date()}")
    print(f"0 占比: {(df['y'] == 0).mean():.4f}")

    loss_fn = build_loss(args.loss, args.huber_delta)
    model = DLinear(
        h=args.horizon,
        input_size=args.input_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        scaler_type="robust",
        val_check_steps=args.val_check_steps,
        loss=loss_fn,
        valid_loss=loss_fn,
    )
    nf = NeuralForecast(models=[model], freq="MS")

    print("开始滚动回测...")
    cv_df = nf.cross_validation(
        df=df,
        n_windows=args.n_windows,
        step_size=args.step_size,
    )

    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    cv_df["DLinear"] = cv_df["DLinear"].clip(lower=0.0)

    cv_df = add_seasonal_naive(df, cv_df, season_length=args.season_length)
    cv_df = add_fallback_flags(
        df,
        cv_df,
        zero_ratio_threshold=args.fallback_zero_ratio,
        nonzero_max=args.fallback_nonzero_max,
    )

    cv_df["yhat_hybrid"] = np.where(cv_df["use_fallback"], cv_df["SeasonalNaive"], cv_df["DLinear"])
    cv_df["yhat_hybrid"] = cv_df["yhat_hybrid"].clip(lower=0.0)

    fallback_ratio = float(cv_df["use_fallback"].mean())
    print(f"回退比例 (SeasonalNaive): {fallback_ratio:.4f}")

    overall_rows = []
    for model_name, pred_col in {"DLinear": "DLinear", "Hybrid": "yhat_hybrid"}.items():
        m = calc_metrics(cv_df["y"].to_numpy(), cv_df[pred_col].to_numpy())
        overall_rows.append({"model": model_name, **m})
    overall_df = pd.DataFrame(overall_rows)

    segment_df = build_segment_metrics(
        cv_df,
        pred_cols={"DLinear": "DLinear", "Hybrid": "yhat_hybrid"},
        high_volume_quantile=args.high_volume_quantile,
    )

    cv_path = Path(args.cv_output)
    overall_path = Path(args.metrics_output)
    segment_path = Path(args.segment_metrics_output)
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    segment_path.parent.mkdir(parents=True, exist_ok=True)

    cv_df.to_csv(cv_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    segment_df.to_csv(segment_path, index=False)

    print("\n总体指标：")
    print(overall_df.to_string(index=False))
    print("\n分层指标（前10行）：")
    print(segment_df.head(10).to_string(index=False))
    print(f"\n已保存回测明细: {cv_path}")
    print(f"已保存总体指标: {overall_path}")
    print(f"已保存分层指标: {segment_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="改进版销量预测：DLinear + 间歇回退 + 分层评估。")
    parser.add_argument("--data", default="./data/data_cleaned.csv", help="清洗后数据路径")
    parser.add_argument("--cv-output", default="./forecast_results_improved.csv", help="回测明细输出路径")
    parser.add_argument("--metrics-output", default="./metrics_improved.csv", help="总体指标输出路径")
    parser.add_argument(
        "--segment-metrics-output",
        default="./metrics_segments_improved.csv",
        help="分层指标输出路径",
    )

    parser.add_argument("--horizon", type=int, default=12, help="预测步长（月）")
    parser.add_argument("--input-size", type=int, default=18, help="历史窗口长度（月）")
    parser.add_argument("--n-windows", type=int, default=2, help="滚动回测窗口数")
    parser.add_argument("--step-size", type=int, default=3, help="窗口滚动步长（月）")

    parser.add_argument("--max-steps", type=int, default=400, help="训练步数")
    parser.add_argument("--val-check-steps", type=int, default=100, help="验证检查步数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber", help="训练损失")
    parser.add_argument("--huber-delta", type=float, default=5.0, help="Huber 损失参数 delta")

    parser.add_argument("--season-length", type=int, default=12, help="SeasonalNaive 季节长度（月）")
    parser.add_argument(
        "--fallback-zero-ratio",
        type=float,
        default=0.8,
        help="零占比超过该阈值时使用 SeasonalNaive 回退",
    )
    parser.add_argument(
        "--fallback-nonzero-max",
        type=int,
        default=6,
        help="历史非零月份数不超过该阈值时使用 SeasonalNaive 回退",
    )
    parser.add_argument(
        "--high-volume-quantile",
        type=float,
        default=0.8,
        help="分层指标中 high_volume 分位点阈值",
    )

    train_and_backtest(parser.parse_args())
