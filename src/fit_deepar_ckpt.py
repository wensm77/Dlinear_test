import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import DeepAR


def acc_vectorized(f: np.ndarray, a: np.ndarray):
    out = np.zeros_like(f, dtype=float)
    both_zero = (f == 0) & (a == 0)
    bad = (f != 0) & (a == 0)
    normal = a != 0
    out[both_zero] = 1.0
    out[bad] = 0.0
    out[normal] = np.maximum(0.0, 1.0 - np.abs(f[normal] - a[normal]) / np.abs(a[normal]))
    return out


def classification_stats(pred: np.ndarray, actual: np.ndarray):
    pred_pos = pred > 0
    actual_pos = actual > 0
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision == precision and recall == recall and (precision + recall))
        else np.nan
    )
    return {
        "cls_accuracy": float(acc) if acc == acc else np.nan,
        "f1_score": float(f1) if f1 == f1 else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def pick_prediction_column(pred_df: pd.DataFrame, alias: str):
    if alias in pred_df.columns:
        return alias
    if f"{alias}-median" in pred_df.columns:
        return f"{alias}-median"
    meta_cols = {"unique_id", "ds", "cutoff"}
    candidate_cols = [c for c in pred_df.columns if c not in meta_cols]
    if not candidate_cols:
        raise ValueError(f"No forecast column found in prediction output: {pred_df.columns.tolist()}")
    return candidate_cols[0]


def _require_month_start(ts: pd.Timestamp, name: str):
    if ts.day != 1:
        raise ValueError(f"{name} must be month-start, for example 2025-09-01")


def build_hist(df_all: pd.DataFrame, ids: np.ndarray, cutoff: pd.Timestamp, input_size: int):
    hist = df_all[(df_all["unique_id"].isin(ids)) & (df_all["ds"] <= cutoff)].copy()
    if hist.empty:
        return hist
    has_cutoff = hist.loc[hist["ds"] == cutoff, "unique_id"].unique()
    hist = hist[hist["unique_id"].isin(has_cutoff)].copy()
    if hist.empty:
        return hist
    counts = hist["unique_id"].value_counts()
    ok = counts[counts >= input_size].index
    return hist[hist["unique_id"].isin(ok)].copy()


def predict_val_window(
    *,
    df_all: pd.DataFrame,
    nf: NeuralForecast,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    input_size: int,
):
    rows = []
    cutoffs = pd.date_range(val_start - pd.offsets.MonthBegin(1), val_end - pd.offsets.MonthBegin(1), freq="MS")
    for cutoff in cutoffs:
        target_ds = cutoff + pd.offsets.MonthBegin(1)
        tg = df_all[df_all["ds"] == target_ds][["unique_id", "y"]].copy()
        if tg.empty:
            continue
        hist = build_hist(df_all, tg["unique_id"].unique(), cutoff, input_size)
        if hist.empty:
            continue
        pred_raw = nf.predict(df=hist)
        pred_col = pick_prediction_column(pred_raw, "deepar")
        pred = pred_raw.rename(columns={pred_col: "yhat"})
        pred["cutoff"] = pd.Timestamp(cutoff)
        pred = pred[pred["ds"] == target_ds].copy()
        if pred.empty:
            continue
        pred = pred.merge(tg, on="unique_id", how="left")
        rows.append(pred)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["cutoff", "unique_id"]).reset_index(drop=True)


def main(args):
    df_all = pd.read_csv(args.data)
    df_all["ds"] = pd.to_datetime(df_all["ds"])
    df_all["y"] = pd.to_numeric(df_all["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df_all = df_all[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    train_end = pd.Timestamp(args.train_end)
    val_start = pd.Timestamp(args.val_start)
    val_end = pd.Timestamp(args.val_end) if args.val_end else df_all["ds"].max()
    _require_month_start(train_end, "train-end")
    _require_month_start(val_start, "val-start")
    _require_month_start(val_end, "val-end")
    if val_end < val_start:
        raise ValueError("val-end cannot be earlier than val-start")
    if val_start <= train_end:
        raise ValueError("val-start must be later than train-end")

    # No leakage: train only with ds <= train_end.
    df_all = df_all[df_all["ds"] <= val_end].copy()
    df_train = df_all[df_all["ds"] <= train_end].copy()
    if df_train.empty:
        raise ValueError("Train set is empty with current split")

    loss = DistributionLoss(distribution=args.distribution)
    model = DeepAR(
        h=1,
        input_size=args.input_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        scaler_type=args.scaler_type,
        loss=loss,
        valid_loss=loss,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=-1,
        alias="deepar",
    )
    nf = NeuralForecast(models=[model], freq="MS")
    nf.fit(df=df_train)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(out_dir), overwrite=True, save_dataset=False)

    val_pred = predict_val_window(
        df_all=df_all,
        nf=nf,
        val_start=val_start,
        val_end=val_end,
        input_size=args.input_size,
    )
    if val_pred.empty:
        raise ValueError("No validation predictions were generated; check input_size and data coverage")

    val_pred["yhat"] = pd.to_numeric(val_pred["yhat"], errors="coerce").fillna(0.0).clip(lower=0.0)

    y = val_pred["y"].to_numpy(dtype=float)
    yhat = val_pred["yhat"].to_numpy(dtype=float)
    overall_ba = float(acc_vectorized(yhat, y).mean())
    overall = {
        "rows": int(len(val_pred)),
        "business_accuracy": overall_ba,
        "business_accuracy_nonzero": float(acc_vectorized(yhat, y)[y > 0].mean()) if (y > 0).any() else np.nan,
        "count_actual_zero": int((y == 0).sum()),
        "count_pred_zero": int((yhat == 0).sum()),
        **classification_stats(yhat, y),
    }

    def _month_metrics(g: pd.DataFrame):
        yy = g["y"].to_numpy(dtype=float)
        yh = g["yhat"].to_numpy(dtype=float)
        accs = acc_vectorized(yh, yy)
        m = {
            "rows": int(len(g)),
            "business_accuracy": float(accs.mean()),
            "business_accuracy_nonzero": float(accs[yy > 0].mean()) if (yy > 0).any() else np.nan,
            "count_actual_zero": int((yy == 0).sum()),
            "count_pred_zero": int((yh == 0).sum()),
        }
        m.update(classification_stats(yh, yy))
        return pd.Series(m)

    by_month = (
        val_pred.assign(month=val_pred["ds"].dt.strftime("%Y%m"))
        .groupby("month", as_index=False)
        .apply(_month_metrics)
        .reset_index(drop=True)
    )

    val_preds_output = Path(args.val_preds_output)
    val_metrics_output = Path(args.val_metrics_output)
    val_preds_output.parent.mkdir(parents=True, exist_ok=True)
    val_metrics_output.parent.mkdir(parents=True, exist_ok=True)
    val_pred.to_csv(val_preds_output, index=False)
    by_month.to_csv(val_metrics_output, index=False)

    params = {
        "model": "DeepAR",
        "distribution": args.distribution,
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "input_size": int(args.input_size),
        "scaler_type": args.scaler_type,
        "val_overall": overall,
    }
    (out_dir / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved ckpt: {out_dir}")
    print(f"saved val preds: {val_preds_output}")
    print(f"saved val metrics: {val_metrics_output}")
    print(f"val overall business_accuracy={overall_ba:.6f} rows={overall['rows']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train DeepAR ckpt and evaluate on validation window (no leakage split).")
    p.add_argument("--data", default="./data/data_cleaned.csv")
    p.add_argument("--save-dir", default="./checkpoints/deepar_monthly")

    p.add_argument("--train-end", default="2025-09-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="", help="Optional; defaults to max ds in data")
    p.add_argument("--val-preds-output", default="./analysis/val_preds_deepar.csv")
    p.add_argument("--val-metrics-output", default="./analysis/val_metrics_deepar.csv")

    p.add_argument("--input-size", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--val-check-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scaler-type", default="robust", choices=["identity", "standard", "robust"])
    p.add_argument(
        "--distribution",
        default="Tweedie",
        choices=["Normal", "StudentT", "Poisson", "NegativeBinomial", "Tweedie"],
        help="Distribution used by DeepAR",
    )
    main(p.parse_args())
