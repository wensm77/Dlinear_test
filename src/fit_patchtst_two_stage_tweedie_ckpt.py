import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, HuberLoss, MAE, MSE
from neuralforecast.models import PatchTST


def build_cls_loss(loss_name: str, huber_delta: float):
    name = loss_name.lower()
    if name == "mae":
        return MAE()
    if name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


def build_reg_loss(loss_name: str, huber_delta: float, tweedie_rho: float):
    name = loss_name.lower()
    if name == "tweedie":
        return DistributionLoss(distribution="Tweedie", rho=tweedie_rho)
    if name == "mae":
        return MAE()
    if name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt_scaling(
    logits: np.ndarray, y: np.ndarray, max_iter: int = 30, sample: int = 200_000
):
    """Fit Platt scaling p = sigmoid(a*logit + b) with Newton updates on a subsample."""
    logits = logits.astype(np.float64)
    y = y.astype(np.float64)
    if sample and len(logits) > sample:
        idx = np.random.RandomState(0).choice(len(logits), size=sample, replace=False)
        logits = logits[idx]
        y = y[idx]

    a = 1.0
    b = 0.0
    for _ in range(max_iter):
        z = a * logits + b
        p = sigmoid(z)
        w = p * (1 - p)
        g_a = np.sum((p - y) * logits)
        g_b = np.sum(p - y)
        h_aa = np.sum(w * logits * logits) + 1e-6
        h_ab = np.sum(w * logits)
        h_bb = np.sum(w) + 1e-6
        det = h_aa * h_bb - h_ab * h_ab
        if det == 0:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (-h_ab * g_a + h_aa * g_b) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-6:
            break
    return float(a), float(b)


def acc_vectorized(f: np.ndarray, a: np.ndarray):
    """Business accuracy (strictly following your definition)."""
    out = np.zeros_like(f, dtype=float)
    both_zero = (f == 0) & (a == 0)
    bad = (f != 0) & (a == 0)
    normal = a != 0
    out[both_zero] = 1.0
    out[bad] = 0.0
    out[normal] = np.maximum(0.0, 1.0 - np.abs(f[normal] - a[normal]) / np.abs(a[normal]))
    return out


def parse_threshold_grid(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    vals = [v for v in vals if 0.0 <= v <= 1.0]
    if not vals:
        raise ValueError("threshold-grid 不能为空，且阈值应在 [0,1]。")
    return sorted(set(vals))


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


def _require_month_start(ts: pd.Timestamp, name: str):
    if ts.day != 1:
        raise ValueError(f"{name} 请使用每月第一天（例如 2025-09-01）。")


def build_hist(df_all: pd.DataFrame, ids: np.ndarray, cutoff: pd.Timestamp, input_size: int):
    """Return history <= cutoff for given ids, requiring ds==cutoff present and len>=input_size."""
    hist = df_all[(df_all["unique_id"].isin(ids)) & (df_all["ds"] <= cutoff)].copy()
    if hist.empty:
        return hist
    # must have observation at cutoff, otherwise predict date won't align to target month
    has_cutoff = hist.loc[hist["ds"] == cutoff, "unique_id"].unique()
    hist = hist[hist["unique_id"].isin(has_cutoff)].copy()
    if hist.empty:
        return hist
    counts = hist["unique_id"].value_counts()
    ok = counts[counts >= input_size].index
    return hist[hist["unique_id"].isin(ok)].copy()


def pick_prediction_column(pred_df: pd.DataFrame, alias: str):
    if alias in pred_df.columns:
        return alias
    if f"{alias}-median" in pred_df.columns:
        return f"{alias}-median"
    meta_cols = {"unique_id", "ds", "cutoff"}
    candidate_cols = [c for c in pred_df.columns if c not in meta_cols]
    if not candidate_cols:
        raise ValueError(f"无法在预测输出中找到数值列，columns={pred_df.columns.tolist()}")
    return candidate_cols[0]


def predict_val_window(
    *,
    df_all: pd.DataFrame,
    nf_cls: NeuralForecast,
    nf_reg: NeuralForecast,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    input_size: int,
    reg_target_transform: str,
):
    """Rolling 1-step forecasts for each month in [val_start, val_end]."""
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

        hist_cls = hist.copy()
        hist_cls["y"] = (hist_cls["y"] > 0).astype(float)
        hist_reg = hist.copy()
        if reg_target_transform == "log1p":
            hist_reg["y"] = np.log1p(hist_reg["y"])

        p1_raw = nf_cls.predict(df=hist_cls)
        p2_raw = nf_reg.predict(df=hist_reg)
        cls_col = pick_prediction_column(p1_raw, "cls")
        reg_col = pick_prediction_column(p2_raw, "reg")
        p1 = p1_raw.rename(columns={cls_col: "score_nonzero"})
        p2 = p2_raw.rename(columns={reg_col: "yhat_reg_model"})
        m = p1.merge(p2, on=["unique_id", "ds"], how="inner")
        m["cutoff"] = pd.Timestamp(cutoff)

        # only keep target month
        m = m[m["ds"] == target_ds].copy()
        if m.empty:
            continue
        m = m.merge(tg, on="unique_id", how="left")
        rows.append(m)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).sort_values(["cutoff", "unique_id"]).reset_index(drop=True)
    return out


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
        raise ValueError("val-end 不能早于 val-start。")
    if val_start <= train_end:
        raise ValueError("val-start 必须晚于 train-end（例如 train_end=2025-09-01, val_start=2025-10-01）。")

    # Avoid leakage: keep only <= val_end; train strictly on <= train_end.
    df_all = df_all[df_all["ds"] <= val_end].copy()
    df_train = df_all[df_all["ds"] <= train_end].copy()
    if df_train.empty:
        raise ValueError("训练集为空：请检查 train-end 是否在数据范围内。")

    out_dir = Path(args.save_dir)
    cls_dir = out_dir / "cls"
    reg_dir = out_dir / "reg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage-1 classifier (score-regression to {0,1}; no early stopping).
    cls_df = df_train.copy()
    cls_df["y"] = (cls_df["y"] > 0).astype(float)
    cls_loss = build_cls_loss(args.cls_loss, args.cls_huber_delta)
    cls_model = PatchTST(
        h=1,
        input_size=args.input_size,
        patch_len=args.patch_len,
        stride=args.stride,
        max_steps=args.cls_max_steps,
        learning_rate=args.lr,
        scaler_type=args.scaler_type,
        loss=cls_loss,
        valid_loss=cls_loss,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=-1,  # disable
        alias="cls",
    )
    nf_cls = NeuralForecast(models=[cls_model], freq="MS")
    nf_cls.fit(df=cls_df)
    nf_cls.save(path=str(cls_dir), overwrite=True, save_dataset=False)

    # Stage-2 regressor (supports Tweedie; no early stopping).
    reg_df = df_train.copy()
    reg_target_transform = "none" if args.reg_loss.lower() == "tweedie" else "log1p"
    if reg_target_transform == "log1p":
        reg_df["y"] = np.log1p(reg_df["y"])
    reg_loss = build_reg_loss(args.reg_loss, args.huber_delta, args.tweedie_rho)
    reg_model = PatchTST(
        h=1,
        input_size=args.input_size,
        patch_len=args.patch_len,
        stride=args.stride,
        max_steps=args.reg_max_steps,
        learning_rate=args.lr,
        scaler_type=args.scaler_type,
        loss=reg_loss,
        valid_loss=reg_loss,
        val_check_steps=args.val_check_steps,
        early_stop_patience_steps=-1,  # disable
        alias="reg",
    )
    nf_reg = NeuralForecast(models=[reg_model], freq="MS")
    nf_reg.fit(df=reg_df)
    nf_reg.save(path=str(reg_dir), overwrite=True, save_dataset=False)

    # Validation rolling predictions (always saved).
    val_base = predict_val_window(
        df_all=df_all,
        nf_cls=nf_cls,
        nf_reg=nf_reg,
        val_start=val_start,
        val_end=val_end,
        input_size=args.input_size,
        reg_target_transform=reg_target_transform,
    )
    if val_base.empty:
        raise ValueError("验证集窗口没有产生任何预测：请检查 input_size 或数据是否覆盖 cutoff 月份。")

    # Probabilities (raw and optional calibrated)
    logits = val_base["score_nonzero"].to_numpy(dtype=float)
    p_raw = np.clip(logits, 0.0, 1.0)
    val_base["p_nonzero"] = p_raw

    platt_a = None
    platt_b = None
    if args.calibrate:
        y_cls = (val_base["y"].to_numpy(dtype=float) > 0).astype(float)
        platt_a, platt_b = fit_platt_scaling(
            logits, y_cls, max_iter=args.calib_max_iter, sample=args.calib_sample
        )
        print(f"platt: a={platt_a:.4f} b={platt_b:.4f}")

    if args.use_calibrated_prob:
        if platt_a is None or platt_b is None:
            raise ValueError("use-calibrated-prob 需要 --calibrate 以得到 platt 参数。")
        p_used = sigmoid(platt_a * logits + platt_b)
    else:
        p_used = p_raw
    val_base["p_nonzero_cal"] = p_used

    # Regression to original scale
    if reg_target_transform == "log1p":
        val_base["yhat_reg_only"] = np.expm1(val_base["yhat_reg_model"]).clip(lower=0.0)
    else:
        val_base["yhat_reg_only"] = pd.to_numeric(
            val_base["yhat_reg_model"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)

    # Select threshold on validation window if not provided.
    grid = parse_threshold_grid(args.threshold_grid)
    threshold = float(args.selected_threshold) if args.selected_threshold >= 0 else None
    if threshold is None and args.select_threshold_on_val:
        reg = val_base["yhat_reg_only"].to_numpy(dtype=float)
        y = val_base["y"].to_numpy(dtype=float)
        best_t = None
        best = -1.0
        for t in grid:
            yhat = np.where(p_used >= t, reg, 0.0)
            score = float(acc_vectorized(yhat, y).mean())
            if score > best:
                best = score
                best_t = float(t)
        threshold = best_t
        print(f"threshold_on_val: best_t={threshold:.3f} mean_acc={best:.6f}")
    if threshold is None:
        threshold = float(args.threshold_fallback)

    val_base["selected_threshold"] = threshold
    val_base["yhat_two_stage"] = np.where(
        val_base["p_nonzero_cal"].to_numpy(dtype=float) >= threshold,
        val_base["yhat_reg_only"].to_numpy(dtype=float),
        0.0,
    ).astype(float)

    # Metrics (overall + by month)
    y = val_base["y"].to_numpy(dtype=float)
    yhat = val_base["yhat_two_stage"].to_numpy(dtype=float)
    overall_ba = float(acc_vectorized(yhat, y).mean())
    overall = {
        "rows": int(len(val_base)),
        "business_accuracy": overall_ba,
        "count_actual_zero": int((y == 0).sum()),
        "count_pred_zero": int((yhat == 0).sum()),
        **classification_stats(yhat, y),
    }

    def _month_metrics(g: pd.DataFrame):
        yy = g["y"].to_numpy(dtype=float)
        yh = g["yhat_two_stage"].to_numpy(dtype=float)
        m = {
            "rows": int(len(g)),
            "business_accuracy": float(acc_vectorized(yh, yy).mean()),
            "count_actual_zero": int((yy == 0).sum()),
            "count_pred_zero": int((yh == 0).sum()),
        }
        m.update(classification_stats(yh, yy))
        return pd.Series(m)

    by_month = (
        val_base.assign(month=val_base["ds"].dt.strftime("%Y%m"))
        .groupby("month", as_index=False)
        .apply(_month_metrics)
        .reset_index(drop=True)
    )

    Path(args.val_preds_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_metrics_output).parent.mkdir(parents=True, exist_ok=True)
    val_base.to_csv(args.val_preds_output, index=False)
    by_month.to_csv(args.val_metrics_output, index=False)

    params = {
        "train_end": str(train_end.date()),
        "val_start": str(val_start.date()),
        "val_end": str(val_end.date()),
        "input_size": int(args.input_size),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "scaler_type": args.scaler_type,
        "reg_loss": args.reg_loss,
        "reg_target_transform": reg_target_transform,
        "tweedie_rho": float(args.tweedie_rho),
        "use_calibrated_prob": bool(args.use_calibrated_prob),
        "platt_a": platt_a,
        "platt_b": platt_b,
        "selected_threshold": threshold,
        "val_overall": overall,
    }

    (out_dir / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"saved cls ckpt: {cls_dir}")
    print(f"saved reg ckpt: {reg_dir}")
    print(f"saved params:   {out_dir / 'params.json'}")
    print(f"saved val preds: {args.val_preds_output}")
    print(f"saved val metrics: {args.val_metrics_output}")
    print(f"val overall business_accuracy={overall_ba:.6f} rows={overall['rows']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="训练 two-stage PatchTST v2 并保存 checkpoint（cls + reg + params），并对验证窗口输出预测与业务准确率。"
    )
    p.add_argument("--data", default="./data/data_cleaned.csv")
    p.add_argument("--save-dir", default="./checkpoints/patchtst_2stage_v2")

    # Per request: fixed split to avoid leakage
    p.add_argument("--train-end", default="2025-09-01")
    p.add_argument("--val-start", default="2025-10-01")
    p.add_argument("--val-end", default="", help="可空，默认用数据最大月份（每月第一天）。")
    p.add_argument("--val-preds-output", default="./analysis/val_preds_two_stage_v2.csv")
    p.add_argument("--val-metrics-output", default="./analysis/val_metrics_two_stage_v2.csv")

    p.add_argument("--input-size", type=int, default=24)
    p.add_argument("--patch-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)

    p.add_argument("--cls-max-steps", type=int, default=5000)
    p.add_argument("--reg-max-steps", type=int, default=5000)
    p.add_argument("--val-check-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--scaler-type", default="robust", choices=["identity", "standard", "robust"]
    )

    p.add_argument("--cls-loss", choices=["mse", "mae", "huber"], default="mse")
    p.add_argument("--cls-huber-delta", type=float, default=1.0)
    p.add_argument("--reg-loss", choices=["mse", "mae", "huber", "tweedie"], default="tweedie")
    p.add_argument("--huber-delta", type=float, default=5.0)
    p.add_argument("--tweedie-rho", type=float, default=1.5, help="Tweedie power parameter rho in (1, 2).")

    # Thresholding
    p.add_argument(
        "--selected-threshold",
        type=float,
        default=-1.0,
        help="手动指定门控阈值；<0 则（若 select-threshold-on-val）在验证集上选。",
    )
    p.add_argument("--threshold-fallback", type=float, default=0.5)
    p.add_argument("--select-threshold-on-val", action="store_true", help="在验证窗口上选单一阈值。")

    # Optional calibration
    p.add_argument("--use-calibrated-prob", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="在验证窗口上做 Platt 校准（不会泄漏到训练）。")
    p.add_argument("--calib-max-iter", type=int, default=30)
    p.add_argument("--calib-sample", type=int, default=200_000)

    p.add_argument(
        "--threshold-grid",
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )

    main(p.parse_args())
