import argparse
import json
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


def main(args):
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    # series-level filtering (same semantics as train_patchtst_two_stage_v2.py)
    if not args.disable_series_filter:
        max_y = df.groupby("unique_id")["y"].max()
        zero_ratio = df["y"].eq(0).groupby(df["unique_id"]).mean()
        keep_ids = max_y.index[
            (max_y <= args.series_max_y) & (zero_ratio < args.series_max_zero_ratio)
        ]
        df = df[df["unique_id"].isin(keep_ids)].copy()

    # Ensure enough length for 1-step prediction after cutoff. (Training itself can use all history.)
    counts = df["unique_id"].value_counts()
    keep_ids = counts[counts >= args.input_size + 1].index
    df = df[df["unique_id"].isin(keep_ids)].copy()

    out_dir = Path(args.save_dir)
    cls_dir = out_dir / "cls"
    reg_dir = out_dir / "reg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage-1 classifier: y in {0,1}
    cls_df = df.copy()
    cls_df["y"] = (cls_df["y"] > 0).astype(float)
    cls_loss = build_loss(args.cls_loss, args.cls_huber_delta)
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
        early_stop_patience_steps=args.early_stop_patience_steps,
        alias="cls",
    )
    nf_cls = NeuralForecast(models=[cls_model], freq="MS")
    nf_cls.fit(df=cls_df)
    nf_cls.save(path=str(cls_dir), overwrite=True, save_dataset=False)

    # Stage-2 regressor: log1p(y)
    reg_df = df.copy()
    reg_df["y"] = np.log1p(reg_df["y"])
    reg_loss = build_loss(args.reg_loss, args.huber_delta)
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
        early_stop_patience_steps=args.early_stop_patience_steps,
        alias="reg",
    )
    nf_reg = NeuralForecast(models=[reg_model], freq="MS")
    nf_reg.fit(df=reg_df)
    nf_reg.save(path=str(reg_dir), overwrite=True, save_dataset=False)

    params = {
        "input_size": args.input_size,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "scaler_type": args.scaler_type,
        "use_calibrated_prob": bool(args.use_calibrated_prob),
        "selected_threshold": float(args.selected_threshold) if args.selected_threshold >= 0 else None,
        "platt_a": None,
        "platt_b": None,
        "prob_col": "p_nonzero_cal" if args.use_calibrated_prob else "p_nonzero",
    }

    # Optional: calibrate (Platt) + select threshold on a holdout "tail" slice.
    if args.calibrate:
        # Use the last N months per series as calibration targets (and only use history before them).
        tail_n = int(args.calib_tail_months)
        grid = parse_threshold_grid(args.threshold_grid)

        # Build a simple cutoff-based calibration set: for each uid, pick the last tail_n months.
        # For month t, cutoff is t-1 month.
        cal_targets = []
        for uid, g in df.groupby("unique_id", sort=False):
            g = g.sort_values("ds")
            if len(g) <= args.input_size + tail_n:
                continue
            cal_targets.append(g.tail(tail_n)[["unique_id", "ds", "y"]])
        if not cal_targets:
            raise ValueError("calibrate 失败：没有足够长的序列生成校准集。")
        cal_targets = pd.concat(cal_targets, ignore_index=True)
        cal_targets["cutoff"] = cal_targets["ds"] - pd.offsets.MonthBegin(1)

        # Run prediction grouped by cutoff (so each prediction uses <= cutoff history).
        preds = []
        for cutoff, tg in cal_targets.groupby("cutoff", sort=True):
            ids = tg["unique_id"].unique().tolist()
            hist = df[(df["unique_id"].isin(ids)) & (df["ds"] <= cutoff)].copy()
            # Require at least input_size points
            c = hist["unique_id"].value_counts()
            ok = c[c >= args.input_size].index
            hist = hist[hist["unique_id"].isin(ok)].copy()
            if hist.empty:
                continue
            p1 = nf_cls.predict(df=hist).rename(columns={"cls": "score_nonzero"})
            p2 = nf_reg.predict(df=hist).rename(columns={"reg": "yhat_log"})
            p1["cutoff"] = pd.Timestamp(cutoff)
            p2["cutoff"] = pd.Timestamp(cutoff)
            preds.append(
                p1.merge(p2, on=["unique_id", "ds"], how="inner").merge(
                    tg, on=["unique_id", "ds"], how="inner"
                )
            )
        if not preds:
            raise ValueError("calibrate 失败：无法产生任何预测用于校准。")
        cal = pd.concat(preds, ignore_index=True)

        logits = cal["score_nonzero"].to_numpy(dtype=float)
        y_cls = (cal["y"].to_numpy(dtype=float) > 0).astype(float)
        a, b = fit_platt_scaling(logits, y_cls, max_iter=args.calib_max_iter, sample=args.calib_sample)
        p_raw = np.clip(logits, 0.0, 1.0)
        p_cal = sigmoid(a * logits + b)

        reg = np.expm1(cal["yhat_log"].to_numpy(dtype=float)).clip(min=0.0)
        y = cal["y"].to_numpy(dtype=float)

        best_t = None
        best = -1.0
        p_used = p_cal if args.use_calibrated_prob else p_raw
        for t in grid:
            yhat = np.where(p_used >= t, reg, 0.0)
            score = float(acc_vectorized(yhat, y).mean())
            if score > best:
                best = score
                best_t = float(t)

        params["platt_a"] = float(a)
        params["platt_b"] = float(b)
        params["selected_threshold"] = float(best_t)
        params["calib_mean_acc"] = float(best)
        params["calib_tail_months"] = int(tail_n)

        print(f"calibrate: platt=(a={a:.4f}, b={b:.4f}) best_t={best_t:.3f} mean_acc={best:.5f}")

    params_path = out_dir / "params.json"
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved cls ckpt: {cls_dir}")
    print(f"saved reg ckpt: {reg_dir}")
    print(f"saved params:   {params_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="训练 two-stage PatchTST v2 并保存 checkpoint（cls + reg + params）。")
    p.add_argument("--data", default="./data/data_cleaned.csv")
    p.add_argument("--save-dir", default="./checkpoints/patchtst_2stage_v2")

    p.add_argument("--input-size", type=int, default=24)
    p.add_argument("--patch-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)

    p.add_argument("--cls-max-steps", type=int, default=5000)
    p.add_argument("--reg-max-steps", type=int, default=5000)
    p.add_argument("--val-check-steps", type=int, default=200)
    p.add_argument("--early-stop-patience-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scaler-type", default="robust", choices=["identity", "standard", "robust"])

    p.add_argument("--cls-loss", choices=["mse", "mae", "huber"], default="mse")
    p.add_argument("--cls-huber-delta", type=float, default=1.0)
    p.add_argument("--reg-loss", choices=["mse", "mae", "huber"], default="huber")
    p.add_argument("--huber-delta", type=float, default=5.0)

    # Same filter semantics as training script
    p.add_argument("--disable-series-filter", action="store_true")
    p.add_argument("--series-max-y", type=float, default=100.0)
    p.add_argument("--series-max-zero-ratio", type=float, default=0.4)

    # threshold & calibration
    p.add_argument("--selected-threshold", type=float, default=-1.0, help="手动指定门控阈值；<0 则（若 calibrate）自动选择。")
    p.add_argument("--use-calibrated-prob", action="store_true")
    p.add_argument("--calibrate", action="store_true", help="在训练后做一次 tail 校准：Platt + 阈值选择。")
    p.add_argument("--calib-tail-months", type=int, default=6)
    p.add_argument("--calib-max-iter", type=int, default=30)
    p.add_argument("--calib-sample", type=int, default=200_000)
    p.add_argument(
        "--threshold-grid",
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )

    main(p.parse_args())

