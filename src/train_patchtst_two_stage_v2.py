import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss, MAE, MSE
from neuralforecast.models import PatchTST


def build_reg_loss(loss_name: str, huber_delta: float):
    name = loss_name.lower()
    if name == "mae":
        return MAE()
    if name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()

def build_cls_loss(loss_name: str, huber_delta: float):
    """NeuralForecast-native loss for classification stage.

    PatchTST in this NeuralForecast version doesn't support a native BCE loss,
    so we model classification as score-regression to {0,1} and then calibrate.
    """
    name = loss_name.lower()
    if name == "mae":
        return MAE()
    if name == "huber":
        return HuberLoss(delta=huber_delta)
    return MSE()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt_scaling(logits: np.ndarray, y: np.ndarray, max_iter: int = 30, sample: int = 200_000):
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
        # gradients
        g_a = np.sum((p - y) * logits)
        g_b = np.sum(p - y)
        # hessian
        h_aa = np.sum(w * logits * logits)
        h_ab = np.sum(w * logits)
        h_bb = np.sum(w)
        # damping for numerical stability
        h_aa += 1e-6
        h_bb += 1e-6
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


def metrics_from_pred(df: pd.DataFrame, pred_col: str):
    y = df["y"].to_numpy(dtype=float)
    yhat = df[pred_col].to_numpy(dtype=float)
    abs_err = np.abs(y - yhat)
    denom = np.abs(y) + np.abs(yhat)

    first_idx = df.groupby(["cutoff", "unique_id"])["ds"].idxmin()
    first = df.loc[first_idx]

    actual_pos = y > 0
    pred_pos = yhat > 0
    tn = int((~pred_pos & ~actual_pos).sum())
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())

    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision == precision and recall == recall and (precision + recall) else np.nan

    wape = float(abs_err.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else np.nan
    return {
        "MeanAccuracy": float(acc_vectorized(yhat, y).mean()),
        "FirstMonthAccuracy": float(acc_vectorized(first[pred_col].to_numpy(dtype=float), first["y"].to_numpy(dtype=float)).mean()),
        "WAPE": wape,
        "BusinessAccuracy": float(1 - wape) if not np.isnan(wape) else np.nan,
        "ClsAccuracy": float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) else np.nan,
        "F1": float(f1) if not np.isnan(f1) else np.nan,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def parse_threshold_grid(s: str):
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    vals = [v for v in vals if 0.0 <= v <= 1.0]
    if not vals:
        raise ValueError("threshold-grid 不能为空，且阈值应在 [0,1]。")
    return sorted(set(vals))


def run_cv_patchtst(df: pd.DataFrame, args, loss):
    # NOTE: NeuralForecast models are step-driven (max_steps). In many setups the "epoch"
    # counter in logs stays at 0/1 because the trainer stops by steps, which is expected.
    trainer_kwargs = {}
    if args.wandb_project:
        trainer_kwargs["logger"] = args._wandb_logger  # set at runtime
        trainer_kwargs["callbacks"] = args._trainer_callbacks  # set at runtime
        trainer_kwargs["log_every_n_steps"] = args.log_every_n_steps
        trainer_kwargs["enable_checkpointing"] = False
        trainer_kwargs["enable_model_summary"] = False

    model = PatchTST(
        h=args.horizon,
        input_size=args.input_size,
        patch_len=args.patch_len,
        stride=args.stride,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        scaler_type=args.scaler_type,
        loss=loss,
        valid_loss=loss,
        val_check_steps=args.val_check_steps,
        # Enable early stopping when >= 0; NF default (-1) disables.
        early_stop_patience_steps=args.early_stop_patience_steps,
        **trainer_kwargs,
    )
    nf = NeuralForecast(models=[model], freq="MS")
    return nf.cross_validation(df=df, n_windows=args.n_windows, step_size=args.step_size)


def main(args):
    df = pd.read_csv(args.data)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    # Optional: filter out "hard" / out-of-scope series by series-level stats.
    # This is series-level (per unique_id) filtering on the available history.
    # Note: in cross-validation this uses the full series; if you want "as-of-cutoff"
    # filtering, we'd need to compute stats per window/cutoff (more complex).
    if not args.disable_series_filter:
        max_y = df.groupby("unique_id")["y"].max()
        zero_ratio = df["y"].eq(0).groupby(df["unique_id"]).mean()

        keep_ids = max_y.index[
            (max_y <= args.series_max_y) & (zero_ratio < args.series_max_zero_ratio)
        ]
        before_uids = df["unique_id"].nunique()
        df = df[df["unique_id"].isin(keep_ids)].copy()
        after_uids = df["unique_id"].nunique()
        print(
            f"series_filter: enabled max_y<={args.series_max_y} zero_ratio<{args.series_max_zero_ratio} "
            f"kept_uids={after_uids}/{before_uids}"
        )
    else:
        print("series_filter: disabled")

    min_len = args.input_size + args.horizon + (args.n_windows - 1) * args.step_size
    id_counts = df["unique_id"].value_counts()
    valid_ids = id_counts[id_counts >= min_len].index
    df = df[df["unique_id"].isin(valid_ids)].copy()

    tokens = (args.input_size - args.patch_len) // args.stride + 1 if args.input_size >= args.patch_len else 0
    print(f"rows={len(df)} uids={df['unique_id'].nunique()} min_len={min_len} zero_ratio={(df['y']==0).mean():.4f}")
    print(f"PatchTST tokens={tokens} (L={args.input_size}, patch_len={args.patch_len}, stride={args.stride})")

    # W&B logging (optional): a single run, log grad_norm for cls/reg with different keys.
    if args.wandb_project:
        try:
            from lightning.pytorch.callbacks import Callback
            from lightning.pytorch.loggers import WandbLogger
        except Exception:
            from pytorch_lightning.callbacks import Callback
            from pytorch_lightning.loggers import WandbLogger
        import wandb

        class GradNormLogger(Callback):
            def __init__(self, key_prefix: str):
                super().__init__()
                self.key_prefix = key_prefix

            def on_before_optimizer_step(self, trainer, pl_module, optimizer):
                total = 0.0
                for p in pl_module.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.detach().data.norm(2).item()
                    total += param_norm * param_norm
                grad_norm = total**0.5
                pl_module.log(f"{self.key_prefix}/grad_norm", grad_norm, on_step=True, logger=True, prog_bar=False)

        # Initialize run once; reuse logger for both trainers.
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name,
            tags=tags or None,
            save_dir=args.wandb_dir,
        )
        wandb_logger.experiment.config.update(
            {
                "horizon": args.horizon,
                "input_size": args.input_size,
                "patch_len": args.patch_len,
                "stride": args.stride,
                "n_windows": args.n_windows,
                "step_size": args.step_size,
                "max_steps": args.max_steps,
                "val_check_steps": args.val_check_steps,
                "early_stop_patience_steps": args.early_stop_patience_steps,
                "lr": args.lr,
                "scaler_type": args.scaler_type,
                "series_filter_disabled": bool(args.disable_series_filter),
                "series_max_y": args.series_max_y,
                "series_max_zero_ratio": args.series_max_zero_ratio,
            },
            allow_val_change=True,
        )
    else:
        wandb_logger = None

    # Stage-1 (classification): score-regression to {0,1} with NF-native losses,
    # then probability calibration (Platt scaling) on that score.
    cls_df = df.copy()
    cls_df["y"] = (cls_df["y"] > 0).astype(float)
    cls_loss = build_cls_loss(args.cls_loss, args.cls_huber_delta)
    args._wandb_logger = wandb_logger
    args._trainer_callbacks = [GradNormLogger("cls")] if wandb_logger else None
    cls_cv = run_cv_patchtst(cls_df, args, loss=cls_loss).rename(columns={"y": "y_cls", "PatchTST": "score_nonzero"})

    # Convert score -> raw prob (clipped), then Platt scaling calibration.
    logits = cls_cv["score_nonzero"].to_numpy(dtype=float)
    y_cls = cls_cv["y_cls"].to_numpy(dtype=float)
    a, b = fit_platt_scaling(logits, y_cls, max_iter=args.calib_max_iter, sample=args.calib_sample)
    cls_cv["p_nonzero"] = np.clip(logits, 0.0, 1.0)
    cls_cv["p_nonzero_cal"] = sigmoid(a * logits + b)
    cls_cv["platt_a"] = a
    cls_cv["platt_b"] = b
    print(f"classifier loss={args.cls_loss} platt=(a={a:.4f}, b={b:.4f})")

    # Stage-2 (regression): log1p(y) + robust loss.
    reg_df = df.copy()
    reg_df["y"] = np.log1p(reg_df["y"])
    reg_loss = build_reg_loss(args.reg_loss, args.huber_delta)
    args._trainer_callbacks = [GradNormLogger("reg")] if wandb_logger else None
    reg_cv = run_cv_patchtst(reg_df, args, loss=reg_loss).rename(columns={"y": "y_log", "PatchTST": "yhat_log"})

    key_cols = ["unique_id", "ds", "cutoff"]
    out = cls_cv[key_cols + ["y_cls", "score_nonzero", "p_nonzero", "p_nonzero_cal", "platt_a", "platt_b"]].merge(
        reg_cv[key_cols + ["yhat_log"]],
        on=key_cols,
        how="inner",
    )
    out = out.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
    out["ds"] = pd.to_datetime(out["ds"])
    out["cutoff"] = pd.to_datetime(out["cutoff"])

    out["yhat_reg_only"] = np.expm1(out["yhat_log"]).clip(lower=0.0)

    grid = parse_threshold_grid(args.threshold_grid)
    best_t = None
    best_score = -1.0
    prob_col = "p_nonzero_cal" if args.use_calibrated_prob else "p_nonzero"
    p = out[prob_col].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    reg = out["yhat_reg_only"].to_numpy(dtype=float)
    for t in grid:
        yhat = np.where(p >= t, reg, 0.0)
        score = float(acc_vectorized(yhat, y).mean())
        if score > best_score:
            best_score = score
            best_t = float(t)

    out["yhat_two_stage"] = np.where(out[prob_col] >= best_t, out["yhat_reg_only"], 0.0).astype(float)
    out["selected_threshold"] = best_t
    out["prob_used"] = prob_col

    metrics = [
        {"model": "RegOnly", **metrics_from_pred(out, "yhat_reg_only")},
        {"model": "TwoStage", **metrics_from_pred(out, "yhat_two_stage"), "selected_threshold": best_t, "prob_used": prob_col},
    ]
    metrics_df = pd.DataFrame(metrics)

    out_cv = Path(args.cv_output)
    out_metrics = Path(args.metrics_output)
    out_cv.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_cv, index=False)
    metrics_df.to_csv(out_metrics, index=False)

    print(f"\nselected threshold={best_t:.3f} mean_acc={best_score:.5f} (prob={prob_col})")
    print("\nmetrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nsaved cv: {out_cv}")
    print(f"saved metrics: {out_metrics}")

    if wandb_logger is not None:
        # Log summary metrics (business accuracy is based on your acc formula).
        mean_acc = float(metrics_df.loc[metrics_df["model"] == "TwoStage", "MeanAccuracy"].iloc[0])
        first_acc = float(metrics_df.loc[metrics_df["model"] == "TwoStage", "FirstMonthAccuracy"].iloc[0])
        wandb_logger.experiment.log(
            {
                "eval/selected_threshold": best_t,
                "eval/mean_accuracy": mean_acc,
                "eval/first_month_accuracy": first_acc,
                "eval/platt_a": a,
                "eval/platt_b": b,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchTST 两阶段 v2：NF-native分类loss + Platt校准 + log1p回归。")
    parser.add_argument("--data", default="./data/data_cleaned.csv")
    parser.add_argument("--cv-output", default="./forecast_results_patchtst_2stage_v2.csv")
    parser.add_argument("--metrics-output", default="./metrics_patchtst_2stage_v2.csv")

    parser.add_argument(
        "--disable-series-filter",
        action="store_true",
        help="关闭按 unique_id 的 series-level 过滤（默认开启：max_y<=series_max_y 且 zero_ratio<series_max_zero_ratio）。",
    )
    parser.add_argument(
        "--series-max-y",
        type=float,
        default=100.0,
        help="series-level 过滤：该 unique_id 的历史销量最大值必须 <= 该阈值。",
    )
    parser.add_argument(
        "--series-max-zero-ratio",
        type=float,
        default=0.4,
        help="series-level 过滤：该 unique_id 的历史销量 0 占比必须 < 该阈值。",
    )

    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=24)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)

    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--val-check-steps", type=int, default=100)
    parser.add_argument(
        "--early-stop-patience-steps",
        type=int,
        default=3,
        help="早停耐心值；-1 关闭早停。该值表示允许多少次验证不提升后停止。",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scaler-type", default="robust", choices=["identity", "standard", "robust"])

    parser.add_argument("--cls-loss", choices=["mse", "mae", "huber"], default="mse")
    parser.add_argument("--cls-huber-delta", type=float, default=1.0)
    parser.add_argument("--reg-loss", choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)

    parser.add_argument("--use-calibrated-prob", action="store_true", help="使用 Platt 校准后的概率选阈值/门控")
    parser.add_argument("--calib-max-iter", type=int, default=30)
    parser.add_argument("--calib-sample", type=int, default=200_000)

    # W&B (optional)
    parser.add_argument("--wandb-project", default="", help="启用 W&B：填写 project 名；留空则关闭。")
    parser.add_argument("--wandb-entity", default="", help="可选：W&B entity/团队。")
    parser.add_argument("--wandb-run-name", default="patchtst-2stage-v2", help="W&B run 名称。")
    parser.add_argument("--wandb-tags", default="patchtst,2stage", help="逗号分隔 tags。")
    parser.add_argument("--wandb-dir", default=".", help="W&B 本地缓存目录。")
    parser.add_argument("--log-every-n-steps", type=int, default=50, help="Lightning logger 记录频率。")

    parser.add_argument(
        "--threshold-grid",
        default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95",
    )

    main(parser.parse_args())
