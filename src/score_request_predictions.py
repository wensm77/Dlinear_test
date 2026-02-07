import argparse

import numpy as np
import pandas as pd


def acc_vectorized(f: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Business accuracy (strictly following the user's definition)."""
    out = np.zeros_like(f, dtype=float)
    both_zero = (f == 0) & (a == 0)
    bad = (f != 0) & (a == 0)
    normal = a != 0
    out[both_zero] = 1.0
    out[bad] = 0.0
    out[normal] = np.maximum(0.0, 1.0 - np.abs(f[normal] - a[normal]) / np.abs(a[normal]))
    return out


def _to_float(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _zeroify(x: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        return x
    x = x.copy()
    x[np.abs(x) <= eps] = 0.0
    return x


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
    f1 = (2 * precision * recall / (precision + recall)) if (precision == precision and recall == recall and (precision + recall)) else np.nan
    return {
        "cls_accuracy": float(acc) if acc == acc else np.nan,
        "precision": float(precision) if precision == precision else np.nan,
        "recall": float(recall) if recall == recall else np.nan,
        "f1_score": float(f1) if f1 == f1 else np.nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main(args):
    df = pd.read_csv(args.input, encoding=args.encoding)
    for c in [args.pred_col, args.actual_col]:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}. 实际列: {list(df.columns)}")

    pred_raw = _zeroify(_to_float(df[args.pred_col]), args.zero_eps)
    actual_raw = _zeroify(_to_float(df[args.actual_col]), args.zero_eps)

    actual_valid = np.isfinite(actual_raw)
    pred_missing_mask = ~np.isfinite(pred_raw)
    if args.missing_pred_policy == "drop":
        valid = np.isfinite(pred_raw) & actual_valid
        pred = pred_raw[valid]
        actual = actual_raw[valid]
        df = df.loc[valid].copy()
        pred_was_missing = pred_missing_mask[valid]
    else:
        # strict mode: keep rows with valid actual; missing pred is treated as 0
        valid = actual_valid
        df = df.loc[valid].copy()
        pred = pred_raw[valid]
        actual = actual_raw[valid]
        pred_was_missing = pred_missing_mask[valid]
        pred[~np.isfinite(pred)] = 0.0

    accs = acc_vectorized(pred, actual)
    df["_acc"] = accs
    df["_actual_zero"] = (actual == 0).astype(int)
    df["_pred_zero"] = (pred == 0).astype(int)
    df["_actual_num"] = actual
    df["_pred_num"] = pred
    df["_pred_was_missing"] = pred_was_missing.astype(int)

    total_rows = len(actual_raw)
    rows_actual_valid = int(actual_valid.sum())
    rows_missing_pred_before = int((~np.isfinite(pred_raw) & actual_valid).sum())
    rows_used = len(df)

    overall = {
        "total_rows": int(total_rows),
        "rows_actual_valid": rows_actual_valid,
        "rows_missing_pred_before_fill": rows_missing_pred_before,
        "rows_used": int(rows_used),
        "rows": int(len(df)),
        "business_accuracy": float(accs.mean()) if len(accs) else np.nan,
        "business_accuracy_nonzero": float(accs[actual > 0].mean()) if (actual > 0).any() else np.nan,
        "count_actual_zero": int((actual == 0).sum()),
        "count_pred_zero": int((pred == 0).sum()),
    }
    overall.update(classification_stats(pred, actual))

    print("overall:")
    for k in [
        "rows",
        "business_accuracy",
        "business_accuracy_nonzero",
        "count_actual_zero",
        "count_pred_zero",
        "cls_accuracy",
        "f1_score",
        "tp",
        "fp",
        "tn",
        "fn",
    ]:
        print(f"- {k}: {overall.get(k)}")

    if args.month_col and args.month_col in df.columns:
        actual_s = df["_actual_num"]
        g = (
            df.groupby(args.month_col, dropna=False)
            .agg(
                rows=("_acc", "size"),
                business_accuracy=("_acc", "mean"),
                business_accuracy_nonzero=("_acc", lambda s: float(s[actual_s.loc[s.index] > 0].mean()) if (actual_s.loc[s.index] > 0).any() else np.nan),
                count_actual_zero=("_actual_zero", "sum"),
                count_pred_zero=("_pred_zero", "sum"),
                count_missing_pred_before_fill=("_pred_was_missing", "sum"),
            )
            .reset_index()
        )
        out_path = args.output
        if out_path:
            g.to_csv(out_path, index=False, encoding="utf-8")
            print(f"\nsaved monthly metrics: {out_path}")
        else:
            print("\nby month:")
            print(g.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="对 request 预测结果计算业务准确率（按你的 acc 公式）+ 分类指标。")
    p.add_argument("--input", required=True, help="predict_two_stage_v2_from_ckpt.py 的输出 CSV")
    p.add_argument("--pred-col", default="pred", help="预测列名")
    p.add_argument("--actual-col", default="M月实际值", help="真实值列名")
    p.add_argument("--month-col", default="M月份", help="月份列名（可选，用于按月汇总）")
    p.add_argument("--encoding", default="utf-8", help="utf-8 / utf-8-sig / gbk 等")
    p.add_argument("--zero-eps", type=float, default=1e-9, help="将 |x|<=eps 的值视为 0，避免浮点误差")
    p.add_argument(
        "--missing-pred-policy",
        choices=["zero", "drop"],
        default="zero",
        help="预测缺失（pred=NaN）如何处理：zero=按0计入评测（默认，严格）；drop=丢弃该行。",
    )
    p.add_argument("--output", default="", help="可选：保存按月汇总 CSV 路径；留空则直接打印")
    main(p.parse_args())
