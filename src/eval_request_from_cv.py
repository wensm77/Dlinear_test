import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_yyyymm(x) -> pd.Timestamp:
    """Parse M月份 like 202510 -> Timestamp(2025-10-01)."""
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if len(s) != 6:
        return pd.NaT
    y = int(s[:4])
    m = int(s[4:6])
    if m < 1 or m > 12:
        return pd.NaT
    return pd.Timestamp(year=y, month=m, day=1)


def acc(f, a):
    # Business accuracy (strictly per your definition)
    if f == 0 and a == 0:
        return 1.0
    if f != 0 and a == 0:
        return 0.0
    return max(0.0, 1.0 - abs(f - a) / a)


def main(args):
    req = pd.read_csv(args.request_csv, encoding=args.request_encoding, dtype=str)
    for c in [args.col_werks, args.col_matnr, args.col_month]:
        if c not in req.columns:
            raise ValueError(f"request_csv 缺少列: {c}. 实际列: {list(req.columns)}")

    req[args.col_werks] = req[args.col_werks].astype(str).str.strip()
    req[args.col_matnr] = req[args.col_matnr].astype(str).str.strip()
    req["unique_id"] = req[args.col_werks] + "_" + req[args.col_matnr]
    req["target_ds"] = req[args.col_month].apply(parse_yyyymm)
    req["cutoff"] = req["target_ds"] - pd.offsets.MonthBegin(1)

    # Drop malformed rows early
    req = req[req["unique_id"].notna() & req["target_ds"].notna()].copy()

    cv = pd.read_csv(args.cv_preds)
    for c in ["unique_id", "ds", "cutoff"]:
        if c not in cv.columns:
            raise ValueError(f"cv_preds 缺少列: {c}. 实际列: {list(cv.columns)}")
    if args.pred_col not in cv.columns:
        raise ValueError(f"cv_preds 缺少预测列 pred-col={args.pred_col}. 实际列: {list(cv.columns)}")

    cv["ds"] = pd.to_datetime(cv["ds"])
    cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv = cv[["unique_id", "ds", "cutoff", args.pred_col]].rename(columns={args.pred_col: "pred"})

    merged = req.merge(
        cv,
        left_on=["unique_id", "target_ds", "cutoff"],
        right_on=["unique_id", "ds", "cutoff"],
        how="left",
    )

    merged["matched"] = merged["pred"].notna()

    if args.col_actual and args.col_actual in merged.columns:
        actual = pd.to_numeric(merged[args.col_actual], errors="coerce")
        pred = pd.to_numeric(merged["pred"], errors="coerce").fillna(np.nan)
        merged["our_acc"] = [
            acc(float(f) if f == f else np.nan, float(a) if a == a else np.nan)
            if (f == f and a == a)
            else np.nan
            for f, a in zip(pred.to_numpy(), actual.to_numpy())
        ]

    if args.drop_unmatched:
        merged = merged[merged["matched"]].copy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"saved: {out}")
    print(f"rows={len(merged)} matched={int(merged['matched'].sum())}/{len(merged)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="将业务侧 request CSV（工厂编码_物料编码 + M月份）映射到滚动回测 CV 预测结果中。"
    )
    p.add_argument("--request-csv", required=True, help="例如 data/南非_11月.csv")
    p.add_argument("--cv-preds", required=True, help="例如 forecast_results_patchtst_2stage_v2.csv")
    p.add_argument("--pred-col", default="yhat_two_stage", help="CV 文件中的预测列名")
    p.add_argument("--output", default="./analysis/request_eval.csv")

    p.add_argument("--request-encoding", default="utf-8", help="utf-8 / utf-8-sig / gbk 等")
    p.add_argument("--col-werks", default="工厂编码")
    p.add_argument("--col-matnr", default="物料编码")
    p.add_argument("--col-month", default="M月份")
    p.add_argument("--col-actual", default="M月实际值", help="可选：用于额外算 our_acc；不存在则跳过")
    p.add_argument("--drop-unmatched", action="store_true", help="丢弃 unique_id/月 未匹配到 CV 的行")

    main(p.parse_args())

