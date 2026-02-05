import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast


def parse_yyyymm(x) -> pd.Timestamp:
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def acc(f, a):
    if f == 0 and a == 0:
        return 1.0
    if f != 0 and a == 0:
        return 0.0
    return max(0.0, 1.0 - abs(f - a) / a)


def main(args):
    ckpt = Path(args.ckpt_dir)
    params_path = ckpt / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"未找到 params.json: {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))

    nf_cls = NeuralForecast.load(str(ckpt / "cls"))
    nf_reg = NeuralForecast.load(str(ckpt / "reg"))

    req = pd.read_csv(args.request_csv, encoding=args.request_encoding, dtype=str)
    for c in [args.col_werks, args.col_matnr, args.col_month]:
        if c not in req.columns:
            raise ValueError(f"request_csv 缺少列: {c}. 实际列: {list(req.columns)}")
    req[args.col_werks] = req[args.col_werks].astype(str).str.strip()
    req[args.col_matnr] = req[args.col_matnr].astype(str).str.strip()
    req["unique_id"] = req[args.col_werks] + "_" + req[args.col_matnr]
    req["target_ds"] = req[args.col_month].apply(parse_yyyymm)
    req["cutoff"] = req["target_ds"] - pd.offsets.MonthBegin(1)
    req = req[req["unique_id"].notna() & req["target_ds"].notna()].copy()

    hist = pd.read_csv(args.history_data)
    hist["ds"] = pd.to_datetime(hist["ds"])
    hist["y"] = pd.to_numeric(hist["y"], errors="coerce").fillna(0.0).clip(lower=0.0)
    hist = hist[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    # Only keep ids that appear in history
    ids_in_hist = set(hist["unique_id"].unique())
    req["in_history"] = req["unique_id"].isin(ids_in_hist)
    if args.drop_unmatched:
        req = req[req["in_history"]].copy()

    # Prediction per cutoff group: uses data <= cutoff, predicts ds=cutoff+1 month (h=1).
    out_rows = []
    for cutoff, g in req.groupby("cutoff", sort=True):
        ids = g["unique_id"].unique().tolist()
        h = hist[(hist["unique_id"].isin(ids)) & (hist["ds"] <= cutoff)].copy()
        if h.empty:
            # keep rows with NaNs
            tmp = g.copy()
            tmp["pred"] = np.nan
            tmp["pred_nonzero_prob"] = np.nan
            tmp["pred_reg"] = np.nan
            out_rows.append(tmp)
            continue

        # Require at least input_size points to build a prediction window.
        input_size = int(params.get("input_size", args.input_size_fallback))
        c = h["unique_id"].value_counts()
        ok = c[c >= input_size].index
        h = h[h["unique_id"].isin(ok)].copy()
        if h.empty:
            tmp = g.copy()
            tmp["pred"] = np.nan
            tmp["pred_nonzero_prob"] = np.nan
            tmp["pred_reg"] = np.nan
            out_rows.append(tmp)
            continue

        cls_pred = nf_cls.predict(df=h).rename(columns={"cls": "score_nonzero"})
        reg_pred = nf_reg.predict(df=h).rename(columns={"reg": "yhat_log"})

        m = cls_pred.merge(reg_pred, on=["unique_id", "ds"], how="inner")
        m["cutoff"] = pd.Timestamp(cutoff)

        # Only keep ds that are requested in this cutoff group.
        targets = g[["unique_id", "target_ds"]].drop_duplicates()
        targets = targets.rename(columns={"target_ds": "ds"})
        m = targets.merge(m, on=["unique_id", "ds"], how="left")

        score = m["score_nonzero"].to_numpy(dtype=float)
        p_raw = np.clip(score, 0.0, 1.0)
        a = params.get("platt_a", None)
        b = params.get("platt_b", None)
        if params.get("use_calibrated_prob", False) and a is not None and b is not None:
            p = sigmoid(float(a) * score + float(b))
        else:
            p = p_raw

        yhat_reg = np.expm1(m["yhat_log"].to_numpy(dtype=float)).clip(min=0.0)
        t = params.get("selected_threshold", None)
        if t is None:
            t = float(args.threshold_fallback)
        pred = np.where(p >= float(t), yhat_reg, 0.0)

        m["pred_nonzero_prob"] = p
        m["pred_reg"] = yhat_reg
        m["pred"] = pred

        tmp = g.merge(m[["unique_id", "ds", "pred", "pred_nonzero_prob", "pred_reg"]], left_on=["unique_id", "target_ds"], right_on=["unique_id", "ds"], how="left")
        tmp = tmp.drop(columns=["ds"])
        out_rows.append(tmp)

    out = pd.concat(out_rows, ignore_index=True)

    # Optional: compute business accuracy if actual exists.
    if args.col_actual and args.col_actual in out.columns:
        actual = pd.to_numeric(out[args.col_actual], errors="coerce")
        pred = pd.to_numeric(out["pred"], errors="coerce")
        out["our_acc"] = [
            acc(float(f), float(a)) if (f == f and a == a) else np.nan
            for f, a in zip(pred.to_numpy(), actual.to_numpy())
        ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"saved: {out_path}")
    print(f"rows={len(out)} matched_history={int(out['in_history'].sum())}/{len(out)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="从 two-stage PatchTST v2 checkpoint 加载并对 request CSV 输出预测（按 cutoff 严格截断历史）。")
    p.add_argument("--ckpt-dir", required=True, help="fit_patchtst_two_stage_v2_ckpt.py 输出目录")
    p.add_argument("--history-data", default="./data/data_cleaned.csv", help="monthly long format: unique_id, ds, y")
    p.add_argument("--request-csv", required=True, help="例如 data/南非_11月.csv")
    p.add_argument("--output", default="./analysis/request_pred_from_ckpt.csv")

    p.add_argument("--request-encoding", default="utf-8")
    p.add_argument("--col-werks", default="工厂编码")
    p.add_argument("--col-matnr", default="物料编码")
    p.add_argument("--col-month", default="M月份")
    p.add_argument("--col-actual", default="M月实际值")
    p.add_argument("--drop-unmatched", action="store_true", help="丢弃没在历史数据中出现的 unique_id")

    # fallback only used if params.json missing fields
    p.add_argument("--input-size-fallback", type=int, default=24)
    p.add_argument("--threshold-fallback", type=float, default=0.5)

    main(p.parse_args())

