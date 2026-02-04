import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FLOW_KEEP = {"中流件", "快流件"}


def _to_num(series):
    return pd.to_numeric(series, errors="coerce")


def _to_holiday(series):
    s = series.astype(str).str.strip().str.lower()
    return s.isin({"1", "true", "t", "yes", "y"}).astype("int8")


def build_panel(input_path: str, output_path: str):
    use_cols = [
        "dt",
        "werks",
        "matnr",
        "flow_rate",
        "kwmeng",
        "is_holiday",
        "invntry_cnt",
        "supply_lt",
        "transportation_lt",
        "prchs_cycl",
    ]
    df = pd.read_csv(input_path, usecols=use_cols, low_memory=False)
    df["dt"] = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["dt"].notna()].copy()

    df["flow_rate"] = df["flow_rate"].astype(str).str.strip()
    df = df[df["flow_rate"].isin(FLOW_KEEP)].copy()

    df["werks"] = df["werks"].astype(str).str.strip()
    df["matnr"] = df["matnr"].astype(str).str.strip()
    df = df[(df["werks"] != "") & (df["matnr"] != "")].copy()
    df["unique_id"] = df["werks"] + "_" + df["matnr"]

    df["y"] = _to_num(df["kwmeng"]).fillna(0.0).clip(lower=0.0)
    df["is_holiday_num"] = _to_holiday(df["is_holiday"])
    for c in ["invntry_cnt", "supply_lt", "transportation_lt", "prchs_cycl"]:
        df[c] = _to_num(df[c])

    # pandas>=2.0 不支持在 Period.to_timestamp 里传 "MS"，这里固定取月初时间戳。
    df["ds"] = df["dt"].dt.to_period("M").dt.to_timestamp(how="start")
    df = df.sort_values(["unique_id", "dt"])

    monthly = (
        df.groupby(["unique_id", "ds"], as_index=False)
        .agg(
            y=("y", "sum"),
            tx_count=("y", "size"),
            holiday_days_in_month=("is_holiday_num", "sum"),
            invntry_cnt=("invntry_cnt", "last"),
            supply_lt=("supply_lt", "last"),
            transportation_lt=("transportation_lt", "last"),
            prchs_cycl=("prchs_cycl", "last"),
            flow_rate=("flow_rate", "last"),
        )
        .sort_values(["unique_id", "ds"])
    )

    panel_parts = []
    for uid, g in monthly.groupby("unique_id", sort=False):
        full_ds = pd.date_range(g["ds"].min(), g["ds"].max(), freq="MS")
        g2 = g.set_index("ds").reindex(full_ds)
        g2["unique_id"] = uid
        g2["ds"] = full_ds
        g2["is_padded_month"] = g2["tx_count"].isna().astype("int8")
        panel_parts.append(g2.reset_index(drop=True))

    panel = pd.concat(panel_parts, ignore_index=True)
    panel = panel.sort_values(["unique_id", "ds"])

    panel["y"] = panel["y"].fillna(0.0)
    panel["tx_count"] = panel["tx_count"].fillna(0).astype(int)
    panel["holiday_days_in_month"] = panel["holiday_days_in_month"].fillna(0).astype(float)
    panel["flow_rate"] = panel.groupby("unique_id")["flow_rate"].ffill().bfill()
    panel["flow_fast"] = (panel["flow_rate"] == "快流件").astype("int8")

    # Padding 后的关键处理：库存/LT 等特征先短窗口前向填充，再保留缺失指示。
    fill_cols = ["invntry_cnt", "supply_lt", "transportation_lt", "prchs_cycl"]
    for c in fill_cols:
        panel[f"{c}_missing"] = panel[c].isna().astype("int8")
        panel[c] = panel.groupby("unique_id")[c].ffill(limit=3)
        panel[c] = panel[c].fillna(panel.groupby("unique_id")[c].transform("median"))
        panel[c] = panel[c].fillna(panel[c].median())

    month = panel["ds"].dt.month
    panel["month_sin"] = np.sin(2 * np.pi * month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * month / 12)

    g = panel.groupby("unique_id", group_keys=False)["y"]
    for lag in [1, 2, 3, 6, 12]:
        panel[f"lag_{lag}"] = g.shift(lag)
    panel["rolling_mean_3"] = g.apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    panel["rolling_mean_6"] = g.apply(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    panel["rolling_nonzero_6"] = g.apply(
        lambda s: s.shift(1).rolling(6, min_periods=1).apply(lambda v: float(np.count_nonzero(v > 0)), raw=True)
    )

    lag_cols = [f"lag_{x}" for x in [1, 2, 3, 6, 12]] + ["rolling_mean_3", "rolling_mean_6", "rolling_nonzero_6"]
    panel[lag_cols] = panel[lag_cols].fillna(0.0)

    keep_cols = [
        "unique_id",
        "ds",
        "y",
        "flow_rate",
        "flow_fast",
        "is_padded_month",
        "holiday_days_in_month",
        "invntry_cnt",
        "supply_lt",
        "transportation_lt",
        "prchs_cycl",
        "invntry_cnt_missing",
        "supply_lt_missing",
        "transportation_lt_missing",
        "prchs_cycl_missing",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_nonzero_6",
    ]
    panel = panel[keep_cols].sort_values(["unique_id", "ds"])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)

    print(f"saved: {out}")
    print(f"rows={len(panel)} uids={panel['unique_id'].nunique()} date={panel['ds'].min().date()}->{panel['ds'].max().date()}")
    print(f"zero_ratio={(panel['y'] == 0).mean():.4f} padded_ratio={panel['is_padded_month'].mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建中快流件月度特征面板（含 padding 与防泄漏特征）。")
    parser.add_argument("--input", default="./data/data_all_until_202601_fix.csv")
    parser.add_argument("--output", default="./data/monthly_feature_panel.csv")
    args = parser.parse_args()
    build_panel(args.input, args.output)
