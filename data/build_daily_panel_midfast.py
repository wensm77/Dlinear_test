import argparse
from pathlib import Path

import pandas as pd


def build_daily_panel(input_path: str, output_path: str, min_days: int):
    cols = ["dt", "werks", "matnr", "flow_rate", "kwmeng"]
    df = pd.read_csv(input_path, usecols=cols, low_memory=False)
    df["dt"] = pd.to_datetime(df["dt"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["dt"].notna()].copy()

    df["werks"] = df["werks"].astype(str).str.strip()
    df["matnr"] = df["matnr"].astype(str).str.strip()
    df = df[(df["werks"] != "") & (df["matnr"] != "")].copy()
    df["unique_id"] = df["werks"] + "_" + df["matnr"]

    df["flow_rate"] = df["flow_rate"].astype(str).str.strip()
    df["y"] = pd.to_numeric(df["kwmeng"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df = df.sort_values(["unique_id", "dt"])

    # 每个ID取最后一次出现时的 flow_rate 作为最终 flow_rate
    last_flow = (
        df[df["flow_rate"] != ""]
        .drop_duplicates(subset=["unique_id", "dt"], keep="last")
        .groupby("unique_id", as_index=False)
        .tail(1)[["unique_id", "flow_rate"]]
        .rename(columns={"flow_rate": "flow_rate_final"})
    )
    keep_ids = set(last_flow[last_flow["flow_rate_final"].isin(["中流件", "快流件"])]["unique_id"])
    if not keep_ids:
        raise ValueError("未找到最终 flow_rate 属于中/快流件的 ID。")

    df = df[df["unique_id"].isin(keep_ids)].copy()
    daily = df.groupby(["unique_id", "dt"], as_index=False)["y"].sum().sort_values(["unique_id", "dt"])
    daily = daily.merge(last_flow, on="unique_id", how="left")

    parts = []
    for uid, g in daily.groupby("unique_id", sort=False):
        full_ds = pd.date_range(g["dt"].min(), g["dt"].max(), freq="D")
        g2 = g.set_index("dt").reindex(full_ds)
        g2["unique_id"] = uid
        g2["ds"] = full_ds
        g2["y"] = g2["y"].fillna(0.0)
        g2["flow_rate_final"] = g2["flow_rate_final"].ffill().bfill()
        parts.append(g2[["unique_id", "ds", "y", "flow_rate_final"]].reset_index(drop=True))

    panel = pd.concat(parts, ignore_index=True).sort_values(["unique_id", "ds"])

    if min_days > 0:
        counts = panel["unique_id"].value_counts()
        valid_ids = counts[counts >= min_days].index
        panel = panel[panel["unique_id"].isin(valid_ids)].copy()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)

    print(f"saved: {out}")
    print(
        f"rows={len(panel)} uids={panel['unique_id'].nunique()} "
        f"date={panel['ds'].min().date()}->{panel['ds'].max().date()} "
        f"zero_ratio={(panel['y']==0).mean():.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建中快流件日度销量面板（按最后 flow_rate 筛选，自动补齐日历）。")
    parser.add_argument("--input", default="./data/data_all_until_202601_fix.csv")
    parser.add_argument("--output", default="./data/daily_midfast_panel.csv")
    parser.add_argument("--min-days", type=int, default=750, help="每个SKU最少天数（默认750，满足720输入+30输出）")
    args = parser.parse_args()
    build_daily_panel(args.input, args.output, min_days=args.min_days)
