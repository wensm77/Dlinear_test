import argparse
import ast
import re
from pathlib import Path

import pandas as pd


def _normalize_uid(werks, material_code):
    if pd.isna(werks) or pd.isna(material_code):
        return None
    werks_str = str(werks).strip()
    if werks_str.endswith(".0"):
        werks_str = werks_str[:-2]
    material_str = str(material_code).strip()
    if not werks_str or not material_str:
        return None
    return f"{werks_str}_{material_str}"


def _parse_history_demand(raw_text):
    if pd.isna(raw_text):
        return None
    clean_str = re.sub(r"np\.float64\((.*?)\)", r"\1", str(raw_text))
    if not clean_str or clean_str.lower() == "nan":
        return None
    try:
        parsed = ast.literal_eval(clean_str)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def clean_and_save(input_path, output_path):
    print(f"读取数据: {input_path}")
    try:
        raw_df = pd.read_csv(input_path, on_bad_lines="skip")
    except TypeError:
        raw_df = pd.read_csv(input_path, error_bad_lines=False)

    records = []
    bad_uid_rows = 0
    bad_history_rows = 0
    total_rows = len(raw_df)

    print("开始清洗与转换...")
    for idx, row in raw_df.iterrows():
        if idx % 10000 == 0:
            print(f"处理进度: {idx}/{total_rows}")

        uid = _normalize_uid(row.get("werks"), row.get("material_code"))
        if uid is None:
            bad_uid_rows += 1
            continue

        demand_dict = _parse_history_demand(row.get("history_demand"))
        if demand_dict is None:
            bad_history_rows += 1
            continue

        for date_str, val in demand_dict.items():
            dt = pd.to_datetime(str(date_str), format="%Y%m", errors="coerce")
            if pd.isna(dt):
                continue
            try:
                y = float(val)
            except Exception:
                continue
            records.append({"unique_id": uid, "ds": dt, "y": y})

    if not records:
        raise ValueError("没有解析出任何有效数据，请检查输入文件格式。")

    long_df = pd.DataFrame(records).dropna().sort_values(["unique_id", "ds"])
    long_df = long_df.groupby(["unique_id", "ds"], as_index=False)["y"].sum()

    print("\n开始补齐时间轴（仅在每个序列自身首末区间内补 0）...")
    padded = []
    for i, (uid, grp) in enumerate(long_df.groupby("unique_id", sort=False)):
        if i % 5000 == 0:
            print(f"补齐进度: {i}/{long_df['unique_id'].nunique()}")
        full_idx = pd.date_range(grp["ds"].min(), grp["ds"].max(), freq="MS")
        s = grp.set_index("ds")["y"].reindex(full_idx, fill_value=0.0)
        out = s.rename("y").reset_index().rename(columns={"index": "ds"})
        out["unique_id"] = uid
        padded.append(out[["unique_id", "ds", "y"]])

    cleaned_df = pd.concat(padded, ignore_index=True).sort_values(["unique_id", "ds"])

    print("\n清洗统计:")
    print(f"- 原始行数: {total_rows}")
    print(f"- 跳过无效 werks/material 行: {bad_uid_rows}")
    print(f"- 跳过无效 history_demand 行: {bad_history_rows}")
    print(f"- 清洗后记录数: {len(cleaned_df)}")
    print(f"- 序列数: {cleaned_df['unique_id'].nunique()}")
    print(f"- 时间范围: {cleaned_df['ds'].min().date()} -> {cleaned_df['ds'].max().date()}")
    print(f"- 0 占比: {(cleaned_df['y'] == 0).mean():.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗销量数据并生成 NeuralForecast 格式。")
    parser.add_argument("--input", default="./data/data_all.csv", help="输入原始 CSV 路径")
    parser.add_argument("--output", default="./data/data_cleaned.csv", help="输出清洗后 CSV 路径")
    args = parser.parse_args()
    clean_and_save(args.input, args.output)
