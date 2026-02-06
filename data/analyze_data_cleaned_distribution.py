import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def quantile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = int((len(sorted_vals) - 1) * p)
    return sorted_vals[idx]


def analyze(input_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = 0
    uids = set()
    min_ds = None
    max_ds = None
    zero_count = 0
    y_vals = []
    nonzero_vals = []
    len_by = defaultdict(int)
    zero_by = defaultdict(int)

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            uid = row["unique_id"]
            y = float(row["y"])
            ds = datetime.strptime(row["ds"], "%Y-%m-%d")

            uids.add(uid)
            len_by[uid] += 1
            y_vals.append(y)
            if y == 0:
                zero_count += 1
                zero_by[uid] += 1
            else:
                nonzero_vals.append(y)

            if min_ds is None or ds < min_ds:
                min_ds = ds
            if max_ds is None or ds > max_ds:
                max_ds = ds

    y_vals.sort()
    nonzero_vals.sort()
    uid_zero_ratio = sorted(zero_by[uid] / len_by[uid] for uid in len_by)

    y_bins = [
        ("0", lambda x: x == 0),
        ("(0,1]", lambda x: 0 < x <= 1),
        ("(1,5]", lambda x: 1 < x <= 5),
        ("(5,20]", lambda x: 5 < x <= 20),
        ("(20,100]", lambda x: 20 < x <= 100),
        ("(100,1000]", lambda x: 100 < x <= 1000),
        (">1000", lambda x: x > 1000),
    ]
    y_bin_counts = [0] * len(y_bins)
    for y in y_vals:
        for i, (_, fn) in enumerate(y_bins):
            if fn(y):
                y_bin_counts[i] += 1
                break

    zr_bins = [
        ("0-10%", 0.0, 0.1),
        ("10-30%", 0.1, 0.3),
        ("30-50%", 0.3, 0.5),
        ("50-70%", 0.5, 0.7),
        ("70-90%", 0.7, 0.9),
        ("90-100%", 0.9, 1.0000001),
    ]
    zr_counts = {name: 0 for name, _, _ in zr_bins}
    for z in uid_zero_ratio:
        for name, l, r in zr_bins:
            if l <= z < r:
                zr_counts[name] += 1
                break

    summary = {
        "rows": rows,
        "unique_ids": len(uids),
        "date_start": min_ds.strftime("%Y-%m-%d") if min_ds else None,
        "date_end": max_ds.strftime("%Y-%m-%d") if max_ds else None,
        "zero_ratio_overall": zero_count / rows if rows else None,
        "y_quantiles": {str(p): quantile(y_vals, p) for p in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
        "nonzero_y_quantiles": {str(p): quantile(nonzero_vals, p) for p in [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]},
        "uid_zero_ratio_quantiles": {
            str(p): quantile(uid_zero_ratio, p) for p in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        },
    }

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out / "y_bin_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "count"])
        for (name, _), cnt in zip(y_bins, y_bin_counts):
            w.writerow([name, cnt])

    with open(out / "uid_zero_ratio_bin_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "count"])
        for name, _, _ in zr_bins:
            w.writerow([name, zr_counts[name]])

    # Optional plotting (if matplotlib available in runtime env).
    try:
        import matplotlib.pyplot as plt

        # Overall y distribution: log1p scale histogram (clip top 99.5% for readability).
        cap = quantile(y_vals, 0.995)
        y_for_plot = [min(v, cap) for v in y_vals]
        plt.figure(figsize=(9, 5))
        plt.hist([v for v in y_for_plot if v > 0], bins=80)
        plt.yscale("log")
        plt.title("Non-zero y Distribution (log y-axis, clipped at 99.5%)")
        plt.xlabel("y")
        plt.ylabel("count (log)")
        plt.tight_layout()
        plt.savefig(out / "overall_y_distribution.png", dpi=150)
        plt.close()

        # uid zero-ratio distribution.
        plt.figure(figsize=(9, 5))
        plt.hist(uid_zero_ratio, bins=40, range=(0, 1))
        plt.title("Zero Ratio Distribution by unique_id")
        plt.xlabel("zero_ratio_per_uid")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out / "uid_zero_ratio_distribution.png", dpi=150)
        plt.close()
    except Exception:
        pass

    print(f"saved analysis to: {out}")
    print(f"rows={rows}, unique_ids={len(uids)}, zero_ratio={summary['zero_ratio_overall']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze & visualize data_cleaned.csv distribution.")
    parser.add_argument("--input", default="./data/data_cleaned.csv")
    parser.add_argument("--out-dir", default="./analysis/data_cleaned_distribution")
    args = parser.parse_args()
    analyze(args.input, args.out_dir)
