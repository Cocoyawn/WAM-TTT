"""Regenerate every PNG under docs/ttt_loss_split/ from the existing CSVs, using
the scientific palette + Times-serif look defined in eval_loss_split.py.
"""
import csv
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.eval_loss_split import plot_run, plot_combined  # noqa: E402


OUT = os.path.join(_REPO, 'docs/ttt_loss_split')


def load_csv(path):
    with open(path) as f:
        rdr = csv.DictReader(f)
        rows = []
        for r in rdr:
            rows.append({
                "step":        int(r["step"]),
                "loss_total":  float(r["loss_total"]),
                "loss_action": float(r["loss_action"]),
                "loss_dct":    float(r["loss_dct"]),
                "loss_img":    float(r["loss_img"]),
            })
        rows.sort(key=lambda r: r['step'])
        return rows


# Model names: per-run title in the combined figure (dataset goes in suptitle).
DISPLAY = {
    "long":              "1/4 TTT-256 (fresh cycle)",
    "early":             "1/4 TTT-256 (0-60k)",
    "cont60k":           "1/4 TTT-256 (resume 60k-80k)",
    "mainline_quarter":  "1/4 TTT-256",
    "allttt":            "all TTT-256",
    "1to1":              "1:1 TTT-256",
}
DATASET_TITLE = "MolmoAct2-DROID"
# Per-run standalone PNG title = model name + dataset.
STANDALONE_TITLE = {k: f"{v}  ·  {DATASET_TITLE}" for k, v in DISPLAY.items()}
# Which files to include in the summary combined figure.
COMBINED_ORDER = ["mainline_quarter", "1to1", "allttt", "long"]


def main():
    all_rows = {}
    for fn in os.listdir(OUT):
        if not fn.endswith('.csv'):
            continue
        name = fn[:-4]
        csv_path = os.path.join(OUT, fn)
        rows = load_csv(csv_path)
        if not rows:
            continue
        png_path = os.path.join(OUT, name + '.png')
        plot_run(rows, STANDALONE_TITLE.get(name, name), png_path)
        print(f"[png] {png_path}  ({len(rows)} points)")
        all_rows[name] = rows

    ordered = {name: all_rows[name] for name in COMBINED_ORDER if name in all_rows}
    if len(ordered) >= 2:
        png = os.path.join(OUT, 'combined.png')
        # Per-subplot title = short model name; dataset goes in suptitle.
        pretty = {DISPLAY.get(k, k): v for k, v in ordered.items()}
        plot_combined(pretty, png, suptitle=DATASET_TITLE)
        print(f"[png] {png}  ({len(ordered)} runs)")


if __name__ == "__main__":
    main()
