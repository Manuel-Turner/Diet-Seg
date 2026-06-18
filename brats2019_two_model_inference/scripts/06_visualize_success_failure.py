from __future__ import annotations

from pathlib import Path

from common import FIGURES_DIR, METRICS_DIR, NORM_DIR, REGIONS, ensure_project_dirs, load_selected_cases, read_csv, read_json


COLORS = {
    1: (1.0, 1.0, 0.0, 0.45),  # yellow
    2: (0.0, 1.0, 0.0, 0.45),  # green
    4: (1.0, 0.0, 0.0, 0.50),  # red
}


def load_nii(path):
    import nibabel as nib

    return nib.load(str(path)).get_fdata()


def choose_slice(gt, flair):
    import numpy as np

    wt = np.isin(gt, REGIONS["WT"])
    areas = wt.sum(axis=(0, 1))
    if areas.max() > 0:
        return int(areas.argmax())
    variances = flair.var(axis=(0, 1))
    return int(variances.argmax())


def overlay_label(ax, base, label):
    import numpy as np

    ax.imshow(base.T, cmap="gray", origin="lower")
    rgba = np.zeros((*label.T.shape, 4), dtype=float)
    for value, color in COLORS.items():
        rgba[label.T == value] = color
    ax.imshow(rgba, origin="lower")
    ax.axis("off")


def dice_lookup(rows, case_id, model):
    for row in rows:
        if row["case_id"] == case_id and row["model"] == model:
            return row
    return None


def title_for(case_id, name, row=None):
    if row is None:
        return f"{case_id}\n{name}"
    return (
        f"{case_id}\n{name} "
        f"WT {float(row['dice_WT']):.3f} TC {float(row['dice_TC']):.3f} ET {float(row['dice_ET']):.3f}"
    )


def make_case_figure(case, out_path: Path, metric_rows):
    import matplotlib.pyplot as plt

    case_id = case["case_id"]
    flair = load_nii(case["flair"])
    t1ce = load_nii(case["t1ce"])
    gt = load_nii(case["seg"]).astype("uint8")
    monai_path = NORM_DIR / "monai" / f"{case_id}_seg.nii.gz"
    kaist_path = NORM_DIR / "kaist" / f"{case_id}_seg.nii.gz"
    if not monai_path.exists() or not kaist_path.exists():
        return False
    monai = load_nii(monai_path).astype("uint8")
    kaist = load_nii(kaist_path).astype("uint8")
    z = choose_slice(gt, flair)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    panels = [
        (flair[:, :, z], None, "FLAIR", None),
        (flair[:, :, z], gt[:, :, z], "Ground truth", None),
        (flair[:, :, z], monai[:, :, z], "MONAI", dice_lookup(metric_rows, case_id, "monai")),
        (flair[:, :, z], kaist[:, :, z], "KAIST", dice_lookup(metric_rows, case_id, "kaist")),
        (t1ce[:, :, z], None, "T1ce", None),
        (t1ce[:, :, z], gt[:, :, z], "Ground truth", None),
        (t1ce[:, :, z], monai[:, :, z], "MONAI", dice_lookup(metric_rows, case_id, "monai")),
        (t1ce[:, :, z], kaist[:, :, z], "KAIST", dice_lookup(metric_rows, case_id, "kaist")),
    ]
    for ax, (base, label, name, row) in zip(axes.flat, panels):
        if label is None:
            ax.imshow(base.T, cmap="gray", origin="lower")
            ax.axis("off")
        else:
            overlay_label(ax, base, label)
        ax.set_title(title_for(case_id, name, row), fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def main() -> int:
    ensure_project_dirs()
    cases = {case["case_id"]: case for case in load_selected_cases()}
    metric_rows = read_csv(METRICS_DIR / "dice_by_case.csv")
    selection = read_json(METRICS_DIR / "success_failure_cases.json")
    made = 0

    success_id = selection.get("success_case")
    failure_id = selection.get("failure_case")
    if success_id and success_id in cases:
        made += int(make_case_figure(cases[success_id], FIGURES_DIR / f"success_case_{success_id}.png", metric_rows))
    if failure_id and failure_id in cases:
        made += int(make_case_figure(cases[failure_id], FIGURES_DIR / f"failure_case_{failure_id}.png", metric_rows))

    all_dir = FIGURES_DIR / "all_cases"
    for case_id, case in cases.items():
        made += int(make_case_figure(case, all_dir / f"{case_id}.png", metric_rows))

    print(f"Figures generated: {made}")
    return 0 if made else 2


if __name__ == "__main__":
    raise SystemExit(main())
