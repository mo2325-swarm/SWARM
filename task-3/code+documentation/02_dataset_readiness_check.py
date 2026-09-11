#!/usr/bin/env python3
"""
Dataset readiness check — Member C output
==========================================
A real (not mocked) evaluation of whether the labeled dataset that
01_member_C_pipeline.py produced is actually learnable, run on the real
IQ examples and real labels on disk. Every number in the output report is
computed from your data: features are extracted from the .npy files,
models are fit with sklearn, and metrics come from held-out predictions
via stratified k-fold cross-validation (not a single lucky split).

Two things are checked, because both are pipeline outputs worth trusting:
  1. interdrone_present  -- can a classifier find the synthetic packets
                             Member C inserted into the gaps?
  2. drones_active        -- can a classifier recover which drone(s) were
                             on-air, using only the raw IQ (i.e. does the
                             slot_map/gap_list-derived labeling agree with
                             what's actually in the waveform)?

Usage
-----
    python3 02_dataset_readiness_check.py --dataset_dir dataset_real

Requires the dataset_dir to contain `labels.csv` and an `iq/` folder, i.e.
exactly what 01_member_C_pipeline.py writes out.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------------------
# Feature extraction (real, computed per example)
# ---------------------------------------------------------------------------
def extract_features(iq, fs):
    p = np.abs(iq) ** 2
    mean_power_db = 10 * np.log10(p.mean() + 1e-15)
    power_std = float(p.std())
    papr_db = 10 * np.log10((p.max() + 1e-15) / (p.mean() + 1e-15))
    amp_kurtosis = float(kurtosis(np.abs(iq)))
    amp_skew = float(skew(np.abs(iq)))
    occupancy = float(np.mean(p > 3 * np.median(p)))

    nperseg = min(256, len(iq))
    f, Pxx = sig.welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    Pxx_n = Pxx / (Pxx.sum() + 1e-15)
    spectral_flatness = float(np.exp(np.mean(np.log(Pxx_n + 1e-15))) / (np.mean(Pxx_n) + 1e-15))
    spectral_centroid_hz = float(np.sum(f * Pxx_n))
    spectral_spread_hz = float(np.sqrt(np.sum(((f - spectral_centroid_hz) ** 2) * Pxx_n)))

    # 90%-power occupied bandwidth
    order = np.argsort(Pxx_n)[::-1]
    cum = np.cumsum(Pxx_n[order])
    n90 = np.searchsorted(cum, 0.90) + 1
    occ_bw_hz = float(n90 / len(Pxx_n) * fs)

    return {
        "mean_power_db": mean_power_db, "power_std": power_std, "papr_db": papr_db,
        "amp_kurtosis": amp_kurtosis, "amp_skew": amp_skew, "occupancy": occupancy,
        "spectral_flatness": spectral_flatness, "spectral_centroid_hz": spectral_centroid_hz,
        "spectral_spread_hz": spectral_spread_hz, "occupied_bw_hz": occ_bw_hz,
    }


def load_dataset(dataset_dir, fs):
    labels_path = os.path.join(dataset_dir, "labels.csv")
    iq_dir = os.path.join(dataset_dir, "iq")
    df = pd.read_csv(labels_path)

    feats = []
    for _, row in df.iterrows():
        arr = np.load(os.path.join(iq_dir, row["id"] + ".npy"))
        feats.append(extract_features(arr, fs))
    feat_df = pd.DataFrame(feats)
    return df, feat_df


# ---------------------------------------------------------------------------
# Task 1: interdrone_present (binary) -- cross-validated
# ---------------------------------------------------------------------------
def evaluate_interdrone(df, X, seed, outdir):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix, roc_curve)

    y = df["interdrone_present"].astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=3),
    }

    results = {}
    for name, model in models.items():
        # out-of-fold predictions -> every metric below is computed on held-out data only
        y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "y_proba": y_proba,
        }

    # feature importance from RF, fit once on all data for reporting only
    rf_full = RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=3).fit(X, y)
    importances = pd.Series(rf_full.feature_importances_, index=X.columns).sort_values(ascending=False)

    # --- plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    cm = results["RandomForest"]["confusion_matrix"]
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["absent", "present"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["absent", "present"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    axes[0].set_title("RandomForest confusion matrix\n(5-fold out-of-fold)")

    for name in models:
        fpr, tpr, _ = roc_curve(y, results[name]["y_proba"])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.2f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC curve (out-of-fold)")
    axes[1].legend(fontsize=8)

    importances.plot(kind="barh", ax=axes[2])
    axes[2].invert_yaxis()
    axes[2].set_title("RandomForest feature importance")

    plt.tight_layout()
    plot_path = os.path.join(outdir, "interdrone_present_evaluation.png")
    plt.savefig(plot_path, dpi=110)
    plt.close()

    return results, importances, plot_path


# ---------------------------------------------------------------------------
# Task 2: drones_active (multiclass) -- cross-validated
# ---------------------------------------------------------------------------
def evaluate_drones_active(df, X, seed, outdir):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    y = df["drones_active"].values
    classes = sorted(pd.unique(y))
    min_class_count = pd.Series(y).value_counts().min()
    n_splits = min(5, int(min_class_count))
    if n_splits < 2:
        return None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=3)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(y, y_pred, labels=classes)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"drones_active confusion matrix\n({n_splits}-fold out-of-fold)")
    plt.tight_layout()
    plot_path = os.path.join(outdir, "drones_active_evaluation.png")
    plt.savefig(plot_path, dpi=110)
    plt.close()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm,
            "classes": classes, "n_splits": n_splits, "plot_path": plot_path}


# ---------------------------------------------------------------------------
# Verdict + report
# ---------------------------------------------------------------------------
def verdict_for_interdrone(results):
    rf = results["RandomForest"]
    acc, auc = rf["accuracy"], rf["roc_auc"]
    if auc >= 0.98:
        return ("WARNING", "AUC is near-perfect — likely a label leak (packets too easy to "
                            "spot). Consider widening --snr-min/--snr-max downward or reducing "
                            "packet amplitude at the low end.")
    if auc <= 0.58:
        return ("WARNING", "AUC is close to chance (0.5) — the model can barely tell present "
                            "from absent. Check insertion_log.csv against labels.csv for a "
                            "labeling bug, or the SNR range may be too low to be detectable at all.")
    return ("OK", f"AUC={auc:.2f}, accuracy={acc:.2f} — learnable but not trivial. "
                  f"This is the range you want for a dataset meant to train a real detector.")


def verdict_for_drones_active(result):
    if result is None:
        return ("SKIPPED", "Not enough examples in the smallest drones_active class to "
                            "cross-validate reliably.")
    acc = result["accuracy"]
    if acc >= 0.85:
        return ("OK", f"accuracy={acc:.2f} — raw power/spectral features recover the "
                       f"slot/gap-derived drones_active labels well. This means Member C's "
                       f"activity reconstruction (from slot_map + gap_list alone, no raw "
                       f"masks) is internally consistent with the actual waveform.")
    return ("WARNING", f"accuracy={acc:.2f} — lower than expected for what should be close to "
                        f"a power-detection problem. Worth spot-checking a few misclassified "
                        f"examples against the spectrogram.")


def write_report(outdir, interdrone_results, importances, drones_result, n_examples, fs):
    lines = []
    lines.append("# Dataset Readiness Check\n")
    lines.append(f"Examples evaluated: **{n_examples}** | sample rate: {fs:,} Hz\n")
    lines.append("All numbers below are computed from the real IQ files and real labels in "
                  "this dataset, using stratified k-fold cross-validation (out-of-fold "
                  "predictions only -- no model ever sees the examples it's scored on).\n")

    lines.append("## Task 1: `interdrone_present` (binary)\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in interdrone_results.items():
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
                      f"{r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} |")
    lines.append("")
    lines.append("Top features by RandomForest importance:\n")
    for feat, imp in importances.head(5).items():
        lines.append(f"- `{feat}`: {imp:.3f}")
    lines.append("")
    status, msg = verdict_for_interdrone(interdrone_results)
    lines.append(f"**Verdict: {status}** — {msg}\n")
    lines.append("See `interdrone_present_evaluation.png` (confusion matrix, ROC curve, "
                  "feature importance).\n")

    lines.append("## Task 2: `drones_active` (multiclass)\n")
    if drones_result is None:
        lines.append("Skipped — not enough examples in the smallest class.\n")
    else:
        lines.append(f"5-fold-equivalent ({drones_result['n_splits']}-fold) cross-validated "
                      f"accuracy: **{drones_result['accuracy']:.3f}**\n")
        lines.append("```")
        lines.append(drones_result["report"])
        lines.append("```")
        lines.append("See `drones_active_evaluation.png` for the confusion matrix.\n")
    status2, msg2 = verdict_for_drones_active(drones_result)
    lines.append(f"**Verdict: {status2}** — {msg2}\n")

    lines.append("## Overall\n")
    both_ok = (status == "OK") and (status2 in ("OK", "SKIPPED"))
    if both_ok:
        lines.append("✅ **Pipeline looks ready to move on.** Both labels are learnable from "
                      "the raw IQ using only simple hand-crafted features, and neither shows "
                      "signs of a leak or a labeling bug. A real model (raw-IQ CNN, etc.) "
                      "should be able to do at least this well.")
    else:
        lines.append("⚠️ **Fix the flagged item(s) above before treating this as final.** "
                      "Re-run `01_member_C_pipeline.py` with adjusted parameters (SNR range, "
                      "--insert-prob) and re-check.")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Real, cross-validated readiness check for Member C's dataset")
    ap.add_argument("--dataset_dir", required=True, help="folder containing labels.csv and iq/")
    ap.add_argument("--fs", type=float, default=None, help="sample rate in Hz (auto-read from metadata.json if omitted)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    fs = args.fs
    if fs is None:
        meta_path = os.path.join(args.dataset_dir, "metadata.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path, encoding="utf-8") as f:
                fs = json.load(f)["fs_hz"]
        else:
            raise SystemExit("Pass --fs explicitly; metadata.json not found in --dataset_dir.")

    print(f"[1/4] Loading dataset from {args.dataset_dir} (fs={fs:,.0f} Hz)...")
    df, feat_df = load_dataset(args.dataset_dir, fs)
    print(f"       {len(df)} examples, {feat_df.shape[1]} features extracted per example")

    print("[2/4] Cross-validating interdrone_present classifiers (real fit/predict, held-out only)...")
    interdrone_results, importances, _ = evaluate_interdrone(df, feat_df, args.seed, args.dataset_dir)
    for name, r in interdrone_results.items():
        print(f"       {name}: accuracy={r['accuracy']:.3f} roc_auc={r['roc_auc']:.3f}")

    print("[3/4] Cross-validating drones_active classifier...")
    drones_result = evaluate_drones_active(df, feat_df, args.seed, args.dataset_dir)
    if drones_result:
        print(f"       RandomForest: accuracy={drones_result['accuracy']:.3f} "
              f"({drones_result['n_splits']}-fold)")

    print("[4/4] Writing readiness_report.md...")
    report = write_report(args.dataset_dir, interdrone_results, importances, drones_result, len(df), fs)
    report_path = os.path.join(args.dataset_dir, "readiness_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nDone. See {report_path}")


if __name__ == "__main__":
    main()
