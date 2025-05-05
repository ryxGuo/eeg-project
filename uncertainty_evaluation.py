import os
import glob
import pickle
import numpy as np
import torch
import csv
from sklearn.metrics import roc_curve


def compute_entropy(logits):
    # logits: Torch Tensor or NumPy array
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def find_best_threshold(preds, logits, targets):
    # Convert tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    ent = compute_entropy(logits)
    labels = (preds != targets).astype(int)
    fpr, tpr, thr = roc_curve(labels, ent)
    J = tpr - fpr
    best_idx = np.argmax(J)
    return thr[best_idx]


def evaluate(preds, logits, targets, threshold, method_name):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    ent = compute_entropy(logits)

    correct = preds == targets
    uncertain = ent >= threshold

    low_unc_mask = ent < threshold
    n_low_unc = low_unc_mask.sum()
    acc_low_unc = (correct[low_unc_mask].sum() / n_low_unc) if n_low_unc > 0 else 0.0

    total = len(targets)
    errors = (~correct).sum()
    rights = correct.sum()
    high_unc_errors = ((~correct) & uncertain).sum()
    low_unc_rights = (correct & ~uncertain).sum()

    print(f"\n=== {method_name} ===")
    print(f"Chosen threshold: {threshold:.4f}")
    print(f"Total: {total}, Errors: {errors}, Correct: {rights}")
    print(f"Errors flagged uncertain: {high_unc_errors}/{errors} ({high_unc_errors/errors*100:.2f}%)")
    print(f"Correct flagged certain: {low_unc_rights}/{rights} ({low_unc_rights/rights*100:.2f}%)")
    print(f"Avg entropy—errors: {ent[~correct].mean():.4f}, rights: {ent[correct].mean():.4f}")
    print(f"Overall test accuracy: {rights/total*100:.2f}%")
    print(f"Accuracy on low-uncertainty samples (< {threshold:.4f}): {acc_low_unc*100:.2f}%")

    # write CSV
    fname = method_name.lower().replace(' ', '_') + '_uncertainty.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pred', 'target', 'entropy', 'correct', 'uncertain'])
        for p, t, e, c, u in zip(preds, targets, ent, correct, uncertain):
            writer.writerow([int(p), int(t), float(e), int(c), int(u)])
    print(f"Saved per-sample data to {fname}")


def load_or_none(fname):
    try:
        return pickle.load(open(fname, 'rb'))
    except (FileNotFoundError, EOFError):
        return None


def main():
    # Automatically find all model prefixes from *_after_preds.pkl files
    pred_files = glob.glob('*_after_preds.pkl')
    prefixes = sorted({os.path.basename(f).replace('_after_preds.pkl', '') for f in pred_files})

    if not prefixes:
        print("No '*_after_preds.pkl' files found in current directory.")
        return

    for prefix in prefixes:
        method_name = prefix.replace('_', ' ').replace('-', ' ').title()
        print(f"\nRunning uncertainty eval for: {method_name}\n{'-'*50}")

        preds_file   = f"{prefix}_after_preds.pkl"
        logits_file  = f"{prefix}_after_predictions.pkl"
        targets_file = f"{prefix}_after_targets.pkl"

        preds   = pickle.load(open(preds_file, 'rb'))
        logits  = pickle.load(open(logits_file, 'rb'))
        targets = pickle.load(open(targets_file, 'rb'))

        # Check for validation set to find optimal threshold
        val_logits_file = f"{prefix}_val_predictions.pkl"
        if os.path.exists(val_logits_file):
            val_preds_file  = f"{prefix}_val_preds.pkl"
            val_targets_file= f"{prefix}_val_targets.pkl"
            val_preds  = pickle.load(open(val_preds_file, 'rb'))
            val_logits = pickle.load(open(val_logits_file, 'rb'))
            val_targets= pickle.load(open(val_targets_file, 'rb'))
            threshold = find_best_threshold(val_preds, val_logits, val_targets)
        else:
            threshold = float(np.median(compute_entropy(logits)))
            print(f"Warning: no validation files for '{prefix}', using median entropy {threshold:.4f}")

        evaluate(preds, logits, targets, threshold, method_name)


if __name__ == "__main__":
    main()
