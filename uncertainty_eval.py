import pickle
import numpy as np
import torch
import csv
from sklearn.metrics import roc_curve

def compute_entropy(logits):
    if isinstance(logits, torch.Tensor):
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)

def find_best_threshold(preds, logits, targets):
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

    # summary
    total = len(targets)
    errors = (~correct).sum()
    rights = correct.sum()
    high_unc_errors = ((~correct) & uncertain).sum()
    low_unc_rights = (correct & ~uncertain).sum()
    print(f"\n=== {method_name} ===")
    print(f"Chosen threshold: {threshold:.4f}")
    print(f"Total: {total}, Errors: {errors}, Correct: {rights}")
    print(f"Errors flagged uncertain: {high_unc_errors}/{errors} "
          f"({high_unc_errors/errors*100:.2f}%)")
    print(f"Correct flagged certain: {low_unc_rights}/{rights} "
          f"({low_unc_rights/rights*100:.2f}%)")
    print(f"Avg entropy—errors: {ent[~correct].mean():.4f}, "
          f"rights: {ent[correct].mean():.4f}")
    print(f"Overall test accuracy: {correct.sum()/len(correct)*100:.2f}%")
    print(f"Accuracy on low-uncertainty samples (< {threshold:.4f}): "
        f"{acc_low_unc*100:.2f}%")


    # write CSV
    fname = method_name.lower().replace(' ', '_') + '_uncertainty.csv'
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pred', 'target', 'entropy', 'correct', 'uncertain'])
        for p, t, e, c, u in zip(preds, targets, ent, correct, uncertain):
            writer.writerow([int(p), int(t), float(e), int(c), int(u)])
    print(f"Saved per-sample data to {fname}")

def load_or_none(name):
    try:
        return pickle.load(open(name, 'rb'))
    except FileNotFoundError:
        return None

def main():
    
    g_preds  = pickle.load(open('gaussian_after_preds.pkl',       'rb'))
    g_logits = pickle.load(open('gaussian_after_predictions.pkl', 'rb'))
    g_targs  = pickle.load(open('gaussian_after_targets.pkl',     'rb'))

    val = load_or_none('gaussian_val_predictions.pkl')
    if val is not None:
        g_val_preds  = pickle.load(open('gaussian_val_preds.pkl',       'rb'))
        g_val_logits = pickle.load(open('gaussian_val_predictions.pkl', 'rb'))
        g_val_targs  = pickle.load(open('gaussian_val_targets.pkl',     'rb'))
        g_thresh = find_best_threshold(g_val_preds, g_val_logits, g_val_targs)
    else:
        g_thresh = float(np.median(compute_entropy(g_logits)))
        print("Warning: no gaussian_val_*.pkl found, using median for threshold")

    evaluate(g_preds, g_logits, g_targs, g_thresh, "Gaussian-Calibration")

    ds_preds  = pickle.load(open('ds_after_preds.pkl',       'rb'))
    ds_logits = pickle.load(open('ds_after_predictions.pkl', 'rb'))
    ds_targs  = pickle.load(open('ds_after_targets.pkl',     'rb'))

    val = load_or_none('ds_val_predictions.pkl')
    if val is not None:
        ds_val_preds  = pickle.load(open('ds_val_preds.pkl',       'rb'))
        ds_val_logits = pickle.load(open('ds_val_predictions.pkl', 'rb'))
        ds_val_targs  = pickle.load(open('ds_val_targets.pkl',     'rb'))
        ds_thresh = find_best_threshold(ds_val_preds, ds_val_logits, ds_val_targs)
    else:
        ds_thresh = float(np.median(compute_entropy(ds_logits)))
        print("Warning: no ds_val_*.pkl found, using median for threshold")

    evaluate(ds_preds, ds_logits, ds_targs, ds_thresh, "Density-Softmax")

    
if __name__ == "__main__":
    main()
