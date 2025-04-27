import numpy as np
import os
import argparse
import pickle

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torcheval.metrics import (
    MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC,
    MulticlassConfusionMatrix, MulticlassF1Score,
    MulticlassPrecision, MulticlassRecall
)
from torchmetrics.functional.classification import multiclass_calibration_error as mce

from models import CNN, RealNVP
from data import EEGDataset
from utils import *

parser = argparse.ArgumentParser(description='Run EEG Raw experiments with tuned density-softmax fusion')
parser.add_argument('--num_channels', type=int, default=19)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--remove_channels', type=bool, default=True)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str, default='./meta_data')
parser.add_argument('--results_dir', type=str, default='./results_split_optimal')
parser.add_argument('--weights_dir', type=str, default='./weights_split_optimal')
parser.add_argument('--data_path', type=str, default='./data/wmci_wctrl_200-contig-len_time_domain.pkl')
parser.add_argument('--sf', type=float, default=250.)
parser.add_argument('--contig_len', type=int, default=200)
parser.add_argument('--art_thresh', type=int, default=0)
parser.add_argument('--lowcut', type=float, default=4.)
parser.add_argument('--highcut', type=float, default=40.)
parser.add_argument('--min_contigs', type=int, default=3)
parser.add_argument('--include_evt', type=bool, default=False)
parser.add_argument('--balanced', type=bool, default=False)
parser.add_argument('--norm_type', type=str, default='meanstd')
parser.add_argument('--study', type=str, default='wmci')
parser.add_argument('--treat', type=str, default='wctrl')
parser.add_argument('--model_type', type=str, default='cnn')
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--model_conf_path', type=str, default='./grid_confs/conf')
parser.add_argument('--split', type=float, default=0.75)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--runs', type=int, default=1)

# Fusion hyperparameters (optimal)
parser.add_argument('--pre_train_epochs', type=int, default=2)
parser.add_argument('--n_coupling_layers', type=int, default=3)
parser.add_argument('--coupling_hidden_dim', type=int, default=64)
parser.add_argument('--flow_lr', type=float, default=0.001672001616746327)
parser.add_argument('--flow_wd', type=float, default=0.002404240921660761)
parser.add_argument('--fusion_tau', type=float, default=2.3462427949136546)
parser.add_argument('--fusion_lambda', type=float, default=0.7298979719100968)
parser.add_argument('--norm_pct', type=float, default=0.782523346244629)
parser.add_argument('--softmax_temp', type=float, default=2.3751007587776183)

args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
    print(f"\t{k}: {v}")

# Set device
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.cuda.set_device(torch.cuda.current_device())

# Get channels
rm_ch = set()
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set(int(i) for i in args.rm_ch.split('_'))
    args.rm_ch_str = args.rm_ch
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# Load data
all_data = pickle.load(open(args.data_path, 'rb'))

# Create output dirs
os.makedirs(args.meta_dir, exist_ok=True)
os.makedirs(f"{args.results_dir}_{args.rm_ch_str}", exist_ok=True)
os.makedirs(f"{args.weights_dir}_{args.rm_ch_str}", exist_ok=True)

# Define types and load CNN config
typs = {args.study: 0, args.treat: 1}
conf_path = f"{args.model_conf_path}_{args.model_type}_{args.contig_len}_{len(channels)}_best.txt"
conf = read_grid_confs(conf_path)[0]

print('Channels:', channels)

for run in range(args.runs):
    # Get cleaned locs and info
    locs = get_clean_locs(args, all_data, typs)
    info = get_info(locs, args)
    print('usable contigs:', info)

    # Patient splits
    train_locs, test_locs = get_train_test_locs(args, all_data, locs)

    # Open result log
    out_file = open(f"{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_run{run}.txt", 'w')
    out_file.write(f"Run: {run}\nContigs: {len(locs)}\n")

    # PSD dimension and normalization
    args.num_psd = all_data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]
    norms = get_norms(args, all_data, channels, typs)

    # DataLoaders
    train_ds = EEGDataset(all_data, train_locs, args.study, args.treat, norms)
    test_ds = EEGDataset(all_data, test_locs, args.study, args.treat, norms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Model and optimizer
    model = CNN(len(channels), args.contig_len, args.num_classes, conf[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(model)

    # Pre-train conv layers and collect latents
    train_latents = []
    model.train()
    for epoch in range(args.pre_train_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = nll(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # collect latents on first epoch
        if epoch == 0:
            with torch.no_grad():
                for X, _ in train_loader:
                    z = model.conv_layers(X.to(device)).view(X.size(0), -1)
                    train_latents.append(z)
    train_latents = torch.cat(train_latents, dim=0)

    # Fit RealNVP
    flow_model = RealNVP(
        num_coupling_layers=args.n_coupling_layers,
        input_dim=train_latents.size(1),
        hidden_dim=args.coupling_hidden_dim
    ).to(device)
    flow_opt = optim.Adam(flow_model.parameters(), lr=args.flow_lr, weight_decay=args.flow_wd)
    flow_model.train()
    for _ in tqdm(range(1000), desc='Fitting flow'):
        flow_opt.zero_grad()
        loss = flow_model.log_loss(train_latents)
        loss.backward()
        flow_opt.step()
    flow_model.eval()

    # Calibration
    with torch.no_grad():
        log_lk = flow_model.score_samples(train_latents).unsqueeze(1)
        mean_log, std_log = log_lk.mean(0), log_lk.std(0) + 1e-6
        norm_log = (log_lk - mean_log) / std_log
        scale_lk = torch.exp(norm_log)

    # Evaluate flow-only accuracy
    flow_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            z = model.conv_layers(X).view(X.size(0), -1)
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            norm_log = (log_lk - mean_log) / std_log
            scale_lk = torch.exp(norm_log)
            fused = model.fc_layers(z) * (scale_lk ** args.fusion_tau) + args.fusion_lambda * norm_log
            logits = fused / args.softmax_temp
            flow_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    flow_acc = flow_correct / total
    out_file.write(f"Flow test acc: {flow_acc:.4f}\n")

    # Freeze conv layers
    for p in model.conv_layers.parameters():
        p.requires_grad = False

    # Density-softmax fine-tuning
    model.train()
    for epoch in range(args.num_epochs - args.pre_train_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            z = model.conv_layers(X).view(X.size(0), -1)
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            norm_log = (log_lk - mean_log) / std_log
            scale_lk = torch.exp(norm_log)
            fused = model.fc_layers(z) * (scale_lk ** args.fusion_tau) + args.fusion_lambda * norm_log
            logits = fused / args.softmax_temp
            loss = nll(logits, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
    
    # Evaluate density-softmax accuracy
    ds_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            z = model.conv_layers(X).view(X.size(0), -1)
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            norm_log = (log_lk - mean_log) / std_log
            scale_lk = torch.exp(norm_log)
            fused = model.fc_layers(z) * (scale_lk ** args.fusion_tau) + args.fusion_lambda * norm_log
            logits = fused / args.softmax_temp
            ds_correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    ds_acc = ds_correct / total
    out_file.write(f"Density-softmax test acc: {ds_acc:.4f}\n")

    # Save models
    torch.save(model.state_dict(), f"{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{run}_ds.pth")
    torch.save(flow_model.state_dict(), f"{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{run}_flow.pth")
    out_file.close()
    
    print(f"Run {run} complete | Flow acc: {flow_acc:.4f} | DS acc: {ds_acc:.4f}")
    exit()
