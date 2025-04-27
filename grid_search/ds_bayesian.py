import numpy as np
import os
import argparse
import pickle

import optuna

import itertools
from collections import namedtuple

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn, optim, autograd
from torch.utils.data import DataLoader

from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchmetrics.functional.classification import multiclass_calibration_error as mce

from models import CNN, RealNVP
from data import EEGDataset
from utils import *

parser = argparse.ArgumentParser(description='Run EEG Raw experiments')
parser.add_argument('--num_channels', type=int, default=19)     
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--remove_channels', type=bool, default=True)
parser.add_argument('--rm_ch', type=str, default='14_15')
parser.add_argument('--meta_dir', type=str,default='./meta_data')
parser.add_argument('--results_dir', type=str,default='./results_split')
parser.add_argument('--weights_dir', type=str,default='./weights_split')
parser.add_argument('--data_path', type=str,default='./data/kmci_kctrl_kdem')
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
parser.add_argument('--runs', type=int, default=5)

# flow hyperparameters
parser.add_argument('--n_coupling_layers', type=int,   default=6)
parser.add_argument('--coupling_hidden_dim', type=int, default=64)
parser.add_argument('--pre_train_epochs', type=int, default=None)
parser.add_argument('--flow_lr',            type=float, default=0.01)
parser.add_argument('--flow_wd',            type=float, default=0.01)

# # density‑softmax fusion knobs
parser.add_argument('--fusion_tau',         type=float, default=1.0)
parser.add_argument('--fusion_lambda',      type=float, default=0.0)
parser.add_argument('--norm_pct',           type=float, default=1.0)
parser.add_argument('--softmax_temp',       type=float, default=1.0)




args = parser.parse_args()
args.pre_train_epochs    = 2
args.n_coupling_layers   = 3
args.coupling_hidden_dim = 64
args.flow_lr             = 0.001672001616746327
args.flow_wd             = 0.002404240921660761



print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

# Set CUDA
# if args.cuda:
#     torch.cuda.set_device(args.gpu)
if torch.cuda.is_available():
    torch.cuda.set_device(torch.cuda.current_device())


ParamSet = namedtuple("ParamSet",
                      ["fusion_tau", "fusion_lambda", "norm_pct", "softmax_temp"])
    
# Get channels
rm_ch = set([])
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set([int(i) for i in args.rm_ch.split('_')])
    args.rm_ch_str = args.rm_ch
    
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# Load data.
# if args.rm_ch_str != '':
#     data_path = args.data_path + '_' + args.rm_ch_str +'_notevt_' + str(args.contig_len) + '.pkl'
# else:
#     data_path = args.data_path +'_notevt_' + str(args.contig_len) + '.pkl'
data_path = './data/wmci_wctrl_200-contig-len_time_domain.pkl'
all_data = pickle.load(open(data_path, 'rb'))

# create dirs
if not os.path.exists(args.meta_dir):
    os.makedirs(args.meta_dir)
if not os.path.exists(args.results_dir + '_' + args.rm_ch_str):
    os.makedirs(args.results_dir + '_' + args.rm_ch_str)
if not os.path.exists(args.weights_dir + '_' + args.rm_ch_str):
    os.makedirs(args.weights_dir + '_' + args.rm_ch_str)

    
# Define class types
typs = {args.study: 0, args.treat: 1}

conf_path = args.model_conf_path + '_' + args.model_type + '_' + str(args.contig_len) + '_' + str(len(channels)) + '_best.txt'

conf = read_grid_confs(conf_path)[0]

print ('Channels:', channels)

def run_one(args: argparse.Namespace,
            all_data: dict,
            typs: dict,
            channels: np.ndarray,
            conf) -> (float, float):
    """
    Run one full experiment: CNN pre-train, fit RealNVP, DS fine-tune.
    Returns the final density-softmax test accuracy.
    """
    # args.pre_train_epochs    = params.pre_train_epochs
    # args.n_coupling_layers   = params.n_coupling_layers
    # args.coupling_hidden_dim = params.coupling_hidden_dim
    # args.flow_lr             = params.flow_lr
    # args.flow_wd             = params.flow_wd

    # locs = get_clean_locs(args, all_data, typs)
    # train_locs, test_locs = get_train_test_locs(args, all_data, locs)

    # norms = get_norms(args, all_data, channels, typs)
    # train_ds = EEGDataset(all_data, train_locs, args.study, args.treat, norms)
    # test_ds  = EEGDataset(all_data, test_locs,  args.study, args.treat, norms)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    # test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    # model = CNN(len(channels), args.contig_len, args.num_classes, conf[1]).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # model.train()
    # train_latents = []
    # for epoch in range(args.pre_train_epochs):
    #     for X, y in train_loader:
    #         X, y = X.to(device), y.to(device)
    #         logits = model(X)
    #         loss = nll(logits, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # collect latents from conv_layers after first epoch
    #     if epoch == 0:
    #         with torch.no_grad():
    #             for X, _ in train_loader:
    #                 X = X.to(device)
    #                 z = model.conv_layers(X).view(X.size(0), -1)
    #                 train_latents.append(z)
    # train_latents = torch.cat(train_latents, dim=0)

    # flow_model = RealNVP(
    #     num_coupling_layers=args.n_coupling_layers,
    #     input_dim=train_latents.size(1),
    #     hidden_dim=args.coupling_hidden_dim
    # ).to(device)
    # flow_opt = optim.Adam(flow_model.parameters(), lr=args.flow_lr, weight_decay=args.flow_wd)
    # flow_model.train()
    # for _ in range(1000):
    #     flow_opt.zero_grad()
    #     loss = flow_model.log_loss(train_latents)
    #     loss.backward()
    #     flow_opt.step()
    # flow_model.eval()

    # with torch.no_grad():
    #     train_lk = torch.exp(flow_model.score_samples(train_latents)).unsqueeze(1)
    #     train_norm = torch.quantile(train_lk, args.norm_pct)


    # flow_correct = 0
    # flow_total   = 0
    # model.eval()
    # with torch.no_grad():
    #     for X, y in test_loader:
    #         X, y = X.to(device), y.to(device)
    #         z = model.conv_layers(X).view(X.size(0), -1)
    #         lk = torch.exp(flow_model.score_samples(z)).unsqueeze(1) / train_norm
    #         logits = model.fc_layers(z) * lk.float()
    #         flow_correct += (logits.argmax(1) == y).sum().item()
    #         flow_total   += y.size(0)
    # flow_acc = flow_correct / flow_total

    # for p in model.conv_layers.parameters():
    #     p.requires_grad = False
    # model.train()
    # for epoch in range(args.num_epochs - args.pre_train_epochs):
    #     for X, y in train_loader:
    #         X, y = X.to(device), y.to(device)
    #         z = model.conv_layers(X).view(X.size(0), -1)
    #         logits = model.fc_layers(z)
    #         lk = torch.exp(flow_model.score_samples(z)).unsqueeze(1) / train_norm
    #         fused = logits * (lk ** args.fusion_tau) + args.fusion_lambda * torch.log(lk + 1e-8)
    #         scaled = fused / args.softmax_temp
    #         loss = nll(scaled, y)
    #         optimizer.zero_grad()
    #         loss.backward(retain_graph=True)
    #         optimizer.step()

    # ds_correct = 0
    # ds_total   = 0
    # model.eval()
    # with torch.no_grad():
    #     for X, y in test_loader:
    #         X, y = X.to(device), y.to(device)
    #         z = model.conv_layers(X).view(X.size(0), -1)
    #         lk = torch.exp(flow_model.score_samples(z)).unsqueeze(1) / train_norm
    #         logits = model.fc_layers(z) * lk.float()
    #         ds_correct += (logits.argmax(1) == y).sum().item()
    #         ds_total   += y.size(0)
    # ds_acc = ds_correct / ds_total

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    locs = get_clean_locs(args, all_data, typs)
    train_locs, test_locs = get_train_test_locs(args, all_data, locs)

    norms       = get_norms(args, all_data, channels, typs)
    train_ds    = EEGDataset(all_data, train_locs, args.study, args.treat, norms)
    test_ds     = EEGDataset(all_data, test_locs,  args.study, args.treat, norms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    model     = CNN(len(channels), args.contig_len, args.num_classes, conf[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    train_latents = []
    for epoch in range(args.pre_train_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = nll(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 0:
            with torch.no_grad():
                for X, _ in train_loader:
                    z = model.conv_layers(X.to(device)).view(X.size(0), -1)
                    train_latents.append(z)
    train_latents = torch.cat(train_latents, dim=0)

    flow_model = RealNVP(
        num_coupling_layers=args.n_coupling_layers,
        input_dim=train_latents.size(1),
        hidden_dim=args.coupling_hidden_dim
    ).to(device)
    flow_opt = optim.Adam(flow_model.parameters(),
                          lr=args.flow_lr,
                          weight_decay=args.flow_wd)
    flow_model.train()
    for _ in range(1000):
        flow_opt.zero_grad()
        loss = flow_model.log_loss(train_latents)
        loss.backward()
        flow_opt.step()
    flow_model.eval()

    with torch.no_grad():
        train_lk   = torch.exp(flow_model.score_samples(train_latents)).unsqueeze(1)
        train_norm = torch.quantile(train_lk, args.norm_pct)

    flow_correct = 0
    flow_total   = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            z     = model.conv_layers(X).view(X.size(0), -1)
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            mean_log = log_lk.mean(0, keepdim=True)
            std_log  = log_lk.std(0, keepdim=True) + 1e-6
            norm_log_lk = (log_lk - mean_log) / std_log

            scaled_lk = torch.exp(norm_log_lk)

            fused = model.fc_layers(z) * (scaled_lk ** args.fusion_tau) \
                    + args.fusion_lambda * norm_log_lk

            logits = fused / args.softmax_temp
            flow_correct += (logits.argmax(1) == y).sum().item()
            flow_total   += y.size(0)
    flow_acc = flow_correct / flow_total

    for p in model.conv_layers.parameters():
        p.requires_grad = False
    model.train()
    for epoch in range(args.num_epochs - args.pre_train_epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            z      = model.conv_layers(X).view(X.size(0), -1)
            # logits = model.fc_layers(z)
            # lk     = torch.exp(flow_model.score_samples(z)).unsqueeze(1) / train_norm
            # fused  = logits * (lk ** args.fusion_tau) \
            #          + args.fusion_lambda * torch.log(lk + 1e-8)
            # scaled = fused / args.softmax_temp
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            mean_log = log_lk.mean(0, keepdim=True)
            std_log  = log_lk.std(0, keepdim=True) + 1e-6
            norm_log_lk = (log_lk - mean_log) / std_log
            scaled_lk = torch.exp(norm_log_lk)
            fused = model.fc_layers(z) * (scaled_lk ** args.fusion_tau) \
                    + args.fusion_lambda * norm_log_lk
            logits = fused / args.softmax_temp
            loss   = nll(logits, y)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

    ds_correct = 0
    ds_total   = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            z      = model.conv_layers(X).view(X.size(0), -1)
            # lk     = torch.exp(flow_model.score_samples(z)).unsqueeze(1) / train_norm
            # logits = model.fc_layers(z) * lk.float()
            # raw_lk = torch.exp(flow_model.score_samples(z)).unsqueeze(1)
            # lk_mean = raw_lk.mean(dim=0, keepdim=True)
            # lk_std  = raw_lk.std(dim=0, keepdim=True).clamp(min=1e-6)
            # norm_lk = (raw_lk - lk_mean) / lk_std

            # fused  = logits * (norm_lk ** args.fusion_tau) \
            #          + args.fusion_lambda * torch.log(norm_lk + 1e-8)
            # scaled = fused / args.softmax_temp
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            mean_log = log_lk.mean(0, keepdim=True)
            std_log  = log_lk.std(0, keepdim=True) + 1e-6
            norm_log_lk = (log_lk - mean_log) / std_log
            scaled_lk = torch.exp(norm_log_lk)
            fused = model.fc_layers(z) * (scaled_lk ** args.fusion_tau) \
                    + args.fusion_lambda * norm_log_lk
            logits = fused / args.softmax_temp

            ds_correct += (logits.argmax(1) == y).sum().item()

            # ds_correct += (logits.argmax(1) == y).sum().item()
            ds_total   += y.size(0)
    ds_acc = ds_correct / ds_total

    return flow_acc, ds_acc


def objective(trial):
    # params = ParamSet(
    #     pre_train_epochs    = trial.suggest_int("pre_train_epochs",    1, 4),
    #     n_coupling_layers   = trial.suggest_int("n_coupling_layers",   2, 6),
    #     coupling_hidden_dim = trial.suggest_categorical("coupling_hidden_dim", [32, 64]),
    #     flow_lr             = trial.suggest_loguniform("flow_lr",   1e-4, 1e-2),
    #     flow_wd             = trial.suggest_loguniform("flow_wd",   1e-4, 1e-2),
    # )
    # flow_acc, ds_acc = run_one(params, args, all_data, typs, channels, conf)

    args.fusion_tau    = trial.suggest_float("fusion_tau",    0.1, 5.0)
    args.fusion_lambda = trial.suggest_float("fusion_lambda", 0.0, 1.0)
    args.norm_pct      = trial.suggest_float("norm_pct",      0.5, 1.0)
    args.softmax_temp  = trial.suggest_float("softmax_temp",  0.1, 5.0)

    flow_acc, ds_acc = run_one(args, all_data, typs, channels, conf)
    trial.set_user_attr("flow_acc", flow_acc)

    return ds_acc


if __name__ == "__main__":
    all_data = pickle.load(open(data_path, "rb"))
    typs     = {args.study: 0, args.treat: 1}
    channels = np.array([i for i in range(args.num_channels)
                         if i not in set(map(int, args.rm_ch.split("_")))])
    conf     = read_grid_confs(conf_path)[0]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    best_ds   = study.best_value
    best_flow = study.best_trial.user_attrs["flow_acc"]
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    • {k}: {v}")
    print(f"Best DS test accuracy:          {best_ds:.4f}")
    print(f"Corresponding flow test accuracy: {best_flow:.4f}")





