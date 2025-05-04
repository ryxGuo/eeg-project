import numpy as np
import os
import argparse
import pickle

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

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

parser.add_argument('--mask_type', type=str, default='block',
                   choices=['alternate', 'random', 'block'])
parser.add_argument('--prior_type', type=str, default='uniform',
                   choices=['gaussian', 'laplace', 'uniform'])
parser.add_argument('--mask_ratio', type=float, default=0.5)

args = parser.parse_args()

args.pre_train_epochs    = 2
args.n_coupling_layers   = 4
args.coupling_hidden_dim = 128
args.flow_lr             = 0.0001
args.flow_wd             = 0.002


print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

# Set CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.set_device(torch.cuda.current_device())
    
# Get channels
rm_ch = set([])
args.rm_ch_str = ''
if args.remove_channels:
    rm_ch = set([int(i) for i in args.rm_ch.split('_')])
    args.rm_ch_str = args.rm_ch
    
channels = np.array([i for i in range(args.num_channels) if i not in rm_ch])

# Load data.
# data_path = './data/wmci_wctrl_200-contig-len_time_domain.pkl'
data_path = '/content/drive/MyDrive/eeg_mci-main/eeg_mci-main/data/wmci_wctrl_200-contig-len_time_domain.pkl'
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

for run in range(args.runs):

    # Get clean locations
    locs = get_clean_locs(args, all_data, typs)

    print (len(locs))

    # Get data infdo
    info = get_info(locs, args)

    # Print class wise patient info.
    print (info)

    pats = {args.study: set(), args.treat: set()}
    for typ in info:
        for pat, _ in info[typ]:
            pats[typ].add(pat)

    for typ in info:
        total = 0
        for i, pat in enumerate(info[typ]):
            total += pat[1]
        print (typ, total / i, i)

    print (pats)

    data = all_data

    # Get clean train_test locations
    train_locs, test_locs = get_train_test_locs(args, data, locs)

    train_pats = {args.study: set(), args.treat: set()}
    test_pats = {args.study: set(), args.treat: set()}

    for loc in train_locs:
        train_pats[loc[0]].add(loc[1])
    for loc in test_locs:
        test_pats[loc[0]].add(loc[1])
    print (train_pats)
    print (test_pats)
    
    file = open(f'{args.results_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_ds.txt', 'a')

    file.write(f'<<>>Run:{run}\n')
    file.write(f'Number of total thresholded contigs: {len(locs)}\n')
    file.write(f'Usable contigs: {str(info)}\n')
    file.write(f'Train patients: {str(train_pats)}\n')
    file.write(f'Test patients: {str(test_pats)}\n')

    args.num_psd = data[train_locs[0][0]][train_locs[0][1]][train_locs[0][2]].shape[1]

    # For normalization
    n = get_norms(args, data, channels, typs)

    train_locs, val_locs = train_test_split(train_locs, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = EEGDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
    val_dataset = EEGDataset(data_dict=data, locs=val_locs, study=args.study, treat=args.treat, norms=n)
    test_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n)

    # Get Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    # Get Model
    if args.model_type == 'mlp':
        model = MLP(num_channels=len(channels), num_psd=args.num_psd, out_dim=args.num_classes, conf=conf[1])
    else:
        model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

    model = model.to(device)
    print (model)

    optimizer = optim.Adam(model.parameters(), args.lr)
    # pre_train_epochs = int (args.num_epochs * 0.75)
    pre_train_epochs = args.pre_train_epochs
    # pre_train_epochs = args.num_epochs
    pbar = tqdm(range(pre_train_epochs))

    train_loss = 0
    train_latents = None

    for epoch in pbar:
        model.train()
        acc = 0
        tot = 0
        for idx, (X, y) in enumerate(train_dataloader):
            logits = model(X)
            loss = nll(logits, y)

            acc += mean_accuracy(logits, y, reduce='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.set_description_str(desc='iter: '+str(idx)+', loss: '+str(train_loss/(idx+1)) + f', train acc: {str((acc/(idx+1)))}')
        
        train_loss /= (idx + 1)
        acc /= (idx + 1)

        model.eval()

        acc = 0
        tot = 1
        print ('Evaluating on train data...')
        for X, y in tqdm(train_dataloader):
            if not epoch:
                encoding = model.conv_layers(X).view(X.shape[0], -1)
                logits = model.fc_layers(encoding)
                if train_latents is None:
                    train_latents = encoding.detach()
                else:
                    train_latents = torch.cat((train_latents, encoding.detach()), dim=0)
            else:
                break
                logits = model(X)
            acc += mean_accuracy(logits, y)
            tot += logits.shape[0]
        print ('train acc: '+str(acc/tot))
        file.write(f'epoch: {epoch}, train acc: {str((acc/tot))}, ')

        acc = 0
        tot = 0
        print ('Evaluating on test data...')
        for X, y in tqdm(test_dataloader):
            logits = model(X)
            acc += mean_accuracy(logits, y)
            tot += logits.shape[0]

        print ('test acc:', acc/tot)
        file.write(f'test acc: {str((acc/tot))}\n')

    # train_latents = torch.randn(1000, 96).cuda()

    # flow_model = RealNVP(num_coupling_layers=6, input_dim=train_latents.shape[1]).to(device)
    # flow_optimizer = optim.Adam(flow_model.parameters(), lr=0.01, weight_decay=0.01)

    # flow_model = RealNVP(
    #     num_coupling_layers   = args.n_coupling_layers,
    #     input_dim             = train_latents.shape[1],
    #     hidden_dim            = args.coupling_hidden_dim
    # ).to(device)

    flow_model = RealNVP(
        num_coupling_layers=args.n_coupling_layers,
        input_dim=train_latents.shape[1],
        hidden_dim=args.coupling_hidden_dim,
        mask_type=args.mask_type,
        prior_type=args.prior_type
    ).to(device)

    best_acc = 0
    best_config = {}

    flow_optimizer = optim.Adam(
        flow_model.parameters(),
        lr           = args.flow_lr, 
        weight_decay = args.flow_wd  
    )

    print ('Fitting flow model...')
    pbar = tqdm(range(1000))
    flow_model.train()
    for i in pbar:
        flow_optimizer.zero_grad()
        loss = flow_model.log_loss(train_latents)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        flow_optimizer.step()
        pbar.set_description_str(desc='iter: '+str(i)+', loss: '+str(loss))
    flow_model.eval()

    train_likelihood = torch.exp(flow_model.score_samples(train_latents))
    train_likelihood = train_likelihood.unsqueeze(1)
    train_likelihood_max = torch.max(train_likelihood)

    # # ——— RealNVP diagnostic ———
    # with torch.no_grad():
    #     # 1) log-density on true latents
    #     logp_train = flow_model.score_samples(train_latents)
    #     # 2) log-density on pure noise of same shape
    #     noise = torch.randn_like(train_latents)
    #     logp_noise = flow_model.score_samples(noise)

    #     print(f"Mean log‐prob (train latents): {logp_train.mean().item():.4f}")
    #     print(f"Mean log‐prob (noise):          {logp_noise.mean().item():.4f}")

    #     # Optional: histogram check
    #     import matplotlib.pyplot as plt
    #     plt.hist(logp_train.cpu().numpy(), bins=50, alpha=0.7, label="train")
    #     plt.hist(logp_noise.cpu().numpy(), bins=50, alpha=0.7, label="noise")
    #     plt.legend(); plt.show()
    # # ————————————————————————


    preds = None
    predictions = None
    targets = None

    acc = 0
    tot = 0
    print ('Evaluating on test data...')
    for X, y in tqdm(test_dataloader):
        z = model.conv_layers(X).view(X.shape[0], -1)
        log_lk = flow_model.score_samples(z).unsqueeze(1)
        mean_log = log_lk.mean(0, keepdim=True)
        std_log  = log_lk.std(0, keepdim=True) + 1e-6
        norm_log_lk = (log_lk - mean_log) / std_log

        # exponentiate
        scaled_lk = torch.exp(norm_log_lk)

        logits = model.fc_layers(z) * scaled_lk

        if preds is not None:
            predictions = torch.cat((predictions, logits.detach().cpu()))
            preds = torch.cat((preds, torch.argmax(logits, dim=1).detach().cpu()))
        else:
            predictions = logits.detach().cpu()
            preds = torch.argmax(logits, dim=1).detach().cpu()
        if targets is not None:
            targets = torch.cat((targets, y.cpu()))
        else:
            targets = y.cpu()
        acc += mean_accuracy(logits, y)
        tot += logits.shape[0]

    print ('Flow test acc:', acc/tot)
    file.write(f'Flow test acc: {str((acc/tot).item())}\n')
    pickle.dump(preds, open(f'ds_before_preds.pkl', 'wb'))
    pickle.dump(predictions, open(f'ds_before_predictions.pkl', 'wb'))
    pickle.dump(targets, open(f'ds_before_targets.pkl', 'wb'))

    torch.save(model.state_dict(), f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_ds_before.pth')
    torch.save(flow_model.state_dict(), f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_flow_ds.pth')


    print ('Continuing training with density softmax...')

    # Freeze conv layers.
    for param in model.conv_layers.parameters():
        param.requires_grad = False

    # optimizer = optim.Adam(model.fc_layers.parameters(), args.lr)
    train_loss = 0
    pbar = tqdm(range(args.num_epochs - pre_train_epochs))
    model.train()
    # pbar = tqdm(range(0))
    for e in pbar:
        acc = 0
        tot = 0
        for idx, (X, y) in enumerate(train_dataloader):
            loss_value = 0
            optimizer.zero_grad()
            
            z = model.conv_layers(X).view(X.size(0), -1)
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            mean_log = log_lk.mean(0, keepdim=True)
            std_log  = log_lk.std(0, keepdim=True) + 1e-6
            norm_log_lk = (log_lk - mean_log) / std_log
            scaled_lk = torch.exp(norm_log_lk)
            logits = model.fc_layers(z) * scaled_lk

            loss_value = nll(logits, y)

            acc += mean_accuracy(logits, y, reduce='mean')

            loss_value.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss_value.item()

            pbar.set_description_str(desc='iter: '+str(idx)+', loss: '+str(train_loss/(idx+1)) + f', train acc: {str((acc/(idx+1)))}')

        train_loss /= (idx + 1)
        acc /= (idx + 1)
        
        print("Training loss at epoch %d: %.4f" % (e, float(train_loss)))
        
    model.eval()

    preds = None
    predictions = None
    targets = None

    acc = 0
    tot = 0
    print ('Evaluating on test data...')
    for X, y in tqdm(test_dataloader):
        z = model.conv_layers(X).view(X.size(0), -1)
        log_lk = flow_model.score_samples(z).unsqueeze(1)
        mean_log = log_lk.mean(0, keepdim=True)
        std_log  = log_lk.std(0, keepdim=True) + 1e-6
        norm_log_lk = (log_lk - mean_log) / std_log
        scaled_lk = torch.exp(norm_log_lk)
        logits = model.fc_layers(z) * scaled_lk

        if preds is not None:
            predictions = torch.cat((predictions, logits.detach().cpu()))
            preds = torch.cat((preds, torch.argmax(logits, dim=1).detach().cpu()))
        else:
            predictions = logits.detach().cpu()
            preds = torch.argmax(logits, dim=1).detach().cpu()
        if targets is not None:
            targets = torch.cat((targets, y.cpu()))
        else:
            targets = y.cpu()
        acc += mean_accuracy(logits, y)
        tot += logits.shape[0]

    print ('ds test acc:', acc/tot)
    file.write(f'ds test acc: {str((acc/tot).item())}\n')
    pickle.dump(preds, open(f'ds_after_preds.pkl', 'wb'))
    pickle.dump(predictions, open(f'ds_after_predictions.pkl', 'wb'))
    pickle.dump(targets, open(f'ds_after_targets.pkl', 'wb'))

    torch.save(model.state_dict(), f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_ds_after.pth')

    # validation predictions/logits dump
    model.eval()
    val_preds = []
    val_logits = []
    val_targets = []

    with torch.no_grad():
        for X, y in val_dataloader:
            z = model.conv_layers(X).view(X.size(0), -1)
            log_lk = flow_model.score_samples(z).unsqueeze(1)
            mean_log = log_lk.mean(0, keepdim=True)
            std_log  = log_lk.std(0, keepdim=True) + 1e-6
            norm_log_lk = (log_lk - mean_log) / std_log
            scaled_lk = torch.exp(norm_log_lk)
            logits = model.fc_layers(z) * scaled_lk

            val_preds.append(torch.argmax(logits, dim=1).detach().cpu())
            val_logits.append(logits.detach().cpu())
            val_targets.append(y.cpu())

    val_preds = torch.cat(val_preds)
    val_logits = torch.cat(val_logits)
    val_targets = torch.cat(val_targets)

    pickle.dump(val_preds, open(f'val_preds.pkl', 'wb'))
    pickle.dump(val_logits, open(f'val_predictions.pkl', 'wb'))
    pickle.dump(val_targets, open(f'val_targets.pkl', 'wb'))

    exit()
    file.write('CONTIG WISE METRICS\n\n')
    acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
    acc_macro.update(predictions, targets)
    acc_none = MulticlassAccuracy(average=None, num_classes=2)
    acc_none.update(predictions, targets)
    acc = MulticlassAccuracy(num_classes=2)
    acc.update(predictions, targets)
    auroc = MulticlassAUROC(num_classes=2)
    auroc.update(predictions, targets)
    auprc = MulticlassAUPRC(num_classes=2)
    auprc.update(predictions, targets)
    recall_macro = MulticlassRecall(num_classes=2, average='macro')
    recall_macro.update(predictions, targets)
    recall_none = MulticlassRecall(num_classes=2, average=None)
    recall_none.update(predictions, targets)
    recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
    recall_weighted.update(predictions, targets)
    recall = MulticlassRecall(num_classes=2)
    recall.update(predictions, targets)
    precision_macro = MulticlassPrecision(num_classes=2, average='macro')
    precision_macro.update(predictions, targets)
    precision_none = MulticlassPrecision(num_classes=2, average=None)
    precision_none.update(predictions, targets)
    precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
    precision_weighted.update(predictions, targets)
    precision = MulticlassPrecision(num_classes=2)
    precision.update(predictions, targets)
    f1_macro = MulticlassF1Score(num_classes=2, average='macro')
    f1_macro.update(predictions, targets)
    f1_none = MulticlassF1Score(num_classes=2, average=None)
    f1_none.update(predictions, targets)
    f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
    f1_weighted.update(predictions, targets)
    f1 = MulticlassF1Score(num_classes=2)
    f1.update(predictions, targets)
    confusion = MulticlassConfusionMatrix(num_classes=2)
    confusion.update(predictions, targets)

    print (f'Mean test accuracy: {acc.compute().item()}')
    file.write(f'Mean test accuracy: {acc.compute().item()}\n')
    print (f'Class mean test accuracy: {acc_macro.compute().item()}')
    file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
    acc_none_vals = acc_none.compute()
    print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
    file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
    print (f'Test AUPRC: {auprc.compute().item()}')
    file.write(f'Test AUPRC: {auprc.compute().item()}\n')
    print (f'Test AUROC: {auroc.compute().item()}')
    file.write(f'Test AUROC: {auroc.compute().item()}\n')
    print (f'Mean test recall: {recall.compute().item()}')
    file.write(f'Mean test recall: {recall.compute().item()}\n')
    print (f'Class mean test recall: {recall_macro.compute().item()}')
    file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
    print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
    file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
    recall_none_vals = recall_none.compute()
    print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
    file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
    print (f'Mean test precision: {precision.compute().item()}')
    file.write(f'Mean test precision: {precision.compute().item()}\n')
    print (f'Class mean test precision: {precision_macro.compute().item()}')
    file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
    print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
    file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
    precision_none_vals = precision_none.compute()
    print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
    file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
    print (f'Mean test f1: {f1.compute().item()}')
    file.write(f'Mean test f1: {f1.compute().item()}\n')
    print (f'Class mean test f1: {f1_macro.compute().item()}')
    file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
    print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
    file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
    f1_none_vals = f1_none.compute()
    print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
    file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
    confuse_vals = confusion.compute()
    print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
    file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

    acc = 0
    tot = 0

    for X, y in train_dataloader:    
        logits = model(X)
        acc += mean_accuracy(logits, y)
        tot += logits.shape[0]
    print ('Train accuracy:', acc/tot)
    file.write(f'Mean train accuracy: {acc/tot}\n')

    file.write('PATIENT WISE METRICS\n\n')
    temp_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    count_0 = 0
    count_1 = 0
    count_pred_0 = 0
    count_pred_1 = 0
    prev_pid = 0

    accs = []

    pat_results = {
                args.study: {},
                args.treat: {}
                }

    acc = 0

    for i, (X, y) in enumerate(temp_loader):
        if test_locs[i][1] not in pat_results[test_locs[i][0]]:
            pat_results[test_locs[i][0]][test_locs[i][1]] = {'c': 0, 't': 0}
        logits = model(X)
        acc = mean_accuracy(logits, y)
        pat_results[test_locs[i][0]][test_locs[i][1]]['t'] += 1
        if acc == 1:
            pat_results[test_locs[i][0]][test_locs[i][1]]['c'] += 1

    for typ in pat_results:
        for pat in pat_results[typ]:
            pat_results[typ][pat]['acc'] = pat_results[typ][pat]['c'] / pat_results[typ][pat]['t']

    for typ in pat_results:
        print (typ)
        for pat in pat_results[typ]:
            print (pat, pat_results[typ][pat])

    total_acc = 0
    total_count = 0
    pat_predictions = []
    pat_targets = []
    for typ in pat_results:
        typ_acc = 0
        typ_count = 0
        for pat in pat_results[typ]:
            true = typs[typ]
            other = 1 if true == 0 else 0
            pat_targets.append(true)
            pred = [0., 0.]
            if pat_results[typ][pat]['acc'] >= 0.5:
                typ_acc += 1
            pred[true] = pat_results[typ][pat]['acc']
            pred[other] = 1 - pat_results[typ][pat]['acc']
            pat_predictions.append(pred)
            typ_count += 1
        print (typ, typ_acc, typ_count, typ_acc / typ_count)
        file.write(f'cls: {typ}, correct: {typ_acc}, total: {typ_count}, acc: {typ_acc/typ_count}\n')
        total_acc += typ_acc
        total_count += typ_count
    print (total_acc, total_count, total_acc / total_count)

    predictions = torch.tensor(pat_predictions)
    targets = torch.tensor(pat_targets)

    acc_macro = MulticlassAccuracy(average="macro", num_classes=2)
    acc_macro.update(predictions, targets)
    acc_none = MulticlassAccuracy(average=None, num_classes=2)
    acc_none.update(predictions, targets)
    acc = MulticlassAccuracy(num_classes=2)
    acc.update(predictions, targets)
    auroc = MulticlassAUROC(num_classes=2)
    auroc.update(predictions, targets)
    auprc = MulticlassAUPRC(num_classes=2)
    auprc.update(predictions, targets)
    recall_macro = MulticlassRecall(num_classes=2, average='macro')
    recall_macro.update(predictions, targets)
    recall_none = MulticlassRecall(num_classes=2, average=None)
    recall_none.update(predictions, targets)
    recall_weighted = MulticlassRecall(num_classes=2, average='weighted')
    recall_weighted.update(predictions, targets)
    recall = MulticlassRecall(num_classes=2)
    recall.update(predictions, targets)
    precision_macro = MulticlassPrecision(num_classes=2, average='macro')
    precision_macro.update(predictions, targets)
    precision_none = MulticlassPrecision(num_classes=2, average=None)
    precision_none.update(predictions, targets)
    precision_weighted = MulticlassPrecision(num_classes=2, average='weighted')
    precision_weighted.update(predictions, targets)
    precision = MulticlassPrecision(num_classes=2)
    precision.update(predictions, targets)
    f1_macro = MulticlassF1Score(num_classes=2, average='macro')
    f1_macro.update(predictions, targets)
    f1_none = MulticlassF1Score(num_classes=2, average=None)
    f1_none.update(predictions, targets)
    f1_weighted = MulticlassF1Score(num_classes=2, average='weighted')
    f1_weighted.update(predictions, targets)
    f1 = MulticlassF1Score(num_classes=2)
    f1.update(predictions, targets)
    confusion = MulticlassConfusionMatrix(num_classes=2)
    confusion.update(predictions, targets)

    print (f'Mean test accuracy: {acc.compute().item()}')
    file.write(f'Mean test accuracy: {acc.compute().item()}\n')
    print (f'Class mean test accuracy: {acc_macro.compute().item()}')
    file.write(f'Class mean test accuracy: {acc_macro.compute().item()}\n')
    acc_none_vals = acc_none.compute()
    print (f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}')
    file.write(f'Individual class test accuracies: "0" - {acc_none_vals[0].item()}, "1" - {acc_none_vals[1].item()}\n')
    print (f'Test AUPRC: {auprc.compute().item()}')
    file.write(f'Test AUPRC: {auprc.compute().item()}\n')
    print (f'Test AUROC: {auroc.compute().item()}')
    file.write(f'Test AUROC: {auroc.compute().item()}\n')
    print (f'Mean test recall: {recall.compute().item()}')
    file.write(f'Mean test recall: {recall.compute().item()}\n')
    print (f'Class mean test recall: {recall_macro.compute().item()}')
    file.write(f'Class mean test recall: {recall_macro.compute().item()}\n')
    print (f'Class weighted mean test recall: {recall_weighted.compute().item()}')
    file.write(f'Class weighted mean test recall: {recall_weighted.compute().item()}\n')
    recall_none_vals = recall_none.compute()
    print (f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}')
    file.write(f'Individual class test recalls: "0" - {recall_none_vals[0].item()}, "1" - {recall_none_vals[1].item()}\n')
    print (f'Mean test precision: {precision.compute().item()}')
    file.write(f'Mean test precision: {precision.compute().item()}\n')
    print (f'Class mean test precision: {precision_macro.compute().item()}')
    file.write(f'Class mean test precision: {precision_macro.compute().item()}\n')
    print (f'Class weighted mean test precision: {precision_weighted.compute().item()}')
    file.write(f'Class weighted mean test precision: {precision_weighted.compute().item()}\n')
    precision_none_vals = precision_none.compute()
    print (f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}')
    file.write(f'Individual class test precisions: "0" - {precision_none_vals[0].item()}, "1" - {precision_none_vals[1].item()}\n')
    print (f'Mean test f1: {f1.compute().item()}')
    file.write(f'Mean test f1: {f1.compute().item()}\n')
    print (f'Class mean test f1: {f1_macro.compute().item()}')
    file.write(f'Class mean test f1: {f1_macro.compute().item()}\n')
    print (f'Class weighted mean test f1: {f1_weighted.compute().item()}')
    file.write(f'Class weighted mean test f1: {f1_weighted.compute().item()}\n')
    f1_none_vals = f1_none.compute()
    print (f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}')
    file.write(f'Individual class test f1s: "0" - {f1_none_vals[0].item()}, "1" - {f1_none_vals[1].item()}\n')
    confuse_vals = confusion.compute()
    print (f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}')
    file.write(f'Confusion matrix ("true_pred"): "0_0" - {int(confuse_vals[0][0].item())}, "0_1" - {int(confuse_vals[0][1].item())}, "1_0" - {int(confuse_vals[1][0].item())}, "1_1" - {int(confuse_vals[1][1].item())}\n')

    file.write(f'All patient results: {str(pat_results)}\n')
    file.close()
