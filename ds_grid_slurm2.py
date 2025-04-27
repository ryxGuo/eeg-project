import numpy as np
import os
import argparse
import pickle

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
parser.add_argument('--runs', type=int, default=3)

# flow hyperparameters
parser.add_argument('--n_coupling_layers', type=int,   default=6,
                    help='Number of RealNVP coupling layers')
parser.add_argument('--flow_lr',            type=float, default=0.01,
                    help='Learning rate for the RealNVP optimizer')
parser.add_argument('--flow_wd',            type=float, default=0.01,
                    help='Weight decay for the RealNVP optimizer')

# density‑softmax fusion knobs
parser.add_argument('--fusion_tau',         type=float, default=1.0,
                    help='Exponent on the likelihood term')
parser.add_argument('--fusion_lambda',      type=float, default=0.0,
                    help='Additive weight on log‑likelihood')
parser.add_argument('--norm_pct',           type=float, default=1.0,
                    help='Which quantile to use for likelihood normalization (1.0=max)')
parser.add_argument('--softmax_temp',       type=float, default=1.0,
                    help='Temperature to divide scaled logits before Softmax')


parser.add_argument('--coupling_hidden_dim', type=int, default=64,
                    help='hidden‐layer size for each coupling MLP')
parser.add_argument('--pre_train_epochs', type=int, default=None,
                    help='if set, overrides 0.75 * num_epochs for the pre‐train phase')





args = parser.parse_args()

print('Args:')
for k, v in sorted(vars(args).items()):
  print("\t{}: {}".format(k, v))

# Set CUDA
# if args.cuda:
#     torch.cuda.set_device(args.gpu)
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

    train_dataset = EEGDataset(data_dict=data, locs=train_locs, study=args.study, treat=args.treat, norms=n)
    test_dataset = EEGDataset(data_dict=data, locs=test_locs, study=args.study, treat=args.treat, norms=n)

    # Get Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Get Model
    model = CNN(num_channels=len(channels), contig_len=args.contig_len, out_dim=args.num_classes, conf=conf[1])

    if args.cuda:
        model = model.cuda()
    print (model)

    optimizer = optim.Adam(model.parameters(), args.lr)
    if args.pre_train_epochs is not None:
        pre_train_epochs = args.pre_train_epochs
    else:
        pre_train_epochs = int(args.num_epochs * 0.75)

    # pre_train_epochs = int (args.num_epochs * 0.75)
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

    # flow_model = RealNVP(num_coupling_layers=args.n_coupling_layers, input_dim=train_latents.shape[1]).cuda()
    flow_model = RealNVP(num_coupling_layers=args.n_coupling_layers,input_dim=train_latents.shape[1],hidden_dim=args.coupling_hidden_dim).cuda()

    flow_optimizer = optim.Adam(flow_model.parameters(), lr=args.flow_lr, weight_decay=args.flow_wd)

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

    train_likelihood = torch.exp(flow_model.score_samples(train_latents)).unsqueeze(1)
    train_norm = torch.quantile(train_likelihood, args.norm_pct)

    preds = None
    predictions = None
    targets = None

    acc = 0
    tot = 0
    print ('Evaluating on test data...')
    for X, y in tqdm(test_dataloader):
        hidden = model.conv_layers(X).view(X.shape[0], -1)
        likelihood = torch.exp(flow_model.score_samples(hidden)).unsqueeze(1) / train_norm
        logits = model.fc_layers(hidden) * likelihood.float()
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

    print ('test acc:', acc/tot)
    file.write(f'test acc: {str((acc/tot).item())}\n')
    flow_acc = (acc/tot).item()
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
            hidden = model.conv_layers(X).view(X.shape[0], -1)
            logits = model.fc_layers(hidden)
            likelihood = torch.exp(flow_model.score_samples(hidden)).unsqueeze(1) / train_norm
            fused = (logits * (likelihood ** args.fusion_tau) + args.fusion_lambda * torch.log(likelihood + 1e-8))
            scaled_logits = fused / args.softmax_temp

            loss_value = nll(scaled_logits, y)

            acc += mean_accuracy(scaled_logits, y, reduce='mean')

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
        hidden = model.conv_layers(X).view(X.shape[0], -1)
        likelihood = torch.exp(flow_model.score_samples(hidden)).unsqueeze(1) / train_norm
        # logits = model.fc_layers(hidden) * likelihood.float()
        base_logits = model.fc_layers(hidden)
        fused       = base_logits * (likelihood**args.fusion_tau) \
                    + args.fusion_lambda * torch.log(likelihood + 1e-8)
        scaled_logits = fused / args.softmax_temp
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
        acc += mean_accuracy(scaled_logits, y)
        tot += logits.shape[0]

    print ('test acc:', acc/tot)
    file.write(f'test acc: {str((acc/tot).item())}\n')
    final_acc = (acc/tot).item()


    summary_path = os.path.join(
        args.results_dir + '_' + args.rm_ch_str,
        'hyperparam_results2.csv'
    )

    if run == 0 and not os.path.exists(summary_path):
        with open(summary_path, 'w') as sf:
            sf.write('run,pre_train_epochs,n_coupling_layers,coupling_hidden_dim,flow_lr,flow_wd,flow_test_acc,ds_test_acc\n')
            
    with open(summary_path, 'a') as sf:
        sf.write(
            f"{run},"
            f"{args.pre_train_epochs},"
            f"{args.n_coupling_layers},"
            f"{args.coupling_hidden_dim},"
            f"{args.flow_lr},"
            f"{args.flow_wd},"
            f"{flow_acc:.4f},"
            f"{final_acc:.4f}\n"
        )

        

    pickle.dump(preds, open(f'ds_after_preds.pkl', 'wb'))
    pickle.dump(predictions, open(f'ds_after_predictions.pkl', 'wb'))
    pickle.dump(targets, open(f'ds_after_targets.pkl', 'wb'))

    torch.save(model.state_dict(), f'{args.weights_dir}_{args.rm_ch_str}/{args.study}_{args.treat}_{args.contig_len}_{args.norm_type}_{args.balanced}_{run}_ds_after.pth')