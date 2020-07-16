"""
[Title] main.py
[Description] The main file to run the unsupervised models.
[Author] Lek'Sai Ye, University of Chicago
[Example Commands]
(Unsupervised rec)
> python training_gaussian.py -nt gaussian3d_rec -op rec_unsupervised
(Unsupervised one class)
> python training_gaussian.py -nt gaussian3d_one_class -op one_class_unsupervised
(Superivsed rec)
> python training_gaussian.py -am 1_1_1 -nt gaussian3d_rec -op rec
(Supervised one class)
> python training_gaussian.py -am 1_1_1 -nt gaussian3d_one_class -op one_class
"""

#############################################
# 0. Preparation
#############################################
import sys
sys.path.append('../dataset/')
sys.path.append('../network/')
sys.path.append('../model/')

import os
import glob
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from utils import *
from main_loading import *
from main_network import *
from main_model_rec import *
from main_model_one_class import *
from sklearn.metrics import roc_auc_score


# Initialize the parser
parser = argparse.ArgumentParser()
parser.add_argument('--random_state', type=int, default=42)

# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='gaussian3d',
                    help='[Choice]: fmnist, kmnist, cifar10')
parser.add_argument('-le', '--loader_eval_name', type=str, default='gaussian3d_eval',
                    help='fmnist_eval, kmnist, cifar10_eval')
parser.add_argument('-nm', '--normal_mu', type=str, default='1_-1_1')
parser.add_argument('-am', '--abnormal_mu', type=str, default='')
parser.add_argument('-am_', '--abnormal_mu_', type=str, default='1_1_1',
                    help='When you need to assign a value for abnormal mu in unsupervised setting.')
parser.add_argument('-ra', '--ratio_abnormal', type=float, default=0.1,
                    help='[Example]: 0.1')

# Arguments for main_network
parser.add_argument('-nt', '--net_name', type=str, default='gaussian3d_one_class',
                    help='[Choice]: gaussian3d_one_class, gaussian3d_rec')
parser.add_argument('-rp', '--rep_dim', type=int, default=2,
                    help='Only apply to DeepSAD model. Does not matter now.')

# Arguments for main_model
parser.add_argument('-pt', '--pretrain', type=int, default=1,
                    help='[Choice]: Only apply to DeepSAD model: 1 if True, 0 if False')
parser.add_argument('--load_model', type=str, default='',
                    help='[Example]: ./model.tar')
parser.add_argument('-op', '--optimizer_', type=str, default='one_class_unsupervised',
                    help='[Choice]: one_class, one_class_unsupervised, rec, rec_unsupervised')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--optimizer_name', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--ae_lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--ae_n_epochs', type=int, default=100)
parser.add_argument('--lr_milestones', type=str, default='50',
                    help='50_100_150')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.5e-6)
parser.add_argument('--ae_weight_decay', type=float, default=0.5e-3)
parser.add_argument('-gpu', '--device_no', type=int, default=1)
parser.add_argument('--n_jobs_dataloader', type=int, default=0)
parser.add_argument('--save_ae', type=bool, default=True,
                    help='Only apply to Deep SAD model.')
parser.add_argument('--load_ae', type=bool, default=False,
                    help='Only apply to Deep SAD model.')

# Arguments for output_paths
parser.add_argument('--txt_filename', type=str, default='full_results.txt')
p = parser.parse_args()

# Extract the arguments
random_state = p.random_state
loader_name, loader_eval_name = p.loader_name, p.loader_eval_name
normal_mu, abnormal_mu, abnormal_mu_ = p.normal_mu, p.abnormal_mu, p.abnormal_mu_
label_normal, label_abnormal = (0,), (1,)  # simply set the label for normal and abnormal data; consistent to the label set
ratio_abnormal = p.ratio_abnormal

net_name, rep_dim, pretrain, load_model = p.net_name, p.rep_dim, int(p.pretrain), p.load_model
optimizer_, eta_str, optimizer_name = p.optimizer_, p.eta_str, p.optimizer_name
ae_lr, lr, n_epochs, ae_n_epochs, batch_size = p.ae_lr, p.lr, p.n_epochs, p.ae_n_epochs, p.batch_size
lr_milestones = tuple(int(i) for i in p.lr_milestones.split('_'))
weight_decay, ae_weight_decay, device_no, n_jobs_dataloader = p.weight_decay, p.ae_weight_decay, p.device_no, p.n_jobs_dataloader
save_ae, load_ae = p.save_ae, p.load_ae
txt_filename = p.txt_filename

# Define folder to save the model and relating results
if optimizer_ in ['one_class', 'one_class_unsupervised']:
    folder_name = '{}_[{}]_[{}]_[{}]_[{}]'.format(optimizer_, p.normal_mu, p.abnormal_mu, str(pretrain), str(ratio_abnormal))
    out_path = './result/one_class/{}'.format(folder_name)
    final_path = '{}/net_{}_eta_{}_epochs_{}_batch_{}'.format(out_path, net_name, eta_str,
                                                              n_epochs, batch_size)
elif optimizer_ in ['rec', 'rec_unsupervised']:
    folder_name = '{}_[{}]_[{}]_[{}]_[{}]'.format(optimizer_, p.normal_mu, p.abnormal_mu, str(pretrain), str(ratio_abnormal))
    out_path = './result/rec/{}'.format(folder_name)
    final_path = '{}/net_{}_eta_{}_epochs_{}_batch_{}'.format(out_path, net_name, eta_str,
                                                              n_epochs, batch_size)

if not os.path.exists(out_path): os.makedirs(out_path)
if not os.path.exists(final_path): os.makedirs(final_path)

# Define the path for others
txt_result_file = Path(final_path) / txt_filename
log_path = Path(final_path) / 'training.log'
model_path = Path(final_path) / 'model.tar'
results_path = Path(final_path) / 'results.json'
ae_results_path = Path(final_path) / 'ae_results.json'
result_df_path = Path(final_path) / 'result_df.pkl'
cut_90_path = Path(final_path) / 'cut_90.npy'
cut_95_path = Path(final_path) / 'cut_95.npy'
cut_99_path = Path(final_path) / 'cut_99.npy'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
print(final_path)

# Define additional stuffs
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)

# Set random state
torch.manual_seed(random_state)

#############################################
# 1. Model Training
#############################################
# Initialize data
dataset = load_dataset(loader_name=loader_name,
                       normal_mu_train=normal_mu,
                       abnormal_mu_train=abnormal_mu,
                       ratio_abnormal=ratio_abnormal)

# Load Deep SAD model
if optimizer_ in ['one_class', 'one_class_unsupervised']:
    # Define model
    model = OneClassModel(optimizer_, eta)
    model.set_network(net_name)

    # Load other models if specified
    if load_model:
        logger.info('Loading model from {}'.format(load_model))
        model.load_model(model_path=load_model,
                         load_ae=True,
                         map_location=device)
    # Pretrain if specified
    if pretrain:
        logger.info('I am pre-training for you.')
        model.pretrain(dataset, optimizer_name, ae_lr, ae_n_epochs, lr_milestones,
                       batch_size, ae_weight_decay, device, n_jobs_dataloader)
        model.save_ae_results(export_json=ae_results_path)

# Load Reconstruction model
elif optimizer_ in ['rec', 'rec_unsupervised']:
    model = RecModel(optimizer_, eta)
    model.set_network(net_name)

# Training model
model.train(dataset, eta, optimizer_name, lr, n_epochs, lr_milestones,
            batch_size, weight_decay, device, n_jobs_dataloader, label_normal)


#############################################
# 2. Model Testing
#############################################
# Test and Save model
model.test(dataset, device, n_jobs_dataloader, label_normal)
model.save_results(export_json=results_path)
model.save_model(export_model=model_path, save_ae=save_ae)

# Prepare to write the results
indices_, labels_, scores_ = zip(*model.results['test_scores'])
indices_, labels_, scores_ = np.array(indices_), np.array(labels_), np.array(scores_)

result_df = pd.DataFrame()
result_df['indices'] = indices_
result_df['labels'] = labels_
result_df['scores'] = scores_
result_df.to_pickle(result_df_path)

result_df.drop('indices', inplace=True, axis=1)
df_normal = result_df[result_df.labels.isin(label_normal)]
df_abnormal = result_df[result_df.labels.isin(label_abnormal)]

# Save the threshold
cut_90 = df_normal.scores.quantile(0.90)
y_90 = [1 if e > cut_90 else 0 for e in df_abnormal['scores'].values]
np.save(cut_90_path, cut_90)

cut_95 = df_normal.scores.quantile(0.95)
y_95 = [1 if e > cut_95 else 0 for e in df_abnormal['scores'].values]
np.save(cut_95_path, cut_95)

cut_99 = df_normal.scores.quantile(0.99)
y_99 = [1 if e > cut_99 else 0 for e in df_abnormal['scores'].values]
np.save(cut_99_path, cut_99)

cut_90 = float(np.load(cut_90_path))
cut_95 = float(np.load(cut_95_path))
cut_99 = float(np.load(cut_99_path))


# Write the basic test file
f = open(txt_result_file, 'a')
f.write('############################################################\n')
f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f.write('\n[DataFrame Name] {}\n'.format(result_df_path))
f.write('[Normal Mean] {}\n'.format(normal_mu))
f.write('[Abnormal Mean] {}\n'.format(abnormal_mu))
f.write('[Model] {}\n'.format(optimizer_))
f.write('[Eta] {}\n'.format(eta))
f.write('[Epochs] {}\n'.format(n_epochs))
f.write('[Cut Threshold with 0.05 FP Rate] {}\n'.format(cut_95))
f.write('[Cut Threshold with 0.01 FP Rate] {}\n'.format(cut_99))
if len(df_abnormal):
    f.write('[A/N Ratio] 1:{}\n'.format(len(df_abnormal) / len(df_normal)))
    f.write('[Recall for {} (FP = 0.05)] {}\n'.format('TEST', sum(y_95) / len(y_95)))
    f.write('[Recall for {} (FP = 0.01)] {}\n'.format('TEST', sum(y_99) / len(y_99)))
f.write('---------------------\n')
f.close()
print('Done Training.')


#############################################
# 3. Model Evaluation
#############################################
f = open(txt_result_file, 'a')

mean_d1i_list = gen_ball(2.5, [int(i) for i in abnormal_mu_.split('_')])
mean_d0i_list = gen_ball(2.5, [int(i) for i in normal_mu.split('_')])

f.write('============================================================\n')
f.write('Recall when d1i changes (FPR=10%):\n')

for mean in mean_d1i_list:
    # Load dataset
    dataset_eval = load_dataset(loader_name=loader_eval_name,
                                abnormal_mu_test=mean,
                                normal_mu_train=normal_mu,
                                abnormal_mu_train=abnormal_mu,
                                ratio_abnormal=ratio_abnormal)

    # Load model
    if optimizer_ in ['one_class', 'one_class_unsupervised']:
        model_eval = OneClassModelEval(optimizer_, eta=eta)
    elif optimizer_ in ['rec', 'rec_unsupervised']:
        model_eval = RecModelEval(optimizer_, eta=eta)

    model_eval.set_network(net_name)
    model_eval.load_model(model_path=model_path, map_location=device)
    model_eval.test(dataset_eval, eta=eta, batch_size=batch_size, device=device,
                    n_jobs_dataloader=n_jobs_dataloader, label_normal=label_normal)
    _, labels, scores = zip(*model_eval.results['test_scores'])

    # Get predicted label
    y_90 = [1 if e > cut_90 else 0 for e in scores]
    recall_90 = sum(y_90) / len(y_90)
    d1i = ((np.array(mean) - [int(i) for i in normal_mu_.split('_')]) ** 2).sum()

    # Write results
    f.write('[d0i=2.5, d1i={}; recall={}]\n'.format(mean, d1i, recall_90))

f.write('============================================================\n')
f.write('Recall when d0i changes (FPR=10%):\n')
for mean in mean_d0i_list:
    # Load dataset
    dataset_eval = load_dataset(loader_name=loader_eval_name,
                                abnormal_mu_test=mean,
                                normal_mu_train=normal_mu,
                                abnormal_mu_train=abnormal_mu,
                                ratio_abnormal=ratio_abnormal)

    # Load model
    if optimizer_ in ['one_class', 'one_class_unsupervised']:
        model_eval = OneClassModelEval(optimizer_, eta=eta)
    elif optimizer_ in ['rec', 'rec_unsupervised']:
        model_eval = RecModelEval(optimizer_, eta=eta)

    model_eval.set_network(net_name)
    model_eval.load_model(model_path=model_path, map_location=device)
    model_eval.test(dataset_eval, eta=eta, batch_size=batch_size, device=device,
                    n_jobs_dataloader=n_jobs_dataloader, label_normal=label_normal)
    _, labels, scores = zip(*model_eval.results['test_scores'])

    # Get predicted label
    y_90 = [1 if e > cut_90 else 0 for e in scores]
    recall_90 = sum(y_90) / len(y_90)
    d0i = ((np.array(mean) - [int(i) for i in abnormal_mu_.split('_')]) ** 2).sum()

    # Write results
    f.write('[d0i={}; d1i=2.5; recall={}]\n'.format(mean, d0i, recall_90))

f.write('###########################################################\n\n\n\n')
f.close()
print('Finished. Now I am going to bed. Bye.')
