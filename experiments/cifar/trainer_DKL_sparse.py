import numpy as np
from tqdm import trange
import argparse
import logging
from pathlib import Path
import math
import json

import torch
import torch.nn as nn
import gpytorch
from gpytorch.likelihoods import SoftmaxLikelihood
from GDKL.GP.models import GPModel, DKLModel, _init_IP
import GDKL.nn.networks as networks
from experiments.cifar.DataClass import CIFARData
from experiments.utils import get_device, set_seed, save_experiment, set_logger, common_parser, \
    detach_to_numpy, str2bool, topk
from experiments.calibrate import ECELoss


parser = argparse.ArgumentParser(description="CIFAR - DKL Sparse", parents=[common_parser])

#############################
#       Dataset Args        #
#############################
parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100'],
                    default="cifar10", help="Number of training samples")
parser.add_argument("--data-path", type=str, default="./dataset", help="Number of training samples")
parser.add_argument("--val-pct", type=float, default=0.0,
                    help="allocation to validation set from the full training set of 50K examples")
parser.add_argument("--train-pct", type=float, default=1.0,
                    help="subsample from training set after allocation to validation")
parser.add_argument("--augment-train", type=str2bool, default=True)
parser.add_argument("--resize-size", type=int, default=32)

##############################
#       GP Model args        #
##############################
parser.add_argument("--mean-function", type=str, default='constant',
                    choices=['zero', 'constant'],
                    help="kernel function")
parser.add_argument("--kernel-function", type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'RQKernel', 'SpectralMixtureKernel'],
                    help="kernel function")
parser.add_argument('--normalize-gp-input', type=str, default='none',
                    choices=['scale', 'norm', 'none'],
                    help='type of normalization of GP input')
parser.add_argument("--ARD", type=str2bool, default=False,
                    help="used ARD")
parser.add_argument('--noise-factor', type=float,
                    default=-4.05,
                    help='the noise constant in the first layer')
parser.add_argument("--num-train-samples", type=int, default=16)
parser.add_argument("--num-test-samples", type=int, default=320)
parser.add_argument("--learn-ip-loc", type=str2bool, default=True,
                    help="learn inducing inputs locations")
parser.add_argument("--num-ip", type=int, default=10, help="number of inducing points")
parser.add_argument("--beta", type=float, default=1.0, help="coef of data kl")

##############################
#     Optimization args      #
##############################
parser.add_argument("--num-epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--test-batch-size", type=int, default=512)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
parser.add_argument("--kernel-lr", type=float, default=1e-1, help="learning rate of kernel params")
parser.add_argument("--var-lr", type=float, default=1e-1, help="learning rate of variational params")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='100, 150')
parser.add_argument("--widen-factor", type=int, default=5, help="widden factor in WRN")
parser.add_argument("--DUE", type=str2bool, default=False,
                    help="learn \sigma_n?")
parser.add_argument("--coeff", type=float, default=3., help="Spectral normalization coefficient")
parser.add_argument(
        "--n-power-iterations", default=1, type=int, help="Number of power iterations"
    )
parser.add_argument('--color', type=str, default='sienna',
                    choices=['darkblue', 'maroon', 'sienna', 'darkgoldenrod', 'darkgreen', 'purple'],
                    help='color for calibration plot')
parser.add_argument('--save-dir', type=str2bool, default=True)

args = parser.parse_args()
if args.DUE:
    args.color = 'darkgoldenrod'

set_logger()

set_seed(args.seed)
device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

exp_name = f'DKL-CIFAR10_sparse_DUE_{args.DUE}_{args.dataset}_seed_{args.seed}_beta_{args.beta}_' \
           f'ARD_{args.ARD}_kernel_{args.kernel_function}_{args.mean_function}_' \
           f'net-conf_{args.widen_factor}_num-ip_{args.num_ip}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)
ECE_module = ECELoss()

################################
# get data
################################
cifar10data = CIFARData(args.dataset, args.data_path)
train_loader, ordered_train_loader, val_loader, test_loader = \
    cifar10data.get_loaders(val_pct=args.val_pct, train_pct=args.train_pct, return_index=True, shuffle_train=True,
                            batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                            augment_train=args.augment_train, resize_size=args.resize_size,
                            num_workers=args.num_workers)

num_classes = 10 if args.dataset == 'cifar10' else 100

###############################
# Model
###############################
network = networks.WideResNet(widen_factor=args.widen_factor,
                              spectral_bn=args.DUE,
                              spectral_conv=args.DUE,
                              dropout_rate=args.dropout,
                              coeff=args.coeff,
                              n_power_iterations=args.n_power_iterations).to(device)

likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

@torch.no_grad()
def get_data(loader):
    for k, batch in enumerate(loader):
        data, labels, indices = tuple(t.to(device) for t in batch)
        feats = network(data)
        X = data if k == 0 else torch.cat((X, data))
        X_f = feats if k == 0 else torch.cat((X_f, feats))
        Y = labels if k == 0 else torch.cat((Y, labels))
        I = indices if k == 0 else torch.cat((I, indices))
    return X, X_f, Y, I

# set ARD
ARD_dim = None
if args.ARD:
    ARD_dim = 64 * args.widen_factor

train_x, train_feat, train_y, train_indices = get_data(ordered_train_loader)  # pass through transforms
initial_lengthscale = 1.
initial_inducing_locs = _init_IP(train_feat, args.num_ip)
gp_layer = GPModel(inducing_locs=initial_inducing_locs, num_outputs=num_classes, kernel_function=args.kernel_function,
                   data_dim=ARD_dim, initial_lengthscale=initial_lengthscale,
                   mean_function=args.mean_function).to(device)
gp_model = DKLModel(network, gp_layer, normalize_gp_input=args.normalize_gp_input).to(device)

###############################
# Optimizer
###############################
params = [{'params': gp_model.feature_extractor.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
          {'params': gp_model.gp_layer.mean_module.parameters(), 'lr': args.kernel_lr},
          {'params': gp_model.gp_layer.covar_module.parameters(), 'lr': args.kernel_lr},
          {'params': gp_model.gp_layer.variational_parameters(), 'lr': args.var_lr},
          {'params': likelihood.parameters(), 'lr': args.lr},
          ]

optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0.0, momentum=0.9, nesterov=False) \
        if args.optimizer == 'sgd' else torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

criteria = nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(dim=-1)

def log_calibration(lbls_vs_target, file_name):
    lbls_preds = torch.tensor(lbls_vs_target)
    probs = lbls_preds[:, 1:]
    targets = lbls_preds[:, 0]

    ece_metrics = ECE_module.forward(probs, targets, (out_dir / file_name).as_posix(), color=args.color)
    logging.info(f"{file_name}, "
                 f"ECE: {ece_metrics[0].item():.3f}, "
                 f"MCE: {ece_metrics[1].item():.3f}, "
                 f"BRI: {ece_metrics[2].item():.3f}")


###############################
# Evaluate
###############################
@torch.no_grad()
def model_evalution(loader):
    gp_model.eval()
    likelihood.eval()

    targets = []
    preds = []
    nll_accum = num_samples = 0

    with gpytorch.settings.num_likelihood_samples(args.num_test_samples):
        for k, batch in enumerate(loader):
            batch = (t.to(device) for t in batch)
            test_data, test_labels = batch

            # the following samples from the joint distribution:
            output = gp_model(test_data)  # returns diagonal covariance matrix
            marginal_ll = likelihood(output)  # dist. over samples of f_* : p(y_* | f_*)
            sample_logits = marginal_ll.logits  # [num_test_samples, N, num_classes]

            # log p_i = log \sum_j exp(f_ij - log \sum_k f_kj) - log |J|
            denominator = torch.logsumexp(sample_logits, dim=-1, keepdim=True)
            exponent = sample_logits - denominator
            logits = torch.logsumexp(exponent, dim=0) - math.log(args.num_test_samples)
            probs = softmax(logits)

            nll_accum += criteria(logits, test_labels) * test_labels.shape[0]
            num_samples += test_data.size(0)

            targets.append(test_labels)
            preds.append(probs)

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    pred_top1 = topk(target, full_pred, 1)
    nll_accum /= num_samples
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return pred_top1, nll_accum, labels_vs_preds


mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model.gp_layer, num_data=len(train_loader.dataset), beta=args.beta)
epoch_iter = trange(args.num_epochs)

gp_model.train()
likelihood.train()

for epoch in epoch_iter:

    cumm_loss = num_samples = num_batches = 0

    with gpytorch.settings.num_likelihood_samples(args.num_train_samples):
        for k, batch in enumerate(train_loader):

            batch = (t.to(device) for t in batch)
            train_data, train_labels, train_idx = batch

            optimizer.zero_grad()
            output = gp_model(train_data)  # returns q(f)
            loss = - mll(output, train_labels)

            loss.backward()
            optimizer.step()

            epoch_iter.set_description(f'[{epoch}][{k}] Training loss {loss.item():.5f}')
            cumm_loss += loss.item() * train_labels.shape[0]
            num_samples += train_labels.shape[0]
            num_batches += 1

    cumm_loss /= num_samples
    scheduler.step()

val_acc = val_nll = 0
if val_loader is not None:
    val_acc, val_nll, val_labels_vs_preds = model_evalution(val_loader)
    print(f"Val - NLL: {val_nll:.5f}, Accuracy: {val_acc:.5f}")
test_acc, test_nll, test_labels_vs_preds = model_evalution(test_loader)
print(f"Test - NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")


log_calibration(test_labels_vs_preds, 'calibration_test.png')

results = dict()
results['final_test_results'] = {
    'test_labels_vs_preds': test_labels_vs_preds.tolist()
}

results_file = "results.json"
with open(str(out_dir / results_file), "w") as file:
    json.dump(results, file, indent=4)

