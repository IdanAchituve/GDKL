import math
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import trange
from datetime import datetime

import torch
import torch.nn as nn
import gpytorch
from GDKL.GP.models import MultiClassGPModelNeuralTangents, GPModelExactPredictiveMultiClass
from gpytorch.likelihoods import DirichletClassificationLikelihood
import GDKL.nn.networks as networks
from experiments.cifar.DataClass import CIFARData
from experiments.utils import get_device, set_seed, save_experiment, set_logger, common_parser, detach_to_numpy, \
    str2bool, univariate_gaussian_entropy, univariate_gaussian_cross_entropy, topk
from experiments.calibrate import ECELoss
from neural_tangents import stax
import argparse

parser = argparse.ArgumentParser(description="CIFAR10 - GDKL", parents=[common_parser])

#############################
#       Dataset Args        #
#############################
parser.add_argument("--data-path", type=str, default="./datasets", help="Number of training samples")
parser.add_argument("--val-pct", type=float, default=0.0,
                    help="allocation to validation set from the full training set of 50K examples")
parser.add_argument("--train-pct", type=float, default=0.008,
                    help="subsample from training set after allocation to validation")
parser.add_argument("--augment-train", type=str2bool, default=False)
parser.add_argument("--resize-size", type=int, default=32)

##############################
#       GP Model args        #
##############################
parser.add_argument("--mean-function", type=str, default='zero',
                    choices=['zero', 'constant'],
                    help="kernel function")
parser.add_argument("--kernel-function", type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'RQKernel'],
                    help="kernel function")
parser.add_argument('--normalize-gp-input', type=str, default='none',
                    choices=['scale', 'norm', 'none'],
                    help='type of normalization of GP input')
parser.add_argument("--ARD", type=str2bool, default=False,
                    help="used ARD")
parser.add_argument("--weight-var", type=float, default=5.0)
parser.add_argument("--bias-var", type=float, default=0.2)
parser.add_argument("--inf-widen-factor", type=int, default=5,
                    help="widen factor in infinite WRN")
parser.add_argument("--beta", type=float, default=1.0, help="coef of data kl")
parser.add_argument("--neural-activation", type=str, default='Relu',
                    choices=['Relu', 'Sigmoid_like', 'Rbf', 'Sign'],
                    help="infinite network activation")
parser.add_argument('--noise-factor', type=float,
                    default=-4.05,
                    help='the noise constant in the first layer')
parser.add_argument("--normalize", type=str2bool, default=False,
                    help="normalize input to inf network")

##############################
#       NN Model args        #
##############################
parser.add_argument("--widen-factor", type=int, default=5, help="widen factor in WRN")
parser.add_argument("--train-ratio", type=float, default=0.5,
                    help='ratio of examples to allocate for training the posterior')
parser.add_argument("--num-test-samples", type=int, default=1024)
parser.add_argument("--num-train-samples", type=int, default=256)

##############################
#     Optimization args      #
##############################
parser.add_argument("--num-pretrain-epochs", type=int, default=1000)
parser.add_argument("--num-epochs", type=int, default=6000)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--test-batch-size", type=int, default=1024)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--kernel-lr", type=float, default=1e-2, help="learning rate of kernel params")
parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='3600,4800')

parser.add_argument("--eval-on-prior-model", type=str2bool, default=False,
                    help="evaluate performance on the training set?")
parser.add_argument('--color', type=str, default='darkblue',
                    choices=['darkblue', 'maroon', 'sienna', 'darkgoldenrod', 'darkgreen', 'purple'],
                    help='color for calibration plot')
parser.add_argument('--save-dir', type=str2bool, default=True)

args = parser.parse_args()

set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

exp_name = f'GDKL-CIFAR10_seed_{args.seed}_train-pct_{args.train_pct}_{args.val_pct}_' \
           f'beta_{args.beta}_kernel_{args.kernel_function}_ARD_{args.ARD}_net-conf_{args.widen_factor}_' \
           f'{args.neural_activation}_{args.weight_var}_{args.bias_var}_{args.inf_widen_factor}'

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
cifar10data = CIFARData('cifar10', args.data_path)
train_loader, val_loader, test_loader = \
    cifar10data.get_loaders(val_pct=args.val_pct, train_pct=args.train_pct, shuffle_train=False,
                            batch_size=args.batch_size, test_batch_size=args.test_batch_size,
                            augment_train=args.augment_train, resize_size=args.resize_size,
                            num_workers=args.num_workers)
num_classes = 10
num_data = len(train_loader.dataset.indices)
train_x, train_y = next(iter(train_loader))
train_x = train_x.to(device)  # [B, C, D, D]
train_y = train_y.to(device)

###############################
# Finite Network
###############################
network = networks.WideResNet(widen_factor=args.widen_factor, dropout_rate=args.dropout)

###############################
# Infinite Network
###############################
inf_likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True).to(device)
inf_likelihood.second_noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor) for _ in range(num_classes)],
                                                                device=device))

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
    Main = stax.serial(
      getattr(stax, args.neural_activation)(),
      stax.Conv(channels, (3, 3), strides, padding='SAME',
                W_std=np.sqrt(args.weight_var), b_std=np.sqrt(args.bias_var)),
      getattr(stax, args.neural_activation)(),
      stax.Conv(channels, (3, 3), padding='SAME',
                W_std=np.sqrt(args.weight_var), b_std=np.sqrt(args.bias_var))
    )
    Shortcut = stax.Identity() if not channel_mismatch else \
               stax.Conv(channels, (3, 3), strides, padding='SAME',
                         W_std=np.sqrt(args.weight_var), b_std=np.sqrt(args.bias_var))

    return stax.serial(stax.FanOut(2),
                     stax.parallel(Main, Shortcut),
                     stax.FanInSum())


def WideResnetGroup(n, channels, strides=(1, 1)):
    blocks = []
    blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
    for _ in range(n - 1):
        blocks += [WideResnetBlock(channels, (1, 1))]
    return stax.serial(*blocks)


def WideResnet(block_size, k, num_classes):
    return stax.serial(
      stax.Conv(16, (3, 3), padding='SAME', W_std=np.sqrt(args.weight_var), b_std=np.sqrt(args.bias_var)),
      WideResnetGroup(block_size, int(16 * k)),
      WideResnetGroup(block_size, int(32 * k), (2, 2)),
      WideResnetGroup(block_size, int(64 * k), (2, 2)),
      #stax.AvgPool((8, 8)),
      #stax.GlobalAvgPool(),
      stax.Flatten(),
      stax.Dense(num_classes, W_std=1., b_std=0.))

init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=args.inf_widen_factor, num_classes=num_classes)

inf_gp_model = MultiClassGPModelNeuralTangents(train_x=train_x.view(num_data, -1),
                                               train_y=inf_likelihood.transformed_targets,
                                               likelihood=inf_likelihood, neural_kernel_fun=kernel_fn,
                                               num_outputs=num_classes,
                                               input_dim=train_x[0, ...].shape, mean_func=args.mean_function,
                                               kernel_type='nngp', normalize=args.normalize).to(device)

inf_gp_model.ker_fun.forward_set_kernel(train_x)
trainset_ind = torch.arange(train_y.shape[0]).to(device)

params = [{'params': inf_gp_model.parameters()}]
optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0.0, momentum=0.9, nesterov=False) \
        if args.optimizer == 'sgd' else torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
epoch_iter = trange(args.num_pretrain_epochs)
criteria = nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(dim=-1)

# ================
# Evaluate
# ================
def log_calibration(lbls_vs_target, file_name):
    lbls_preds = torch.tensor(lbls_vs_target)
    probs = lbls_preds[:, 1:]
    targets = lbls_preds[:, 0]

    ece_metrics = ECE_module.forward(probs, targets, (out_dir / file_name).as_posix(), color=args.color)
    logging.info(f"{file_name}, "
                 f"ECE: {ece_metrics[0].item():.3f}, "
                 f"MCE: {ece_metrics[1].item():.3f}, "
                 f"BRI: {ece_metrics[2].item():.3f}")


@torch.no_grad()
def model_evalution(loader, gp_model):
    gp_model.eval()

    targets = []
    preds = []
    nll_accum = num_samples = 0

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} - Accumulated nll: {nll_accum:.5f}, Num Samples: {num_samples}")

    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        test_data, test_labels = batch
        num_test_data = test_data.shape[0]

        with gpytorch.settings.lazily_evaluate_kernels(state=not isinstance(gp_model, MultiClassGPModelNeuralTangents)):
            post_pred_dist = gp_model(test_data.view(num_test_data, -1), **{'phase': 'test_cache'})

        mean_pred = post_pred_dist.mean
        var_pred = post_pred_dist.variance
        pred_dist = torch.distributions.Normal(loc=mean_pred, scale=var_pred.clamp_min(1e-8).sqrt())
        # logits: f_ij = [Samples, N, C]
        pred_samples = pred_dist.sample(torch.Size((args.num_test_samples,))).permute(0, 2, 1)

        # log p_i = log \sum_j exp(f_ij - log \sum_k f_kj) - log |J|
        denominator = torch.logsumexp(pred_samples, dim=-1, keepdim=True)
        exponent = pred_samples - denominator
        logits = torch.logsumexp(exponent, dim=0) - math.log(args.num_test_samples)
        nll_accum += criteria(logits, test_labels) * test_labels.shape[0]
        probs = softmax(logits)
        num_samples += test_data.size(0)

        targets.append(test_labels)
        preds.append(probs)

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"{dt_string} - Accumulated nll: {nll_accum:.5f}, Num Samples: {num_samples}")

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    pred_top1 = topk(target, full_pred, 1)
    nll_accum /= num_samples
    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)

    return pred_top1, nll_accum, labels_vs_preds

inf_mll = gpytorch.mlls.ExactMarginalLogLikelihood(inf_likelihood, inf_gp_model)
inf_likelihood.train()
inf_gp_model.train()

for epoch in epoch_iter:

    optimizer.zero_grad()
    output = inf_gp_model(train_x.view(num_data, -1), **{'phase': 'train'})
    loss = - inf_mll(output, inf_likelihood.transformed_targets).sum()

    loss.backward()
    optimizer.step()

    to_print = f"Train loss: {loss.item():.5f}\n" \
               f"noise:{np.round(inf_gp_model.likelihood.second_noise_covar.noise.squeeze().detach().cpu().numpy(), 5)}\n"
    to_print += f"outputscale:{np.round(inf_gp_model.covar_module.outputscale.detach().cpu().numpy(), 5)}\n"
    print(to_print)

if args.eval_on_prior_model:
    train_acc = train_nll = 0
    test_acc, test_nll, test_labels_vs_preds = model_evalution(test_loader, inf_gp_model)
    logging.info(f"Test - NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")

###############################
# Model
###############################
likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True).to(device)
likelihood.second_noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor) for _ in range(num_classes)],
                                                            device=device))
# set ARD
ARD_dim = None
if args.ARD:
    ARD_dim = 64 * args.widen_factor

gp_model = GPModelExactPredictiveMultiClass(train_x=train_x.view(num_data, -1), train_y=likelihood.transformed_targets,
                                            likelihood=likelihood, feature_extractor=network, num_outputs=num_classes,
                                            input_dim=train_x[0, ...].shape, mean_func=args.mean_function,
                                            normalize_gp_input=args.normalize_gp_input,
                                            kernel_function=args.kernel_function,
                                            data_dim=ARD_dim).to(device)

###############################
# Optimizer
###############################
params = [{'params': gp_model.feature_extractor.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
          {'params': gp_model.covar_module.parameters(), 'lr': args.kernel_lr},
          {'params': gp_model.mean_module.parameters(), 'lr': args.kernel_lr},
          {'params': gp_model.likelihood.parameters(), 'lr': args.kernel_lr},
          ]

optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0.0, momentum=0.9, nesterov=False) \
        if args.optimizer == 'sgd' else torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

epoch_iter = trange(args.num_epochs)
num_inner_steps = 0
for epoch in epoch_iter:
    gp_model.train()
    inf_gp_model.eval()

    optimizer.zero_grad()

    num_samples = int(num_data * args.train_ratio)  # number of samples to take from training data
    X_train_ind = torch.sort(torch.argsort(torch.rand(num_data, device=train_x.device))[:num_samples])[0]
    X_test_ind = torch.as_tensor([i for i in range(num_data) if i not in X_train_ind], device=X_train_ind.device)

    X_train, y_train = train_x[X_train_ind, ...], train_y[X_train_ind, ...]
    X_test, y_test = train_x[X_test_ind, ...], train_y[X_test_ind, ...]
    transformed_y_train = likelihood.transformed_targets[:, X_train_ind]
    transformed_y_test = likelihood.transformed_targets[:, X_test_ind]

    # obtain prior distribution
    inf_gp_model.set_train_data(X_train.view(num_samples, -1), transformed_y_train, strict=False)
    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        prior_dist = inf_gp_model(X_test.view(num_data - num_samples, -1), **{'phase': 'train2',
                                                                              'train_indices': X_train_ind,
                                                                              'test_indices': X_test_ind})
    # obtain posterior distribution
    X = torch.cat((X_train, X_test))
    Y = torch.cat((transformed_y_train, transformed_y_test), dim=1)
    gp_model.set_train_data(X.view(num_data, -1), Y, strict=False)
    # the following enforce taking gradients w.r.t the parameters from the predictive distribution
    # in exact_prediction_strategies L231
    with gpytorch.settings.detach_test_caches(False):
        posterior_dist = gp_model.forward_predictive(X.view(num_data, -1), Y, num_train=X_train.shape[0])

    post_dist_mean = posterior_dist.mean
    post_dist_var = posterior_dist.variance.clamp_min(1e-8)
    pred_dist = torch.distributions.Normal(loc=post_dist_mean, scale=post_dist_var.sqrt())
    pred_samples = pred_dist.rsample(torch.Size((args.num_train_samples,))).permute(0, 2, 1)

    denominator = torch.logsumexp(pred_samples, dim=-1, keepdim=True)
    exponent = pred_samples - denominator
    logits = torch.logsumexp(exponent, dim=0) - math.log(args.num_train_samples)
    ELL = - criteria(logits, y_test)

    entropy_term = univariate_gaussian_entropy(post_dist_var, 1)
    ce_trem = univariate_gaussian_cross_entropy(prior_dist.mean, prior_dist.variance.clamp_min(1e-8),
                                                post_dist_mean, post_dist_var, 1)
    KL = (entropy_term - ce_trem).mean(0).mean()

    loss = - ELL + args.beta * KL
    loss.backward()
    optimizer.step()

    to_print = f"Train loss: {loss.item():.5f}, " \
               f"ELL:{ELL.item():.5f}, " \
               f"KL: {KL.item():.5f}"

    net_norm = torch.cat([p.view(-1) for p in gp_model.feature_extractor.parameters()]).norm()
    to_print += f", net norm: {net_norm.item():.5f}\n"
    to_print += f", noise post: {np.round(gp_model.likelihood.second_noise_covar.noise.squeeze().detach().cpu().numpy(), 5)}\n"
    to_print += f", noise prior: {np.round(inf_gp_model.likelihood.second_noise_covar.noise.squeeze().detach().cpu().numpy(), 5)}\n"
    to_print += f", outputscale post: {np.round(gp_model.covar_module.outputscale.detach().cpu().numpy(), 5)}\n"
    to_print += f", outputscale prior: {np.round(inf_gp_model.covar_module.outputscale.detach().cpu().numpy(), 5)}"

    print(to_print)

    scheduler.step()

gp_model.set_train_data(train_x.view(num_data, -1), likelihood.transformed_targets, strict=False)

val_acc = val_nll = 0
if val_loader is not None:
    val_acc, val_nll, val_labels_vs_preds = model_evalution(val_loader, gp_model)
    print(f"Val - NLL: {val_nll:.5f}, Accuracy: {val_acc:.5f}")
test_acc, test_nll, test_labels_vs_preds = model_evalution(test_loader, gp_model)
print(f"Test - NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")

log_calibration(test_labels_vs_preds, 'calibration_test.png')

results = dict()
results['final_test_results'] = {
    'test_labels_vs_preds': test_labels_vs_preds.tolist()
}

results_file = "results.json"
with open(str(out_dir / results_file), "w") as file:
    json.dump(results, file, indent=4)


