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
from GDKL.GP.models import MultiClassGPModelNeuralTangents, GPModel, DKLModel, _init_IP
from gpytorch.likelihoods import DirichletClassificationLikelihood, SoftmaxLikelihood
import GDKL.nn.networks as networks
from experiments.cifar.DataClass import CIFARData
from experiments.utils import get_device, set_seed, save_experiment, set_logger, common_parser, detach_to_numpy, \
    str2bool, univariate_gaussian_entropy, univariate_gaussian_cross_entropy, topk, save_data, load_data
from experiments.calibrate import ECELoss
from sklearn.model_selection import train_test_split
from neural_tangents import stax
import argparse

parser = argparse.ArgumentParser(description="CIFAR10 - GDKL Sparse", parents=[common_parser])

#############################
#       Dataset Args        #
#############################
parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100'], default="cifar10")
parser.add_argument("--data-path", type=str, default="./datasets")
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
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'RQKernel'],
                    help="kernel function")
parser.add_argument('--normalize-gp-input', type=str, default='none',
                    choices=['scale', 'norm', 'none'],
                    help='type of normalization of GP input')
parser.add_argument("--ARD", type=str2bool, default=False,
                    help="used ARD")
parser.add_argument("--weight-var", type=float, default=5.0)
parser.add_argument("--bias-var", type=float, default=0.2)
parser.add_argument("--inf-widen-factor", type=int, default=5, help="widen factor in infinite WRN")
parser.add_argument("--subset-size-pct", type=float, default=0.05, help="subset size to train prior model hps (~5%)")
parser.add_argument("--beta", type=float, default=0.1, help="coef of data kl")
parser.add_argument("--neural-activation", type=str, default='Relu',
                    choices=['Relu', 'Sigmoid_like', 'Rbf', 'Sign'],
                    help="infinite network activation")
parser.add_argument('--noise-factor', type=float,
                    default=-4.05,
                    help='the noise constant in the first layer')
parser.add_argument("--normalize", type=str2bool, default=False,
                    help="normalize input to inf network")
parser.add_argument("--learn-ip-loc", type=str2bool, default=True,
                    help="learn inducing inputs locations")
parser.add_argument("--num-ip", type=int, default=10, help="number of inducing points")
parser.add_argument("--fragment-size", type=int, default=2500)
parser.add_argument('--multiplicative-factor', type=float,
                    default=1e-6,
                    help='multiplicative factor to scale the NNGP kernel to have an order of 100 at most. '
                         'For full cifar10 use 1e-6, and for full cifar100 1e-1')
parser.add_argument("--jitter-val", type=float, default=1e-3,
                    help="jitter to add to diagonal in infinite network; needed only for cifar100 a jitter of 1e-3")

##############################
#       NN Model args        #
##############################
parser.add_argument("--widen-factor", type=int, default=5, help="widden factor in WRN")
parser.add_argument("--train-ratio", type=float, default=0.5,
                    help='ratio of examples to allocate for training the posterior in each batch')
parser.add_argument("--num-train-samples", type=int, default=16)
parser.add_argument("--num-test-samples", type=int, default=320)

##############################
#     Optimization args      #
##############################
parser.add_argument("--num-pretrain-epochs", type=int, default=1000)
parser.add_argument("--num-epochs", type=int, default=194)  # 194 since 1000 gradient steps is ~ 5 epochs with bs=256
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--test-batch-size", type=int, default=512)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
parser.add_argument("--kernel-lr", type=float, default=1e-1, help="learning rate of kernel params")
parser.add_argument("--var-lr", type=float, default=1e-1, help="learning rate of variational params")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='97, 146')

parser.add_argument('--K-path', type=str, default=None, help='path to pre-computed infinite network kernel matrix')
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

exp_name = f'GDKL-CIFAR10_sparse_{args.dataset}_seed_{args.seed}_beta_{args.beta}_' \
           f'{args.neural_activation}_{args.weight_var}_{args.bias_var}_{args.inf_widen_factor}_ARD_{args.ARD}_' \
           f'kernel_{args.kernel_function}_{args.mean_function}_net-conf_{args.widen_factor}_' \
           f'num-ip_{args.num_ip}'

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

#############################################################################################
# Define the posterior GP first to be comparable to other methods in terms of initializations
#############################################################################################
network = networks.WideResNet(widen_factor=args.widen_factor, dropout_rate=args.dropout).to(device)
likelihood = SoftmaxLikelihood(num_classes=num_classes, mixing_weights=False)

@torch.no_grad()
def get_data(loader):
    for k, batch in enumerate(loader):
        data, labels, indices = tuple(t.to(device) for t in batch)
        feats = network(data)
        X = data.cpu() if k == 0 else torch.cat((X, data.cpu()))
        X_f = feats.cpu() if k == 0 else torch.cat((X_f, feats.cpu()))
        Y = labels.cpu() if k == 0 else torch.cat((Y, labels.cpu()))
        I = indices.cpu() if k == 0 else torch.cat((I, indices.cpu()))
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
# Infinite Network
###############################
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

num_data = train_indices.shape[0]
num_inf_samples = int(num_data * args.subset_size_pct)  # number of samples to take from training data
if num_inf_samples < num_data:
    inf_train_ind, _ = train_test_split(detach_to_numpy(train_indices),
                                          train_size=num_inf_samples, stratify=detach_to_numpy(train_y), random_state=42)
else:
    inf_train_ind = detach_to_numpy(train_indices)
inf_train_ind = torch.sort(torch.LongTensor(inf_train_ind))[0]
inf_train_x = torch.clone(train_x[inf_train_ind, ...]).contiguous().to(device)
inf_train_y = torch.clone(train_y[inf_train_ind, ...]).contiguous().to(device)

inf_likelihood_partial = DirichletClassificationLikelihood(inf_train_y, learn_additional_noise=True).to(device)
inf_likelihood_partial.second_noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor)
                                                                         for _ in range(num_classes)], device=device))
inf_transfomred_train_y = inf_likelihood_partial.transformed_targets
inf_gp_model_partial = MultiClassGPModelNeuralTangents(train_x=inf_train_x.view(num_inf_samples, -1),
                                                       train_y=inf_transfomred_train_y,
                                                       likelihood=inf_likelihood_partial, neural_kernel_fun=kernel_fn,
                                                       num_outputs=num_classes,
                                                       input_dim=inf_train_x[0, ...].shape,
                                                       kernel_type='nngp', normalize=args.normalize,
                                                       jitter_val=args.jitter_val).to(device)

if args.K_path is None:
    if train_x.shape[0] > args.fragment_size:
        K = inf_gp_model_partial.ker_fun.forward_set_kernel_fragments(train_x,
                                                                      args.fragment_size,
                                                                      args.multiplicative_factor)  # populate the full kernel matrix
        K_device = 'cpu'
    else:
        K = inf_gp_model_partial.ker_fun.forward_set_kernel(train_x,
                                                            args.multiplicative_factor)  # populate the full kernel matrix
        inf_train_ind = inf_train_ind.to(K.device)
        K_device = K.device

    save_data(K, file=out_dir / "K.pt")
else:
    K = args.multiplicative_factor * load_data(args.K_path)
    K_device = 'cpu' if train_x.shape[0] > args.fragment_size else device
    K.to(K_device)
    inf_gp_model_partial.ker_fun.set_K(K)
    print(f'max: {torch.max(K).item()}, min: {torch.min(K).item()}, Nans: {torch.any(torch.isnan(K)).item()}')

params = [{'params': inf_gp_model_partial.parameters()}]
optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0.0, momentum=0.9, nesterov=False) \
        if args.optimizer == 'sgd' else torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
epoch_iter = trange(args.num_pretrain_epochs)
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
def model_evalution(loader, gp_model):
    gp_model.eval()
    likelihood.eval()

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

        with gpytorch.settings.num_likelihood_samples(args.num_test_samples):
            if isinstance(gp_model, MultiClassGPModelNeuralTangents):
                with gpytorch.settings.lazily_evaluate_kernels(state=False):
                    post_pred_dist = gp_model(test_data.view(num_test_data, -1), **{'phase': 'test_cache'})

                mean_pred = post_pred_dist.mean
                var_pred = post_pred_dist.variance
                marginal_ll = torch.distributions.Normal(loc=mean_pred, scale=var_pred.clamp_min(1e-8).sqrt())
                # logits: f_ij = [Samples, N, C]
                sample_logits = marginal_ll.sample(torch.Size((args.num_test_samples,))).permute(0, 2, 1)
            else:
                output = gp_model(test_data)  # returns diagonal covariance matrix
                marginal_ll = likelihood(output)  # marginal dist. over samples of f_* : p(y_* | f_*)
                sample_logits = marginal_ll.logits

            # log p_i = log \sum_j exp(f_ij - log \sum_k f_kj) - log |J|
            denominator = torch.logsumexp(sample_logits, dim=-1, keepdim=True)
            exponent = sample_logits - denominator
            logits = torch.logsumexp(exponent, dim=0) - math.log(args.num_test_samples)
            nll_accum += criteria(logits, test_labels) * test_labels.shape[0]
            probs = softmax(logits)

        num_samples += num_test_data
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


inf_mll = gpytorch.mlls.ExactMarginalLogLikelihood(inf_likelihood_partial, inf_gp_model_partial)
inf_likelihood_partial.train()
inf_gp_model_partial.train()

for epoch in epoch_iter:

    optimizer.zero_grad()
    output = inf_gp_model_partial(inf_train_x.view(num_inf_samples, -1), **{'phase': 'train_idx',
                                                                            'train_indices': inf_train_ind})
    loss = - inf_mll(output, inf_transfomred_train_y).sum()

    loss.backward()
    optimizer.step()

    to_print = f"Train loss: {loss.item():.5f}\n" \
               f"noise:{np.round(inf_gp_model_partial.likelihood.second_noise_covar.noise.squeeze().detach().cpu().numpy(), 5)}\n"
    to_print += f"outputscale:{np.round(inf_gp_model_partial.covar_module.outputscale.detach().cpu().numpy(), 5)}\n"
    print(to_print)

if args.eval_on_prior_model:
    inf_likelihood_partial.eval()
    test_acc, test_nll, test_labels_vs_preds = model_evalution(test_loader, inf_gp_model_partial)
    logging.info(f"Test - NLL: {test_nll:.5f}, Accuracy: {test_acc:.5f}")

#################################################
# Init full infinite model with learned hps
#################################################
# transform all labels
inf_likelihood = DirichletClassificationLikelihood(train_y.to(device), learn_additional_noise=True).to(device)
# the following line doesn't really matter
inf_likelihood.second_noise_covar.initialize(noise=inf_likelihood_partial.second_noise_covar.noise)
inf_gp_model = MultiClassGPModelNeuralTangents(train_x=train_x.view(num_data, -1).to(device),
                                               train_y=inf_likelihood.transformed_targets,
                                               likelihood=inf_likelihood, neural_kernel_fun=kernel_fn,
                                               num_outputs=num_classes,
                                               input_dim=train_x[0, ...].shape,
                                               kernel_type='nngp',
                                               normalize=args.normalize, jitter_val=args.jitter_val).to(device)
inf_gp_model.ker_fun.set_K(K)
# init hp too
for ((full_n, full_p), (partial_n, partial_p)) in zip(inf_gp_model.named_parameters(),
                                                      inf_gp_model_partial.named_parameters()):
    full_p.data = partial_p.data

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
mll = gpytorch.mlls.VariationalELBO(likelihood, gp_model.gp_layer, num_data=len(train_loader.dataset),
                                    combine_terms=False)

epoch_iter = trange(args.num_epochs)
num_inner_steps = 0


def one_direction_loss(X_train, X_test, Y_train, Y_test, train_ind, test_ind):

    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    transformed_y_train = inf_likelihood.transformed_targets[:, train_ind]

    # obtain prior distribution
    inf_gp_model.set_train_data(X_train.view(num_train, -1), transformed_y_train, strict=False)
    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        prior_dist = inf_gp_model(X_test.view(num_test, -1),
                                  **{'phase': 'train2',
                                     'train_indices': train_ind.to(K_device),
                                     'test_indices': test_ind.to(K_device)})

    post_dist = gp_model(X_test)  # returns q(f)
    ELL, _, _ = mll(post_dist, Y_test)
    post_dist_mean = post_dist.mean
    post_dist_var = post_dist.variance.clamp_min(1e-8)

    entropy_term = univariate_gaussian_entropy(post_dist_var, 1)
    ce_trem = univariate_gaussian_cross_entropy(prior_dist.mean.t(), prior_dist.variance.clamp_min(1e-8).t(),
                                                post_dist_mean, post_dist_var, 1)
    KL = (entropy_term - ce_trem).mean(1).mean()

    return ELL, KL


# In the below make sure you know the kernel entries to take as it may get mixed because of the training shuffling
for epoch in epoch_iter:
    gp_model.train()
    likelihood.train()
    inf_gp_model.eval()

    cumm_loss = num_samples = num_batches = 0

    with gpytorch.settings.num_likelihood_samples(args.num_train_samples):
        for k, batch in enumerate(train_loader):

            batch = (t.to(device) for t in batch)
            train_data, train_labels, train_idx = batch
            num_batch_data = train_data.shape[0]
            optimizer.zero_grad()

            D1_size = int(num_batch_data * args.train_ratio)  # number of samples to take from training data
            D1_batch_ind = torch.sort(torch.argsort(torch.rand(num_batch_data, device=train_data.device))[:D1_size])[0]
            D2_batch_ind = torch.as_tensor([i for i in range(num_batch_data) if i not in D1_batch_ind],
                                            device=D1_batch_ind.device)

            D1_X, D1_y, D1_ind = train_data[D1_batch_ind, ...], train_labels[D1_batch_ind, ...], train_idx[D1_batch_ind]
            D2_X, D2_y, D2_ind = train_data[D2_batch_ind, ...], train_labels[D2_batch_ind, ...], train_idx[D2_batch_ind]

            ELL1, KL1 = one_direction_loss(D1_X, D2_X, D1_y, D2_y, D1_ind, D2_ind)
            ELL2, KL2 = one_direction_loss(D2_X, D1_X, D2_y, D1_y, D2_ind, D1_ind)

            loss1 = - ELL1 + args.beta * KL1
            loss2 = - ELL2 + args.beta * KL2

            loss = args.train_ratio * loss1 + (1 - args.train_ratio) * loss2
            loss.backward()
            optimizer.step()

            epoch_iter.set_description(f'[{epoch}][{k}] losses: {loss.item():.5f}, {loss1.item():.5f}, '
                                       f'{loss2.item():.5f}, ELLs: {ELL1.item():.5f}, {ELL2.item():.5f}, '
                                       f'KLs: {KL1.item():.5f}, {KL2.item():.5f}')

            cumm_loss += loss.item() * train_labels.shape[0]
            num_samples += train_labels.shape[0]
            num_batches += 1
            num_inner_steps += 1

    cumm_loss /= num_samples

    net_norm = torch.cat([p.view(-1) for p in gp_model.feature_extractor.parameters()]).norm()
    print(f"epoch: {epoch}, cumm_loss {cumm_loss:.5f}, net norm: {net_norm.item():.5f}\n")
    scheduler.step()

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

