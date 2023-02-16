import logging
from pathlib import Path
import math
import numpy as np
from tqdm import trange
import torch
import gpytorch
from GDKL.GP.models import GPModelExactPredictive, GPModelNeuralTangents
import gpytorch.likelihoods as likelihoods
import GDKL.nn.networks as networks
from experiments.uci.data import get_regression_data, split_train, get_data_loaders
from experiments.utils import get_device, set_seed, save_experiment, set_logger, common_parser, \
    str2bool, log_marginal_likelihood, univariate_gaussian_entropy, univariate_gaussian_cross_entropy
from neural_tangents import stax
import argparse


parser = argparse.ArgumentParser(description="Regression", parents=[common_parser])

#############################
#       Dataset Args        #
#############################
parser.add_argument('--data_path', type=str, default='./datasets', metavar='PATH',
                    help='path to datasets location (MUST be datasets)')
parser.add_argument('--dataset', type=str, default='boston',
                    choices=['boston', 'concrete', 'energy', 'buzz', 'ctslice'])
parser.add_argument('--pct', type=float, default=1.0,
                    help='portion of trainset (out of train+valid)')
parser.add_argument('--num-samples-train', type=int, default=None,
                    help='number of samples to take from the training set')

##############################
#       GP Model args        #
##############################
parser.add_argument("--mean-function", type=str, default='zero',
                    choices=['zero', 'linear'],
                    help="mean function for the first layer")
parser.add_argument("--kernel-function", type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'RQKernel'],
                    help="kernel function")
parser.add_argument('--normalize-gp-input', type=str, default='none',
                    choices=['scale', 'norm', 'none'],
                    help='type of normalization of GP input')
parser.add_argument("--ARD", type=str2bool, default=False,
                    help="used ARD")
parser.add_argument("--weight-var", type=float, default=1.6)
parser.add_argument("--bias-var", type=float, default=0.2)
parser.add_argument("--beta", type=float, default=1.0, help="coef of data kl")
parser.add_argument("--inf-depth", type=int, default=4,
                    help="infinite network depth")
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
parser.add_argument("--network", type=str, default='FCNet',
                    choices=['FCNet'],
                    help="Neural Network")
parser.add_argument('--layers', type=lambda s: [int(item.strip()) for item in s.split('.')],
                    default='100.100.100.20')
parser.add_argument("--net-activation", type=str, default='relu',
                    choices=['relu', 'elu'])
parser.add_argument("--dropout-rate", type=float, default=0.0)
parser.add_argument("--train-ratio", type=float, default=0.5,
                    help='ratio of examples to allocate for training the posterior')

##############################
#     Optimization args      #
##############################
parser.add_argument("--num-pretrain-epochs", type=int, default=1000)
parser.add_argument("--num-epochs", type=int, default=7000)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--test-batch-size", type=int, default=2048)
parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='4200,5600')

args = parser.parse_args()

set_logger()
set_seed(args.seed)
state = torch.get_rng_state()

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

exp_name = f'GDKL_UCI_{args.dataset}_seed_{args.seed}_num-samples-train_{args.num_samples_train}_' \
           f'train-ratio_{args.train_ratio}_beta_{args.beta}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)

################################
# get data
################################
data = get_regression_data(args.dataset)
X_train, X_val, Y_train, Y_val = split_train(data, val_size=1.0 - args.pct)
train_loader, _, test_loader = \
    get_data_loaders(X_train, Y_train, X_val, Y_val, data.X_test, data.Y_test,
                     args.batch_size, args.test_batch_size, num_workers=args.num_workers,
                     shuffle=False, num_samples_train=args.num_samples_train)

data_dim = data.D
trainset = train_loader.dataset.tensors[0].to(device)
trainset_y = train_loader.dataset.tensors[1].to(device)
num_data = trainset.shape[0]

###############################
# Infinite Network
###############################
network = getattr(networks, args.network)(input_dim=data_dim,
                                          layers=args.layers,
                                          dropout_rate=args.dropout_rate,
                                          activation=args.net_activation).to(device)

###############################
# Infinite Kernel
###############################
inf_likelihood = likelihoods.GaussianLikelihood().double().to(device)
inf_likelihood.noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor)], device=device))

def stax_layer():
    return stax.serial(
      stax.Dense(1, W_std=np.sqrt(args.weight_var), b_std=np.sqrt(args.bias_var)),
      getattr(stax, args.neural_activation)())

prior_net_layers = []
prior_net_layers += [stax_layer() for d in range(args.inf_depth - 1)]
prior_net_layers += [stax.Dense(1, W_std=np.sqrt(args.weight_var), b_std=np.sqrt(args.bias_var))]
_, __, kernel_fn = stax.serial(*prior_net_layers)
inf_gp_model = GPModelNeuralTangents(train_x=trainset, train_y=trainset_y,
                                     likelihood=inf_likelihood, neural_kernel_fun=kernel_fn,
                                     normalize=args.normalize).to(device)

# first need to set the kernel one time only
inf_gp_model.ker_fun.forward_set_kernel(trainset)
trainset_ind = torch.arange(trainset_y.shape[0]).to(device)

###############################
# Evaluate
###############################
@torch.no_grad()
def model_evalution(loader, gp_model):
    gp_model.eval()

    ll = rmse = mae = num_samples = 0

    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        test_data, test_labels = batch

        output = gp_model(test_data, **{'phase': 'test'})
        p_y_X = gp_model.likelihood(output)  # returns the marginal distribution
        ll += log_marginal_likelihood(test_labels, p_y_X).sum()
        rmse += torch.sum((test_labels - output.mean) ** 2)
        mae += torch.sum(torch.abs(test_labels - output.mean))
        num_samples += test_labels.shape[0]

    ll = (ll / num_samples) - math.log(data.Y_std.item())
    rmse = ((rmse / num_samples) ** 0.5) * data.Y_std.item()
    mae = (mae / num_samples) * data.Y_std.item()

    return ll, rmse, mae

params = [{'params': inf_gp_model.parameters()}]
optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=0.0, momentum=0.9, nesterov=False) \
        if args.optimizer == 'sgd' else torch.optim.Adam(params, lr=args.lr, weight_decay=0.0)
epoch_iter = trange(args.num_pretrain_epochs)

inf_mll = gpytorch.mlls.ExactMarginalLogLikelihood(inf_likelihood, inf_gp_model)
inf_gp_model.train()

for epoch in epoch_iter:
    loss = 0.0

    for k, batch in enumerate(train_loader):
        batch = (t.to(device) for t in batch)
        x_batch, y_batch = batch

        optimizer.zero_grad()
        output = inf_gp_model(x_batch, **{'phase': 'train'})
        loss = - inf_mll(output, y_batch)

        loss.backward()
        optimizer.step()

    to_print = f"Train loss: {loss.item():.5f}, "
    to_print += f"noise prior: {inf_gp_model.likelihood.noise.detach().item():.5f}, "
    to_print += f"outputscale prior: {inf_gp_model.covar_module.outputscale.detach().item():.5f}"

    print(to_print)

train_ll, train_rmse, train_mae = model_evalution(train_loader, inf_gp_model)
test_ll, test_rmse, test_mae = model_evalution(test_loader, inf_gp_model)
logging.info(
    f"Train1 - LL: {train_ll:.5f}, RMSE1: {train_rmse:.5f}, MAE1: {train_mae:.5f}")
logging.info(
    f"Test1 - LL: {test_ll:.5f}, RMSE1: {test_rmse:.5f}, MAE1: {test_mae:.5f}")


###############################
# Model
###############################
likelihood = likelihoods.GaussianLikelihood().double().to(device)
likelihood.noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor)], device=device))

ARD_dim = None
if args.ARD:
    ARD_dim = args.layers[-1]

gp_model = GPModelExactPredictive(train_x=trainset, train_y=trainset_y,
                                  likelihood=likelihood, feature_extractor=network,
                                  normalize_gp_input=args.normalize_gp_input, kernel_function=args.kernel_function,
                                  data_dim=ARD_dim).double().to(device)

###############################
# Optimizer
###############################
params = [{'params': gp_model.feature_extractor.parameters(), 'weight_decay': args.wd},
          {'params': gp_model.covar_module.parameters()},
          {'params': gp_model.mean_module.parameters()},
          {'params': gp_model.likelihood.parameters()},
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
    X_train_ind = torch.sort(torch.argsort(torch.rand(num_data, device=trainset.device))[:num_samples])[0]
    X_test_ind = torch.as_tensor([i for i in range(num_data) if i not in X_train_ind], device=X_train_ind.device)

    X_train, y_train = trainset[X_train_ind, ...], trainset_y[X_train_ind, ...]
    X_test, y_test = trainset[X_test_ind, ...], trainset_y[X_test_ind, ...]

    # obtain prior distribution
    inf_gp_model.set_train_data(X_train, y_train, strict=False)
    with gpytorch.settings.lazily_evaluate_kernels(state=False):
        prior_dist = inf_gp_model(X_test, **{'phase': 'train2',
                                             'train_indices': X_train_ind,
                                             'test_indices': X_test_ind})

    # obtain posterior distribution
    X = torch.cat((X_train, X_test))
    Y = torch.cat((y_train, y_test))
    gp_model.set_train_data(X, Y, strict=False)
    # the following enforce taking gradients w.r.t the parameters from the predictive mean
    # in exact_prediction_strategies L231
    with gpytorch.settings.detach_test_caches(False):
        posterior_dist = gp_model.forward_predictive(X, Y, num_train=X_train.shape[0])

    ELL = likelihood.expected_log_prob(y_test, input=posterior_dist).mean()

    entropy_term = univariate_gaussian_entropy(posterior_dist.variance, 1)
    ce_trem = univariate_gaussian_cross_entropy(prior_dist.mean, prior_dist.variance,
                                                posterior_dist.mean, posterior_dist.variance, 1)
    KL = (entropy_term - ce_trem).mean()

    loss = - ELL + args.beta * KL
    loss.backward()
    optimizer.step()
    scheduler.step()

    to_print = f"Train loss: {loss.item():.5f}, " \
               f"ELL:{ELL.item():.5f}, " \
               f"KL: {KL.item():.5f}"

    net_norm = torch.cat([p.view(-1) for p in gp_model.feature_extractor.parameters()]).norm()
    to_print += f", net norm: {net_norm.item():.5f}"
    to_print += f", noise post: {likelihood.noise.detach().item():.5f}"
    to_print += f", noise prior: {inf_gp_model.likelihood.noise.detach().item():.5f}"
    to_print += f", outputscale post: {gp_model.covar_module.outputscale.detach().item():.5f}"
    to_print += f", outputscale prior: {inf_gp_model.covar_module.outputscale.detach().item():.5f}"

    print(to_print)

    num_inner_steps += 1

gp_model.set_train_data(trainset, trainset_y, strict=False)
train_ll, train_rmse, train_mae = model_evalution(train_loader, gp_model)
test_ll, test_rmse, test_mae = model_evalution(test_loader, gp_model)
logging.info(
    f"Train2 - LL: {train_ll:.5f}, RMSE2: {train_rmse:.5f}, MAE2: {train_mae:.5f}")
logging.info(
    f"Test2 - LL: {test_ll:.5f}, RMSE2: {test_rmse:.5f}, MAE2: {test_mae:.5f}")
