import math
import numpy as np
from tqdm import trange
import argparse
import logging
from pathlib import Path
import torch
import gpytorch
from GDKL.GP.models import GPModelExact
import gpytorch.likelihoods as likelihoods
import GDKL.nn.networks as networks
from experiments.uci.data import get_regression_data, split_train, get_data_loaders
from experiments.utils import get_device, set_seed, save_experiment, set_logger, common_parser, \
    str2bool, log_marginal_likelihood

parser = argparse.ArgumentParser(description="UCI - GP/DKL", parents=[common_parser])

#############################
#       Dataset Args        #
#############################
parser.add_argument('--data_path', type=str, default='./datasets', metavar='PATH',
                    help='path to datasets location (MUST be datasets)')
parser.add_argument('--dataset', type=str, default='boston',
                    choices=['boston', 'concrete', 'energy', 'buzz', 'ctslice'])
parser.add_argument('--perc', type=float, default=1.0,
                    help='portion of the trainset to allocate for validation ')
parser.add_argument('--num-samples-train', type=int, default=None,
                    help='number of samples to take from the training set')

##############################
#       GP Model args        #
##############################
parser.add_argument("--kernel-function", type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'RQKernel'],
                    help="kernel function")
parser.add_argument('--normalize-gp-input', type=str, default='none',
                    choices=['scale', 'norm', 'none'],
                    help='type of normalization of GP input')
parser.add_argument("--ARD", type=str2bool, default=False,
                    help="used ARD")
parser.add_argument("--learn-noise", type=str2bool, default=True,
                help="learn \sigma_n?")
parser.add_argument('--noise-factor', type=float,
                    default=-4.05,
                    help='the noise constant in the first layer')

##############################
#       NN Model args        #
##############################
parser.add_argument("--network", type=str, default='Identity',
                    choices=['FCNet', 'Identity'],
                    help="Neural Network")
parser.add_argument('--layers', type=lambda s: [int(item.strip()) for item in s.split('.')],
                    default='100.100.100.20')
parser.add_argument("--activation", type=str, default='relu',
                    choices=['relu', 'elu'])
parser.add_argument("--dropout-rate", type=float, default=0.0)
parser.add_argument("--DUE", type=str2bool, default=False,
                    help="learn \sigma_n?")
parser.add_argument("--coeff", type=float, default=3., help="Spectral normalization coefficient")
parser.add_argument(
        "--n-power-iterations", default=1, type=int, help="Number of power iterations"
    )

##############################
#     Optimization args      #
##############################
parser.add_argument("--num-epochs", type=int, default=8000)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--test-batch-size", type=int, default=2048)
parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='4800,6400')

args = parser.parse_args()

set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

exp_name = f'DKL-UCI_DUE_{args.DUE}_{args.dataset}_seed_{args.seed}_num-samples-train_{args.num_samples_train}_' \
           f'net_{args.network}_kernel_{args.kernel_function}'

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
X_train, X_val, Y_train, Y_val = split_train(data, val_size=1.0 - args.perc)
trainloader, _, testloader = \
    get_data_loaders(X_train, Y_train, X_val, Y_val, data.X_test, data.Y_test,
                     args.batch_size, args.test_batch_size, num_workers=args.num_workers,
                     shuffle=False, num_samples_train=args.num_samples_train)
data_dim = data.D
num_data = trainloader.dataset.tensors[0].shape[0]

###############################
# Model
###############################
likelihood = likelihoods.GaussianLikelihood().double().to(device)
likelihood.noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor)], device=device))

# network_args = (data_dim, args.num_features, args.depth, args.spectral_norm,
#                 args.dropout_rate, args.activation, num_outputs) if args.network == 'FCResNet' else ()
network = getattr(networks, args.network)(input_dim=data_dim,
                                          layers=args.layers,
                                          dropout_rate=args.dropout_rate,
                                          activation=args.activation,
                                          DUE=args.DUE,
                                          coeff=args.coeff,
                                          n_power_iterations=args.n_power_iterations).to(device)

# set ARD
ARD_dim = None
if args.ARD:
    if args.network == 'Identity':
        ARD_dim = data_dim
    else:
        ARD_dim = args.layers[-1]

gp_model = GPModelExact(train_x=trainloader.dataset.tensors[0],
                        train_y=trainloader.dataset.tensors[1],
                        likelihood=likelihood,
                        feature_extractor=network,
                        normalize_gp_input=args.normalize_gp_input,
                        kernel_function=args.kernel_function,
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

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

###############################
# Evaluate
###############################
# evaluating test/val data
@torch.no_grad()
def model_evalution(loader):
    gp_model.eval()

    ll = rmse = mae = num_samples = 0

    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        test_data, test_labels = batch

        output = gp_model(test_data)
        p_y_X = gp_model.likelihood(output)  # returns the marginal distribution
        ll += log_marginal_likelihood(test_labels, p_y_X).sum()
        rmse += torch.sum((test_labels - output.mean) ** 2)
        mae += torch.sum(torch.abs(test_labels - output.mean))
        num_samples += test_labels.shape[0]

    ll = (ll / num_samples) - math.log(data.Y_std.item())
    rmse = ((rmse / num_samples) ** 0.5) * data.Y_std.item()
    mae = (mae / num_samples) * data.Y_std.item()

    return ll, rmse, mae

###############################
# train loop
###############################
epoch_iter = trange(args.num_epochs)
for epoch in epoch_iter:

    gp_model.train()
    likelihood.train()

    for k, batch in enumerate(trainloader):
        batch = (t.to(device) for t in batch)
        x_batch, y_batch = batch

        optimizer.zero_grad()
        output = gp_model(x_batch)
        loss = - mll(output, y_batch)
        loss.backward()
        optimizer.step()

        to_print = f"Train NMLL: {loss.item():.5f}, " \
                   f"noise:{gp_model.likelihood.noise.item():.5f}, " \
                   f"outputscale: {gp_model.covar_module.outputscale.item():.5f}"

        if args.network != "Identity":
            to_print += f" ,net norm: {torch.cat([p.view(-1) for p in gp_model.feature_extractor.parameters()]).norm():.5f}"

        print(to_print)

    scheduler.step()

train_ll, train_rmse, train_mae = model_evalution(trainloader)
test_ll, test_rmse, test_mae = model_evalution(testloader)
logging.info(f"Train - LL: {train_ll:.5f}, RMSE: {train_rmse:.5f}, MAE: {train_mae:.5f}")
logging.info(f"Test - LL: {test_ll:.5f}, RMSE: {test_rmse:.5f}, MAE: {test_mae:.5f}")