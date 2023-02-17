import numpy as np
from tqdm import trange
import argparse
import logging
from pathlib import Path
import math
from datetime import datetime
import json

import torch
import torch.nn as nn
import gpytorch
from gpytorch.likelihoods import DirichletClassificationLikelihood
from GDKL.GP.models import MultiClassGPModelExact
import GDKL.nn.networks as networks
from experiments.cifar.DataClass import CIFARData
from experiments.utils import get_device, set_seed, save_experiment, set_logger, common_parser, \
    detach_to_numpy, str2bool, topk
from experiments.calibrate import ECELoss


parser = argparse.ArgumentParser(description="CIFAR10 - DKL", parents=[common_parser])

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
parser.add_argument("--kernel-function", type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel', 'RQKernel', 'SpectralMixtureKernel'],
                    help="kernel function")
parser.add_argument("--mean-function", type=str, default='zero',
                    choices=['zero', 'constant'],
                    help="kernel function")
parser.add_argument('--normalize-gp-input', type=str, default='none',
                    choices=['scale', 'norm', 'none'],
                    help='type of normalization of GP input')
parser.add_argument("--ARD", type=str2bool, default=False,
                    help="used ARD")
parser.add_argument('--noise-factor', type=float,
                    default=-4.05,
                    help='the noise constant in the first layer')
parser.add_argument("--num-test-samples", type=int, default=1024)


##############################
#     Optimization args      #
##############################
parser.add_argument("--num-epochs", type=int, default=7000)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--test-batch-size", type=int, default=512)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--kernel-lr", type=float, default=1e-2, help="learning rate of kernel params")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout")
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='4200,5600')
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

exp_name = f'DKL-CIFAR10_DUE_{args.DUE}_seed_{args.seed}_train-pct_{args.train_pct}_{args.val_pct}_' \
           f'kernel_{args.kernel_function}_ARD_{args.ARD}_net-conf_{args.widen_factor}'

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
cifar10data = CIFARData(args.data_path)
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
# Model
###############################
network = networks.WideResNet(widen_factor=args.widen_factor,
                              spectral_bn=args.DUE,
                              spectral_conv=args.DUE,
                              dropout_rate=args.dropout,
                              coeff=args.coeff,
                              n_power_iterations=args.n_power_iterations)

likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True).to(device)
likelihood.second_noise_covar.initialize(noise=torch.tensor([np.exp(args.noise_factor) for _ in range(num_classes)],
                                                            device=device))

# set ARD
ARD_dim = None
if args.ARD:
    ARD_dim = 64 * args.widen_factor

gp_model = MultiClassGPModelExact(train_x=train_x.view(num_data, -1), train_y=likelihood.transformed_targets,
                                  likelihood=likelihood, feature_extractor=network, num_outputs=num_classes,
                                  input_dim=train_x[0, ...].shape, mean_func=args.mean_function,
                                  normalize_gp_input=args.normalize_gp_input,
                                  kernel_function=args.kernel_function,
                                  data_dim=ARD_dim).to(device)

###############################
# Optimizer
###############################
params = [{'params': gp_model.feature_extractor.parameters(), 'weight_decay': args.wd, 'lr': args.lr},
          {'params': gp_model.likelihood.parameters(), 'lr': args.kernel_lr},
          {'params': gp_model.mean_module.parameters(), 'lr': args.kernel_lr},
          {'params': gp_model.covar_module.parameters(), 'lr': args.kernel_lr},
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


#############################
# Evaluate
#############################
@torch.no_grad()
def model_evalution(loader):
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

        # TODO:Note that we do not take into account the second raw noise that is learned during training
        post_pred_dist = gp_model(test_data.view(num_test_data, -1)) # returns the full covariance matrix of the predictive dist.

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


# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
epoch_iter = trange(args.num_epochs)

for epoch in epoch_iter:

    gp_model.train()
    likelihood.train()
    cumm_loss = num_samples = num_batches = 0

    optimizer.zero_grad()
    output = gp_model(train_x.view(num_data, -1))
    loss = - mll(output, likelihood.transformed_targets).sum()

    loss.backward()
    optimizer.step()

    to_print = f"Train loss: {loss.item():.5f}\n" \
               f"noise:{np.round(gp_model.likelihood.second_noise_covar.noise.squeeze().detach().cpu().numpy(), 3)}\n"
    to_print += f"outputscale:{np.round(gp_model.covar_module.outputscale.detach().cpu().numpy(), 3)}\n"
    print(to_print)
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

