import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
import argparse
from contextlib import contextmanager
import os
import json
from pathlib import Path
import sys
from io import BytesIO
import warnings

common_parser = argparse.ArgumentParser(add_help=False, description="shared parser")

### General ###
common_parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
common_parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
common_parser.add_argument('--num-workers', default=0, type=int, help='num workers')
common_parser.add_argument("--eval-every", type=int, default=1, help="eval every X selected steps")
common_parser.add_argument("--save-path", type=str, default="./output", help="dir path for output file")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")


def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def get_device(cuda=True, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and cuda else "cpu"
    )

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# create folders for saving models and logs
def _init_(out_path, exp_name):
    script_path = os.path.dirname(__file__)
    script_path = '.' if script_path == '' else script_path
    if not os.path.exists(out_path + '/' + exp_name):
        os.makedirs(out_path + '/' + exp_name)
    # save configurations
    os.system('cp -r ' + script_path + '/*.py ' + out_path + '/' + exp_name)


def get_art_dir(args):
    art_dir = Path(args.out_dir)
    art_dir.mkdir(exist_ok=True, parents=True)

    curr = 0
    existing = [
        int(x.as_posix().split('_')[-1])
        for x in art_dir.iterdir() if x.is_dir()
    ]
    if len(existing) > 0:
        curr = max(existing) + 1

    out_dir = art_dir / f"version_{curr}"
    out_dir.mkdir()

    return out_dir


def save_experiment(args, results, return_out_dir=False, save_results=True):
    out_dir = get_art_dir(args)

    json.dump(
        vars(args),
        open(out_dir / "meta.experiment", "w")
    )

    # loss curve
    if save_results:
        json.dump(results, open(out_dir / "results.experiment", "w"))

    if return_out_dir:
        return out_dir


def topk(true, pred, k):
    max_pred = np.argsort(pred, axis=1)[:, -k:]  # take top k
    two_d_true = np.expand_dims(true, 1)  # 1d -> 2d
    two_d_true = np.repeat(two_d_true, k, axis=1)  # repeat along second axis
    return (two_d_true == max_pred).sum()/true.shape[0]


# def min_max_scaling(x, minlogits, temp):
#     # softmax with temperature per output
#     return (self.max_std - self.min_std) * (y - self.min_int) / (self.max_int - self.min_int) + self.min_std


def tempered_softmax(logits, temp):
    # softmax with temperature per output
    probs = F.softmax(logits / (temp + 1e-8), dim=-1)
    return probs


def to_one_hot(y, num_classes, dtype=torch.double):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], num_classes), dtype=dtype, device=y.device)
    return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)


def CE_loss(y, y_hat, num_classes, reduction='mean', convert_to_one_hot=True):
    # convert a single label into a one-hot vector
    y_output_onehot = to_one_hot(y, num_classes, dtype=y_hat.dtype) if convert_to_one_hot else y_hat
    if reduction == 'mean':
        return - torch.sum(y_output_onehot * torch.log(y_hat + 1e-12), dim=1).mean()
    return - torch.sum(y_output_onehot * torch.log(y_hat + 1e-12))


def entropy_loss(probs, reduction='mean'):
    if reduction == 'mean':
        return - torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean()
    return - torch.sum(probs * torch.log(probs + 1e-12))


def calc_metrics(results):
    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])
    avg_loss = np.mean([val['loss'] for val in results.values()])
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def model_save(model, file=None):
    if file is None:
        file = BytesIO()
    torch.save({'model_state_dict': model.state_dict()}, file)
    return file


def model_load(model, file):
    if isinstance(file, BytesIO):
        file.seek(0)

    model.load_state_dict(
        torch.load(file, map_location=lambda storage, location: storage)['model_state_dict']
    )

    return model


def save_data(tensor, file):
    torch.save(tensor, file)


def load_data(file):
    return torch.load(file, map_location=lambda storage, location: storage)


def map_labels(labels):
    new_labels = np.arange(max(labels) + 1)
    original_labels = np.unique(labels)
    orig_to_new = {o: n for o, n in zip(original_labels, new_labels)}
    return np.asarray([orig_to_new[l] for l in labels]).astype(np.long)


def take_subset_of_classes(data, labels, classes):
    targets = np.asarray(labels)
    indices = np.isin(targets, classes)
    new_data, new_labels = data[indices], targets[indices].tolist()
    return new_data, map_labels(new_labels)


class DuplicateToChannels:
    """Duplicate single channel 3 times"""

    def __init__(self):
        pass

    def __call__(self, x):
        return x.repeat((3, 1, 1))


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

# Adapted from: https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/cholesky.py
def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
            uses settings.cholesky_jitter.value()
        :attr:`max_tries` (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    L = _psd_safe_cholesky(A, out=out, jitter=jitter, max_tries=max_tries)
    if upper:
        if out is not None:
            out = out.transpose_(-1, -2)
        else:
            L = L.mT
    return L


def _psd_safe_cholesky(A, out=None, jitter=None, max_tries=None):

    if out is not None:
        out = (out, torch.empty(A.shape[:-2], dtype=torch.int32, device=out.device))

    L, info = torch.linalg.cholesky_ex(A, out=out)
    isnan = torch.isnan(A)
    if isnan.any():
        raise warnings.warn(f"cholesky_cpu: {isnan.sum().item()} of "
                            f"{A.numel()} elements of the {A.shape} tensor are NaN.")

    if jitter is None:
        jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
    if max_tries is None:
        max_tries = 3
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # add jitter only where needed
        diag_add = ((info > 0) * (jitter_new - jitter_prev)).unsqueeze(-1).expand(*Aprime.shape[:-1])
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        warnings.warn(f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal")
        L, info = torch.linalg.cholesky_ex(Aprime, out=out)
        if not torch.any(info):
            return L
    raise ValueError(f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}.")


def log_marginal_likelihood(values, p_Y_X=None, loc=None, var=None):
    """
    computes the log marginal likelihood of a diagonal Gaussian
    :param p_Y_X: the marginal distribution
    :param values: the values to the ll over
    :return:
    """
    if loc is None:
        loc = p_Y_X.mean
    if var is None:
        var = p_Y_X.covariance_matrix.diagonal()
    if any(var < 0.0):
        warnings.warn('Warning: negative variance clamped')
    # clamp min value as in gpytorch:
    # https://github.com/cornellius-gp/gpytorch/blob/1c490ddea83fca607ebb209d0d9d8816bb363d9b/gpytorch/likelihoods/gaussian_likelihood.py#L60
    p_Y_X_diag = torch.distributions.Normal(loc=loc,
                                            scale=var.clamp_min(1e-8).sqrt())
    return p_Y_X_diag.log_prob(values)


def univariate_gaussian_entropy(variance, N):
    log_var = torch.log(variance)
    res = N * math.log(2 * math.pi) + log_var + N
    return - res.mul(0.5)


def univariate_gaussian_cross_entropy(p_mu, p_var, q_mu, q_var, N):
    log_p_var = torch.log(p_var)
    var_ratio = q_var / p_var
    quadratic_term = ((q_mu - p_mu) ** 2) / p_var
    res = N * math.log(2 * math.pi) + log_p_var + var_ratio + quadratic_term
    return - res.mul(0.5)