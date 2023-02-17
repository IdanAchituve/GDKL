import torch.nn.functional as F
import gpytorch
from gpytorch import constraints
import gpytorch.kernels as kernels
from scipy.cluster.vq import kmeans2
from GDKL.GP.NeuralKernels import NeuralTangentKernel
from gpytorch.models import *
from gpytorch.models.exact_gp import *


@torch.no_grad()
def _init_IP(X, num_ip, kmeans=True):
    """
    Initialize inducing inputs according to kmeans
    :param X: training data
    :param num_ip: number if IPs
    :param kmeans: do kmeans?
    :return: initial lenghthscale
    """
    # Inducing points locations
    if num_ip < X.shape[0] and kmeans:
        Xbar = kmeans2(X.detach().cpu().numpy(), num_ip, minit='points')[0]
        Xbar = torch.tensor(Xbar, dtype=X.dtype).to(X.device)
    else:
        quotient = num_ip // X.shape[0]
        reminder = num_ip - quotient * X.shape[0]
        for i in range(int(quotient)):
            Xbar = X.clone() if i == 0 else torch.cat((Xbar, X.clone()))
        if reminder > 0:
            Xbar = torch.cat((Xbar, X[:reminder, :].clone()))

    return Xbar


class GPModelExact(ExactGP):

    def __init__(self, train_x, train_y, feature_extractor, likelihood, kernel_function,
                 normalize_gp_input='none', grid_bounds=(-4., 4.), data_dim=None):
        super(GPModelExact, self).__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor
        self.normalize_gp_input = normalize_gp_input
        self.mean_module = gpytorch.means.ZeroMean()
        self.kernel_function = kernel_function

        if kernel_function == "RBFKernel":
            self.ker_fun = kernels.RBFKernel(lengthscale_constraint=constraints.GreaterThan(1e-2),
                                             ard_num_dims=data_dim)
        elif kernel_function == "LinearKernel":
            self.ker_fun = kernels.LinearKernel(ard_num_dims=data_dim)
        elif kernel_function == "MaternKernel":
            self.ker_fun = kernels.MaternKernel(ard_num_dims=data_dim)
        elif kernel_function == 'RQKernel':
            self.ker_fun = kernels.RQKernel(ard_num_dims=data_dim)
        else:
            raise ValueError("Specified kernel not known.")

        self.covar_module = kernels.ScaleKernel(self.ker_fun)
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def forward(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        if self.normalize_gp_input == 'scale':
            x = self.scale_to_bounds(x)  # Make the NN values "nice"
        elif self.normalize_gp_input == 'norm':
            x = F.normalize(x, dim=-1)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiClassGPModelExact(ExactGP):

    def __init__(self, train_x, train_y, feature_extractor, likelihood, kernel_function, num_outputs, input_dim=(),
                 mean_func='zero', normalize_gp_input='none', grid_bounds=(-4., 4.), data_dim=None):
        super(MultiClassGPModelExact, self).__init__(train_x, train_y, likelihood)

        self.feature_extractor = feature_extractor
        self.normalize_gp_input = normalize_gp_input
        self.real_input_dims = input_dim
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size((num_outputs,))) if mean_func == 'zero' \
            else gpytorch.means.ConstantMean(batch_shape=torch.Size((num_outputs,)))
        self.kernel_function = kernel_function

        if kernel_function == "RBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun = kernels.RBFKernel(batch_shape=torch.Size((num_outputs,)),
                                             lengthscale_constraint=constraints.GreaterThan(1e-2),
                                             ard_num_dims=data_dim)
        elif kernel_function == "LinearKernel":
            self.ker_fun = kernels.LinearKernel(batch_shape=torch.Size((num_outputs,)), ard_num_dims=data_dim)
        elif kernel_function == "MaternKernel":
            self.ker_fun = kernels.MaternKernel(batch_shape=torch.Size((num_outputs,)), ard_num_dims=data_dim)
        elif kernel_function == 'RQKernel':
            self.ker_fun = kernels.RQKernel(batch_shape=torch.Size((num_outputs,)), ard_num_dims=data_dim)
        else:
            raise ValueError("Specified kernel not known.")

        self.covar_module = kernels.ScaleKernel(self.ker_fun, batch_shape=torch.Size((num_outputs,)))
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def forward(self, x, *args, **kwargs):
        x = x.view(*x.shape[:-1], *self.real_input_dims)
        x = self.feature_extractor(x)
        if self.normalize_gp_input == 'scale':
            x = self.scale_to_bounds(x)  # Make the NN values "nice"
        elif self.normalize_gp_input == 'norm':
            x = F.normalize(x, dim=-1)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(ApproximateGP):
    def __init__(self, inducing_locs, num_outputs, kernel_function, initial_lengthscale, num_kernels=1, data_dim=None,
                 learn_ip_loc=True, mean_function='zero'):

        batch_shape = torch.Size([]) if num_outputs == 1 else torch.Size([num_outputs])
        variational_distribution = gpytorch.variational. \
            CholeskyVariationalDistribution(num_inducing_points=inducing_locs.size(0), batch_shape=batch_shape)

        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_locs,
                                                                        variational_distribution,
                                                                        learn_inducing_locations=learn_ip_loc)
        if num_outputs > 1:
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(variational_strategy,
                                                                                                num_tasks=num_outputs)

        super(GPModel, self).__init__(variational_strategy)

        self.num_kernels = torch.Size([num_kernels])
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape) if mean_function == 'zero' else \
            gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.kernel_function = kernel_function
        if kernel_function == "RBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun = kernels.RBFKernel(lengthscale_constraint=constraints.GreaterThan(1e-2),
                                             ard_num_dims=data_dim,
                                             batch_shape=batch_shape)
        elif kernel_function == "LinearKernel":
            self.ker_fun = kernels.LinearKernel(ard_num_dims=data_dim,
                                                batch_shape=batch_shape)
        elif kernel_function == "MaternKernel":
            self.ker_fun = kernels.MaternKernel(ard_num_dims=data_dim,
                                                batch_shape=batch_shape)
        elif kernel_function == 'RQKernel':
            self.ker_fun = kernels.RQKernel(ard_num_dims=data_dim,
                                            batch_shape=batch_shape)
        elif kernel_function == 'SpectralMixtureKernel':
            self.covar_module = kernels.SpectralMixtureKernel(ard_num_dims=data_dim,
                                                              num_mixtures=4, batch_shape=batch_shape)
        else:
            raise ValueError("Specified kernel not known.")

        self.ker_fun.lengthscale = initial_lengthscale * torch.ones_like(self.ker_fun.lengthscale)
        if kernel_function != 'SpectralMixtureKernel':
            self.covar_module = kernels.ScaleKernel(self.ker_fun, batch_shape=batch_shape)

    def forward(self, x, *args, **kwargs):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, gp_layer=None, normalize_gp_input='none', grid_bounds=(-4., 4.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gp_layer
        self.normalize_gp_input = normalize_gp_input

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(grid_bounds[0], grid_bounds[1])

    def set_gp_layer(self, gp_layer):
        self.gp_layer = gp_layer

    def forward(self, x, *args, **kwargs):
        features = self.feature_extractor(x)
        if self.normalize_gp_input == 'scale':
            features = self.scale_to_bounds(features)  # Make the NN values "nice"
            #features = features.transpose(-1, -2).unsqueeze(-1)
        elif self.normalize_gp_input == 'norm':
            features = F.normalize(features, dim=-1)
        return self.gp_layer(features)


class GPModelNeuralTangents(ExactGP):

    def __init__(self, train_x, train_y, likelihood, neural_kernel_fun,
                 mean_func='zero', kernel_type='nngp', normalize=False):
        super(GPModelNeuralTangents, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean() if mean_func == 'zero' else gpytorch.means.ConstantMean()
        self.ker_fun = NeuralTangentKernel(neural_kernel_fun, kernel_type, normalize)
        self.covar_module = kernels.ScaleKernel(self.ker_fun)

    def forward(self, x, **params):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **params)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiClassGPModelNeuralTangents(ExactGP):

    def __init__(self, train_x, train_y, likelihood, neural_kernel_fun, num_outputs, input_dim=(),
                 mean_func='zero', kernel_type='nngp', normalize=False, jitter_val=0.0):
        super(MultiClassGPModelNeuralTangents, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size((num_outputs,))) if mean_func == 'zero' \
            else gpytorch.means.ConstantMean(batch_shape=torch.Size((num_outputs,)))
        self.ker_fun = NeuralTangentKernel(neural_kernel_fun, kernel_type, normalize, num_outputs,
                                           real_input_dims=input_dim)
        self.covar_module = kernels.ScaleKernel(self.ker_fun, batch_shape=torch.Size((num_outputs,)))
        self.num_data = train_x.shape[0]
        self.jitter_val = jitter_val
        # self.covar_module = kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size((num_classes,))), batch_shape=torch.Size((num_classes,)))

    def forward(self, x, **params):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, **params).add_jitter(self.jitter_val).evaluate()
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPPredictive(ExactGP):
    """
    Override the __call__ method of ExactGP to give the predictive distribution during training as well
    """
    def forward_predictive(self, X, Y, num_train, **kwargs):

        full_output = super().__call__(X, **kwargs)
        full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix
        train_output = gpytorch.distributions.MultivariateNormal(full_mean[..., :num_train],
                                                                 full_covar[..., :num_train, :num_train])

        self.prediction_strategy = prediction_strategy(
            train_inputs=X[:num_train, ...],
            train_prior_dist=train_output,
            train_labels=Y[..., :num_train],  # y can have multiple outputs
            likelihood=self.likelihood,
        )

        # Determine the shape of the joint distribution
        batch_shape = full_output.batch_shape
        joint_shape = full_output.event_shape
        tasks_shape = joint_shape[1:]  # For multitask learning
        test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

        # Make the prediction
        with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
            (
                predictive_mean,
                predictive_covar,
            ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

        # Reshape predictive mean to match the appropriate event shape
        predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
        return full_output.__class__(predictive_mean, predictive_covar)


class GPModelExactPredictive(GPModelExact, ExactGPPredictive):

    def __init__(self, train_x, train_y, feature_extractor, likelihood, kernel_function,
                 normalize_gp_input='none', grid_bounds=(-4., 4.), data_dim=None):
        super(GPModelExactPredictive, self).__init__(train_x, train_y, feature_extractor, likelihood, kernel_function,
                                                     normalize_gp_input, grid_bounds, data_dim)


class GPModelExactPredictiveMultiClass(MultiClassGPModelExact, ExactGPPredictive):

    def __init__(self, train_x, train_y, feature_extractor, likelihood, kernel_function, num_outputs, input_dim=(),
                 mean_func='zero', normalize_gp_input='none', grid_bounds=(-4., 4.), data_dim=None):
        super(GPModelExactPredictiveMultiClass, self).__init__(train_x, train_y, feature_extractor,
                                                               likelihood, kernel_function, num_outputs, input_dim,
                                                               mean_func, normalize_gp_input, grid_bounds, data_dim)
