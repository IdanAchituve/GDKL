import torch
import torch.nn as nn
import numpy as np
from gpytorch.kernels import Kernel
from typing import Optional
from neural_tangents import stax


class NeuralTangentKernel(Kernel):
    def __init__(self, neural_kernel_fun, kernel_type='nngp', normalize=False, num_outputs=1,
                 real_input_dims=(),
                 batch_shape: Optional[torch.Size] = torch.Size([]), **kwargs):

        Kernel.__init__(self, has_lengthscale=False, batch_shape=batch_shape, **kwargs)
        self.neural_kernel_fun = neural_kernel_fun
        self.kernel_type = kernel_type
        self.normalize = normalize
        self.num_outputs = num_outputs
        self.real_input_dims = real_input_dims  # the dimension of the one data sample
        self.K = None

    def set_K(self, K):
        self.K = K

    def forward_set_kernel_fragments(self, x, fragment_size=2500, multiplicative_factor=1):
        """
        Set the kernel one time at the beginning of the run for the whole training stage
        :param x: training data
        :param fragment_size: build kernel for big datasets in fragments
        :param multiplicative_factor: to make the kernel values nice (~ < 100)
        """
        num_data = x.shape[0]
        num_fragments = num_data // fragment_size  # assume 2500 divides fragment

        d_0 = torch.numel(x[0, ...])

        # flatten features
        x1_ = x.contiguous().view(x.shape[0], -1) if len(self.real_input_dims) > 0 else x
        # normalize
        x1_ = x1_ / torch.linalg.norm(x1_, dim=-1, ord=2, keepdim=True) * d_0 if self.normalize else x1_
        # return to original shape
        x1_ = x1_.view(*x1_.shape[:-1], *self.real_input_dims) if len(self.real_input_dims) > 0 else x1_

        K = np.zeros((num_data, num_data))
        for i in range(num_fragments):
            x_i = x1_[fragment_size * i: fragment_size * (i + 1)].detach().cpu().numpy()
            for j in range(i, num_fragments):
                x_j = x1_[fragment_size * j: fragment_size * (j + 1)].detach().cpu().numpy()
                K_fragment = self.neural_kernel_fun(x_i, x_j, self.kernel_type).block_until_ready()
                K_strip = K_fragment if j == i else np.concatenate((K_strip, K_fragment), axis=1)
            K[fragment_size * i: fragment_size * (i + 1), fragment_size * i:] = K_strip
            K[fragment_size * i:, fragment_size * i:fragment_size * (i + 1)] = K_strip.transpose()

        # store it on cpu and add device only later when using it.
        self.K = multiplicative_factor * torch.tensor(np.asarray(K), dtype=x.cpu().dtype)
        self.first_time_pred_strategy = True
        return self.K

    def forward_set_kernel(self, x, multiplicative_factor=1):
        """
        Set the kernel one time at the beginning of the run for the whole training stage
        :param x: training data
        :param multiplicative_factor: to make the kernel values nice (~ < 100)
        """
        d_0 = torch.numel(x[0, ...])

        # flatten features
        x1_ = x.contiguous().view(x.shape[0], -1) if len(self.real_input_dims) > 0 else x
        # normalize
        x1_ = x1_ / torch.linalg.norm(x1_, dim=-1, ord=2, keepdim=True) * d_0 if self.normalize else x1_
        # return to original shape
        x1_ = x1_.view(*x1_.shape[:-1], *self.real_input_dims) if len(self.real_input_dims) > 0 else x1_
        K = self.neural_kernel_fun(x1_.detach().cpu().numpy(), None,
                                   self.kernel_type).block_until_ready()

        self.K = multiplicative_factor * torch.tensor(np.asarray(K), dtype=x.dtype).to(x.device)
        self.first_time_pred_strategy = True
        return self.K

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        Three possible phases:
        1. "train": Training of the GP with the neural kernel: here I need to use the indices
        2. "test": Test with new samples: here I need to use the data
        3. "train2": Training the DKL model. here I need to use the indices in prediction mode of ExactGP
            a. First call on the kernel of the training indices (train-train)
            b. second call on the kernel of train and test indices (with that order) (train+test - train+test)
        """
        if params['phase'] == 'train':
            K = self.K.to(x1.device)

        elif params['phase'] == 'train_idx':
            train_indices = params['train_indices']
            K = torch.clone(self.K[train_indices, :][:, train_indices].contiguous()).to(x1.device)

        elif params['phase'] == 'test':
            d_0 = torch.numel(x1[0, ...])

            # normalize input since nngp/ntk doesn't do it automatically
            x1_ = x1 / torch.linalg.norm(x1, dim=-1, ord=2, keepdim=True) * d_0 if self.normalize else x1
            x2_ = x2 / torch.linalg.norm(x2, dim=-1, ord=2, keepdim=True) * d_0 if self.normalize else x2

            if len(self.real_input_dims) > 0:
                # squeeze since pytorch may add batch dimension of 1 for the classes
                x1_ = x1_.view(*x1_.shape[:-1], *self.real_input_dims).squeeze(0)
                x2_ = x2_.view(*x2_.shape[:-1], *self.real_input_dims).squeeze(0)

            K = self.neural_kernel_fun(x1_.detach().cpu().numpy(), x2_.detach().cpu().numpy(),
                                       self.kernel_type).block_until_ready()
            K = torch.tensor(np.asarray(K), dtype=x1.dtype).to(x1.device)

        elif params['phase'] == 'test_cache':
            if self.first_time_pred_strategy:  # first time get the kernel between training data
                K = self.K
                self.first_time_pred_strategy = False

            else:  # all other times: train+test - train+test
                d_0 = torch.numel(x1[0, ...])

                # normalize input since nngp/ntk doesn't do it automatically
                x1_ = x1 / torch.linalg.norm(x1, dim=-1, ord=2, keepdim=True) * d_0 if self.normalize else x1
                x2_ = x2 / torch.linalg.norm(x2, dim=-1, ord=2, keepdim=True) * d_0 if self.normalize else x2

                if len(self.real_input_dims) > 0:
                    # squeeze since pytorch may add batch dimension of 1 for the classes
                    x1_ = x1_.view(*x1_.shape[:-1], *self.real_input_dims).squeeze(0)
                    x2_ = x2_.view(*x2_.shape[:-1], *self.real_input_dims).squeeze(0)

                num_train = self.K.shape[0]
                K_tr_tr = self.K
                K_tr_ts = self.neural_kernel_fun(x1_[:num_train, ...].detach().cpu().numpy(),
                                                 x2_[num_train:, ...].detach().cpu().numpy(),
                                                 self.kernel_type).block_until_ready()
                K_tr_ts = torch.tensor(np.asarray(K_tr_ts), dtype=x1.dtype).to(x1.device)
                K_ts_ts = self.neural_kernel_fun(x1_[num_train:, ...].detach().cpu().numpy(),
                                                 None, self.kernel_type).block_until_ready()
                K_ts_ts = torch.tensor(np.asarray(K_ts_ts), dtype=x1.dtype).to(x1.device)

                K1 = torch.cat((K_tr_tr, K_tr_ts), dim=1)
                K2 = torch.cat((K_tr_ts.t(), K_ts_ts), dim=1)
                K = torch.cat((K1, K2), dim=0)

                # K = self.neural_kernel_fun(x1_.detach().cpu().numpy(), x2_.detach().cpu().numpy(),
                #                            self.kernel_type).block_until_ready()
                #K = torch.tensor(np.asarray(K), dtype=x1.dtype).to(x1.device)

        elif params['phase'] == 'train2':
            train_indices = params['train_indices']
            test_indices = params['test_indices']
            indices = torch.cat((train_indices, test_indices))
            # first check if we are on the first call
            #if all(torch.equal(train_input, test_input) for train_input, test_input in zip(x1, x2)):
            if x1.shape[0] == train_indices.shape[0]:  # train-train mode
                K = torch.clone(self.K[train_indices, :][:, train_indices]).contiguous().to(x1.device)
            else:
                K = torch.clone(self.K[indices, :][:, indices]).contiguous().to(x1.device)

        else:
            raise Exception("Unknown neural kernel phase")

        if self.num_outputs > 1:
            K = K.repeat(self.num_outputs, 1, 1)
        #neural_kernel = torch.tensor(np.asarray(K), dtype=x1.dtype).to(x1.device)
        return K