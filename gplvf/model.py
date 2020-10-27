'''
Implements model API.

The setup is as follows:

EncoderBase implements the parameters of q_b(X), which is Gaussian
EncoderFlow implements the flow transformed q_f(X)

GPLVF implements the Gaussian Process Latent Variable Flows Model,
    using EncoderFlow as a variational approximation to X|Y.
'''

from uuid import uuid4
import pyro
import torch
import numpy as np
from tqdm import trange
import pyro.contrib.gp as gp
import pyro.distributions as dist
from torch import tensor as tt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.nn.functional import relu
import pickle as pkl
from gplvf.data import float_tensor
import scipy


class EncoderBase(torch.nn.Module):
    ''' '''
    def __init__(self, Y, num_latent, model, nn_layers=(4, 2),
                 gp_inducing_num=10):
        '''Initiates the base variational distribution.

        Parameters
        ----------
        Y : torch.tensor
            Dataset for the autoencoder/backconstraint.
        num_data : int
        num_latent : int
        model : str
            One of 'mf' (meanfield), 'pca', 'nn' or 'gp'.
        nn_layers : tuple
            Defaults to (4, 2). Both neural networks (for the mean and covar)
            have the len(nn_layers) of hidden layers. Non-output layer
            activations are relu, outputs have a linear activation. The input
            shape is d = len(Y.T) for both networks.

            Mean NN: The number of nodes per hidden layer is specified in the
            tuple for the mean neural network. Output dimension is num_latent.

            Covar NN: The number of nodes for the output layer of this network
            is such that a dxd lower triangular matrix can be constructed from
            the output layer. The number of nodes in the hidden layers is an
            average between the input and output layers. The diagonal elements
            of the lower triangular dxd matrix have a relu activation.
        gp_inducing_num : int
            Defaults to 10. Number of inducing points for the encoder sparse
            gaussian process.
        '''

        super().__init__()

        num_data = len(Y)
        self._Y = Y
        self.model = model

        if model == 'mf':
            self._init_mean_field(num_data, num_latent)
        elif model == 'pca':
            self._init_pca_param(num_latent)
        elif model == 'nn':
            self._init_nn(d=len(Y.T), q=num_latent, nn_layers=nn_layers)
        elif model == 'gp':
            self._init_gp(num_latent, gp_inducing_num)
        else:
            raise NotImplementedError('Invalid encoder/backconstraint.')

        self.update_parameters()

    def update_parameters(self):
        '''Updates mu and sigma, which are dependent on model parameters.'''
        if self.model == 'gp':
            self._update_gp_params()
        elif self.model == 'nn':
            self._update_nn_params()
        else:
            self._update_sigma()

    ''' ----------Meanfield and PCA methods---------- '''

    def _init_mean_field(self, n, q):
        Y_for_init = self._Y.numpy().copy()
        Y_for_init[np.isnan(Y_for_init)] = 0.0
        mu = PCA(q).fit_transform(Y_for_init)
        mu = mu + np.random.normal(scale=0.1, size=mu.shape)
        self.mu = torch.nn.Parameter(float_tensor(mu))

        log_sigma = float_tensor(np.zeros((n, q)))
        self._log_sigma = torch.nn.Parameter(log_sigma)

    def _init_pca_param(self, q):
        self._pca = PCA(q)
        self._pca.fit(self._Y)
        mu = self._pca.transform(self._Y)
        self.mu = float_tensor(mu)

        log_sigma = float_tensor(np.zeros(q))
        self._log_sigma = torch.nn.Parameter(log_sigma)

    def _update_sigma(self):
        self.sigma = self._log_sigma.exp()

    ''' ----------Neural network encoder methods---------- '''

    # Look into layer = torch.nn.Linear(nn_layers[i], nn_layers[i + 1]) and
    # nn.ModuleList([nn.Linear...
    # Haven't done this here as the parameters do not register.

    def _init_nn(self, d, q, nn_layers):
        n_hidden = len(nn_layers)
        mu_layers = (d,) + nn_layers + (q,)

        # tril_indices help convert from a flat vector to a lower-tri matrix
        self._tril_indices = torch.tril_indices(row=q, col=q, offset=0)
        sg_output_len = len(self._tril_indices.T)
        num_nodes_sigma = (d + sg_output_len)//2
        sg_layers = (d,) + (num_nodes_sigma,)*n_hidden + (sg_output_len,)

        def param(shape):
            return torch.nn.Parameter(
                float_tensor(np.random.uniform(size=shape)))

        max_iter = len(mu_layers) - 1
        self._nn = {}  # nn parameters
        for i in range(max_iter):
            # layer output = relu(w*input + b). Declare w and b here:
            _i = str(i)
            self._nn['mu_w' + _i] = param((mu_layers[i], mu_layers[i + 1]))
            self._nn['mu_b' + _i] = param(mu_layers[i + 1])
            self._nn['cov_w' + _i] = param((sg_layers[i], sg_layers[i + 1]))
            self._nn['cov_b' + _i] = param(sg_layers[i + 1])

            self.register_parameter('mu_w' + _i, self._nn['mu_w' + _i])
            self.register_parameter('mu_b' + _i, self._nn['mu_b' + _i])
            self.register_parameter('cov_w' + _i, self._nn['cov_w' + _i])
            self.register_parameter('cov_b' + _i, self._nn['cov_b' + _i])

    def _update_nn_params(self):
        max_iter = len(self._nn)//4
        pred_mu = self._Y
        pred_sg = self._Y
        for i in range(max_iter):
            w = self._nn['mu_w' + str(i)]
            b = self._nn['mu_b' + str(i)]
            non_lin = relu if i != (max_iter - 1) else lambda x: x
            pred_mu = non_lin(pred_mu@w + b)

            w = self._nn['cov_w' + str(i)]
            b = self._nn['cov_b' + str(i)]
            pred_sg = non_lin(pred_sg@w + b)
        self.mu = pred_mu

        n = len(pred_mu)
        q = len(pred_mu.T)

        # set diagonal to be positive
        diag_elems = ~np.logical_xor(
            self._tril_indices[0, :], self._tril_indices[1, :]).bool()
        pred_sg[:, diag_elems] = relu(pred_sg[:, diag_elems]) + 1e-10

        self.sigma = torch.zeros((n, q, q))
        self.sigma[:, self._tril_indices[0], self._tril_indices[1]] = pred_sg

        jitter = torch.eye(q).unsqueeze(0)*1e-4
        jitter = torch.cat([jitter for i in range(n)], axis=0)
        self.sigma += jitter
        # self.sigma is the cholesky factor of the covariance

    ''' ----------Sparse GP encoder methods---------- '''

    def _init_gp(self, num_latent, inducing_num):
        d = len(self._Y.T)
        self.gps = {}

        # one GP for every latent dimension
        for i in range(num_latent):
            X_inducing =\
                float_tensor(np.random.normal(size=(inducing_num, d)))
            kernel = gp.kernels.Matern52(
                input_dim=d,
                lengthscale=torch.ones(d))
            gp_model = gp.models.VariationalSparseGP(
                X=self._Y,
                y=None,
                kernel=kernel,
                Xu=X_inducing,
                likelihood=gp.likelihoods.Gaussian())

            # register gp parameters
            self.gps['model_' + str(i)] = gp_model
            self._register_module(gp_model, i)

    def _update_gp_params(self):
        n = len(self._Y)
        jitter = torch.eye(n).reshape(n, n)*1e-4

        # need to concatenate the different gps across latent dimensions
        gp_mu_sigmas =\
            [gp.forward(self._Y, full_cov=True) for _, gp in self.gps.items()]

        mu = [gp_mu_sigmas[i][0].reshape(-1, 1) for i in range(len(self.gps))]
        self.mu = torch.cat(mu, axis=1)

        # sigma_proper = ... matrix.cholesky().reshape(1, n, n) ... axis=0)
        sigmas = [gp_mu_sigmas[i][1] + jitter for i in range(len(self.gps))]
        sigmas = [matrix.diag().sqrt().reshape(-1, 1) for matrix in sigmas]
        sigma_fitc = torch.cat(sigmas, axis=1)
        self.sigma = sigma_fitc

    ''' ----------Internals---------- '''

    def _register_module(self, module, i=''):
        for (name, param) in module.named_parameters():
            self.register_parameter(name.replace('.', '_') + str(i), param)

    def use_grads(self, trainable=True):
        '''Freeze or unfreeze parameters.

        Parameters
        ----------
        trainable : bool
            True to unfreeze, False to freeze parameters.
        '''
        for param in self.parameters():
            param.requires_grad_(trainable)


# class _Transpose(dist.transforms.Transform):
#     def __init__(self):
#         super().__init__(cache_size=1)
#     def _call(self, X):
#         return X.transpose(-1, -2)
#     def _inverse(self, X):
#         return X.transpose(-1, -2)
#     def log_abs_det_jacobian(self, *args):
#         return torch.tensor(0.0)


class EncoderFlow(torch.nn.Module):
    ''' '''
    def __init__(self, num_latent, num_flows, base_params=None,
                 flow_type='planar', flows=None, activate_flow=True):
        '''Initiates the flow.

        You can manually provide a list of flows to use (each one from
        pyro.distributions.transforms). This is useful if you want to
        input tranforms like the exponential.

        If this isn't given, n_flows of flow_type are initialized.

        Parameters
        ----------
        num_latent : int
            Number of dimensions for the flow to act on.
        num_flows : int
            Number of flows.
        base_params : EncoderBase instance
        flow_type : str
            One of 'sylvester', 'planar' or 'radial'.
        flows : list
            List of relevant pyro.distributions.transforms.
        activate_flow : bool
            If False, the flow is set to the identity transform (not
            implemented for custom flows via the 'flows' argument).

        '''
        super().__init__()

        if flows is None:
            if flow_type == 'sylvester':
                sylvester = dist.transforms.Sylvester
                q = num_latent
                self.flows = [sylvester(q, q) for i in range(num_flows)]
            elif flow_type == 'planar':
                planar = dist.transforms.Planar
                self.flows = [planar(num_latent) for i in range(num_flows)]
            elif flow_type == 'radial':
                radial = dist.transforms.Radial
                self.flows = [radial(num_latent) for i in range(num_flows)]
            else:
                raise NotImplementedError('Flow type not implemented.')
        else:
            self.flows = flows

        if not activate_flow:
            self.deactivate_flow(num_latent)

        for i, flow in enumerate(self.flows):
            if hasattr(flow, 'parameters'):
                super().add_module('flow_module_' + str(i), flow)

        if base_params is not None:
            self.generate_base_dist(base_params)
        else:
            print('Remember to call `self.generate_base_dist(base_params)`.')

    def generate_base_dist(self, base_params):
        ''' Generate the transformed distribution.
        'base_params` must have the `model`, `mu` and `sigma` attributes.'''

        if base_params.model in ('mf', 'pca', 'gp'):
            self.base_dist = dist.Normal(base_params.mu, base_params.sigma)

        # elif base_params.model == 'gp':

            # Non FITC GP Approx:
            # mu = base_params.mu.T
            # sigma = base_params.sigma
            # MVN = dist.MultivariateNormal
            # X_latent = pyro.sample('X_latent', MVN(mu, scale_tril=sigma))
            # self.base_dist = dist.Normal(X_latent.T, 1e-4)

            # the proper way of doing this would be something like:
            # (from my expts and advice from the pyro forum)
            # mvn = dist.MultivariateNormal(mu, sigma).to_event(1)
            # _flows = [_Transpose(), dist.transforms.Planar(d), _Transpose()]
            # flow = dist.TransformedDistribution(mvn, _flows)

        elif base_params.model == 'nn':
            mu = base_params.mu
            sigma = base_params.sigma
            self.base_dist = dist.MultivariateNormal(mu, scale_tril=sigma)

        base = self.base_dist
        self.flow_dist = dist.TransformedDistribution(base, self.flows)

    def deactivate_flow(self, q):
        ''' Set flow to the identity function.'''
        loc = torch.zeros(q).float()
        scale = torch.ones(q).float()
        self.flows = [dist.transforms.AffineTransform(loc, scale)]

    def use_grads(self, trainable=True):
        '''Freeze or unfreeze parameters.

        Parameters
        ----------
        trainable : bool
            True to unfreeze, False to freeze parameters.
        '''
        for param in self.parameters():
            param.requires_grad_(trainable)
        if hasattr(self, 'flows'):
            for flow in self.flows:
                if hasattr(flow, 'parameters'):
                    for param in flow.parameters():
                        param.requires_grad_(trainable)

    def forward(self, X):
        '''Compute the flow function X_f = flow(X_b).'''
        for i in range(len(self.flows)):
            X = self.flows[i](X)
        return X

    def X_map(self, n_restarts=3, use_base_mu=False):
        '''Compute mode of the flow. This uses scipy because torch fails.'''

        if use_base_mu:
            Z = float_tensor(self.base_dist.loc.detach().numpy())
            return self.forward(Z).detach().float()

        torch.manual_seed(42)

        def loss(Z):
            X = self.forward(float_tensor(Z.reshape(shape)))
            lp = self.flow_dist.log_prob(X)
            return -lp.sum().detach().numpy()

        def d_loss(Z):
            return scipy.optimize.approx_fprime(Z, loss, 1e-3)

        optimize = scipy.optimize.lbfgsb.fmin_l_bfgs_b

        shape = tuple(self.base_dist.loc.shape)
        best_loss = np.inf

        for _ in range(n_restarts):
            Z_guess = self.base_dist().detach().numpy().reshape(-1)
            Z_opt, loss_cur, _ = optimize(loss, Z_guess, fprime=d_loss)

            if loss_cur < best_loss:
                best_loss = loss_cur
                Z_best = Z_opt

        torch.seed()
        Z = float_tensor(Z_best.reshape(shape))
        return self.forward(Z).detach().float()

    def plot_flow(self, n_grid=500, mu=torch.zeros(2), sg=torch.ones(2),
                  distb='norm'):
        '''Plot a flow transformed normal distribution.

        The viz is basically:
        flow_grid = flow.forward(uniform_grid)
        log_prob_on_flow_grid = flow.log_prob(flow_grid)
        heatmap(flow_grid[0], flow_grid[1], log_prob_on_flow_grid)

        Parameters
        ----------
        n_grid : int
            Resolution of the grid of the base distribution that the flow
            acts on.
        mu : torch.tensor
            Size (2,). Means of the normal distribution the flow acts on.
        sg : torch.tensor
            Size (2,). SD of the normal distribution the flow acts on.
        distb : str
            Base distribution, one of 'norm' or 'unif'.
        '''

        grid_base_dist = np.linspace(-5, 5, n_grid)
        grid_base_dist = np.vstack([np.repeat(grid_base_dist, n_grid),
                                    np.tile(grid_base_dist, n_grid)]).T

        if distb == 'norm':
            distb = dist.Normal(mu, sg)
        elif distb == 'unif':
            distb = dist.Uniform(-5.*sg + mu, 5*sg + mu)

        flow_dist = dist.TransformedDistribution(
            base_distribution=distb,
            transforms=self.flows)

        transformed_grid = self.forward(float_tensor(grid_base_dist))
        log_p = flow_dist.log_prob(transformed_grid).detach()

        # plt.figure()
        plt.hexbin(transformed_grid.detach()[:, 0],
                   transformed_grid.detach()[:, 1],
                   log_p.exp(), gridsize=300)


class MaskedGaussian(gp.likelihoods.Gaussian):
    def forward(self, f_loc, f_var, y=None):
        y_var = f_var + self.variance
        y_dist = dist.Normal(f_loc, y_var.sqrt())

        if y is not None:
            if y.isnan().any():
                y_dist = dist.MaskedDistribution(y_dist, ~y.isnan())
                y = torch.masked_fill(y, y.isnan(), -999.)

            y_dist =\
                y_dist.expand_by(y.shape[:-f_loc.dim()]).to_event(y.dim())
        return pyro.sample(self._pyro_get_fullname("y"), y_dist, obs=y)


class GPLVF:
    def __init__(self, Y, latent_dim):
        '''Initiates the model.

        Parameters
        ----------
        Y : torch.tensor
            Data, shape nxd.
        latent_dim : int
            Latent dimension.
        '''
        self._Y = Y
        self._n, self._d = Y.shape
        self._q = latent_dim

    def init_encoder(self, model='pca', num_flows=1, activate_flow=True,
                     flow_type='planar', nn_layers=(4, 2),
                     gp_inducing_num=10, flows=None):
        '''Initiates the base dist and flow classes.
        All parameters are passed to EncoderBase and EncoderFlow.

        Parameters
        ----------
        model : str
        num_flows : int
            Number of flows.
        activate_flow : bool
            If False, the flow is set to the identity transform (not
            implemented for custom flows via the 'flows' argument).
        flow_type : str
            One of 'sylvester', 'planar' or 'radial'.
        nn_layers : tuple
        gp_inducing_num : int
        flows : list
            List of relevant pyro.distributions.transforms.
        '''

        assert model in ('mf', 'pca', 'nn', 'gp')
        self.enc_base = EncoderBase(self._Y, self._q, model,
                                    nn_layers, gp_inducing_num)
        self.enc_flow = EncoderFlow(self._q, num_flows, self.enc_base,
                                    flow_type, flows, activate_flow)

    def _register_encoder_base(self):
        id = str(uuid4())
        for i in range(len(self.enc_flow.flows)):
            if hasattr(self.enc_flow.flows[i], 'parameters'):
                pyro.module(id + 'enc_flow_' + str(i), self.enc_flow.flows[i])
        pyro.module(id + 'enc_var_params', self.enc_base)
        # pyro.module(id + 'enc_flow_main', self.enc_flow)

    def encoder_model(self):
        '''To be used as a 'guide' for SVI.'''

        # register all encoder params
        self._register_encoder_base()
        return pyro.sample('X', self.enc_flow.flow_dist)

    def init_decoder(self, kernel=None, inducing_n=25, base_model='sgp'):
        '''Initiates the pyro forward/decoder gplvm.

        Parameters
        ----------
        kernel : pyro.contrib.gp.kernels instance
        inducing_n : int
            Number of inducing points for the decoder.
        base_model : str
            One of 'sgp' (SparseGPRegression) or 'vsgp' (VariationalSparseGP).
            If missing data is found, 'vsgp' will be selected.
        '''

        Y_for_init = self._Y.numpy().copy()
        Y_for_init[np.isnan(Y_for_init)] = 0.0
        X_init = float_tensor(PCA(self._q).fit_transform(Y_for_init))

        X_inducing =\
            float_tensor(np.random.normal(size=(inducing_n, self._q)))

        if kernel is None:
            kernel = gp.kernels.Matern52(
                input_dim=self._q,
                lengthscale=torch.ones(self._q))

        base_args = dict(
            X=X_init,
            y=self._Y.T,
            kernel=kernel,
            Xu=X_inducing,
            jitter=1e-4)

        if self._Y.isnan().any() or base_model == 'vsgp':
            base_args['likelihood'] = MaskedGaussian()
            gp_module = gp.models.VariationalSparseGP(**base_args)
        elif base_model == 'sgp':
            gp_module = gp.models.SparseGPRegression(**base_args)
        else:
            raise NotImplementedError('Unsupported base_model.')
        self.decoder = gp.models.GPLVM(gp_module)

    @property
    def decoder_model(self):
        '''To be used as the model with SVI.'''
        return self.decoder.model

    def update_parameters(self):
        self.enc_base.update_parameters()
        self.enc_flow.generate_base_dist(self.enc_base)

    def y_given_x(self, X, mu_only=True):
        '''This returns the parameters of distribution (Y_new | X_new, ...).

        Parameters
        ----------
        mu_only : bool
            Return the mean only. If False, both the mean and covariance
            are returned.
        '''
        print('Remember to set, for example gplvm.decoder.X = X_recon' +
              ' BEFORE calling this function.')

        mu, sigma = self.decoder.forward(Xnew=X, full_cov=True)
        if mu_only:
            return mu.detach().numpy().T
        else:
            return mu, sigma

    def use_grads(self, trainable=True):
        '''Freeze or unfreeze parameters.

        BUG: The train function needs to be run at least once so that
        all the necessary params are present when freezing.

        Parameters
        ----------
        trainable : bool
            True to unfreeze, False to freeze parameters.
        '''
        for param in self.decoder.parameters():
            param.requires_grad_(trainable)

    def train(self, steps=1000, lr=0.01, n_elbo_samples=3):
        self.update_parameters()

        svi = pyro.infer.SVI(
            model=self.decoder_model,
            guide=self.encoder_model,
            optim=pyro.optim.Adam(dict(lr=lr)),
            loss=pyro.infer.Trace_ELBO(n_elbo_samples, retain_graph=True)
        )

        losses = np.zeros(steps)
        bar = trange(steps, leave=False)
        for step in bar:
            self.update_parameters()
            losses[step] = svi.step()
            bar.set_description(str(int(losses[step])))
        return losses

    def predict(self, Y_test, mf_kern=None, mf_num_inducing=25, n_restarts=3,
                n_train_mf=2000, use_base_mu_mf=False):
        ''' The idea here is that you obtain the distribution of X_new
        using the encoder directly or a meanfield predict algorithm. '''
        raise NotImplementedError('MF has a bug that needs to be fixed.')
