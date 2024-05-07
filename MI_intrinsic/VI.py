import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt
from torchsummary import summary
from statistics import median
from torchhk import transform_model


def get_all_thetas(model):
    ret = []
    for layers in model:
        if isinstance(layers,(bnn.BayesLinear, bnn.BayesConv2d)):
            ret.append(layers)
    return ret

def copy_theta(sets):
    ret = []
    for layers in sets:
        w_mu = torch.clone(layers.weight_mu)
        w_log_sigma = torch.clone(layers.weight_log_sigma)
        b_mu = torch.clone(layers.bias_mu)
        b_log_sigma = torch.clone(layers.bias_log_sigma)
        ret.append({'w_mu':w_mu, 'w_log_sigma':w_log_sigma, 'b_mu':b_mu,'b_log_sigma':b_log_sigma})
    return ret

def log_to_std(rho):
    return torch.exp(rho)

def compute_KL_Gaussians(sets1, sets2):
    length = len(sets1)
    total = 0
    param_num = 0
    for i in range(length):
        w_mu1 = sets1[i]['w_mu']
        w_mu2 = sets2[i]['w_mu']
        w_sigma1 = log_to_std(sets1[i]['w_log_sigma'])
        w_sigma2 = log_to_std(sets2[i]['w_log_sigma'])
        b_mu1 = sets1[i]['b_mu']
        b_mu2 = sets2[i]['b_mu']
        b_sigma1 = log_to_std(sets1[i]['b_log_sigma'])
        b_sigma2 = log_to_std(sets2[i]['b_log_sigma'])
        param_num += w_mu1.shape[0] * w_mu1.shape[1] + w_sigma1.shape[0] * w_sigma1.shape[1] \
                        + b_mu1.shape[0] + b_sigma1.shape[0]
        local = 1/2 * (torch.sum(torch.square(w_sigma1/w_sigma2)) + torch.sum(torch.square(b_sigma1/b_sigma2)) \
            + torch.sum(2*torch.log(w_sigma2 /w_sigma1)) + torch.sum(2*torch.log(b_sigma2/b_sigma1))\
            + torch.sum(torch.square(w_mu1 - w_mu2)/torch.square(w_sigma2))  + torch.sum(torch.square(b_mu1 - b_mu2)/torch.square(b_sigma2)))
        total += local
    total = total/2 - param_num/2
    return total


def KL_preprocess(model,inputs, outputs):
    inputs = inputs.float()
    outputs = outputs.float()
    pre = model(inputs)
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    kl_weight = 0.01
    kl = kl_loss(model)
    ce = mse_loss(pre, outputs)
    cost = ce + kl_weight*kl
    optimizer.zero_grad()
    cost.backward()
    
def fisher_information_KL_approximate(all_layers):

    approximate = 0

    for layers in all_layers:
        grad_w_mu = torch.flatten(layers.weight_mu.grad)
        grad_w_log_sigma = torch.flatten(layers.weight_log_sigma.grad)
        grad_b_mu = torch.flatten(layers.bias_mu.grad)
        grad_b_log_sigma = torch.flatten(layers.bias_log_sigma.grad)

        hessian_w_mu = 1/torch.square(torch.flatten(torch.exp(layers.weight_log_sigma)))
        hessian_w_sigma = 1/2 * torch.square(hessian_w_mu)
        hessian_b_mu = 1/torch.square(torch.flatten(torch.exp(layers.bias_log_sigma)))
        hessian_b_sigma = 1/2 * torch.square(hessian_b_mu)

        l = 1
        approximate += 1/2 * l * l * torch.sum(grad_b_mu  * grad_b_mu * hessian_b_mu).item()\
                        + 1/2 * l * l * torch.sum(grad_w_mu  * grad_w_mu * hessian_w_mu).item()\
                        + 1/2 * l * l * torch.sum(grad_b_log_sigma  * grad_b_log_sigma * hessian_b_sigma).item()\
                        + 1/2 * l * l * torch.sum(grad_w_log_sigma  * grad_w_log_sigma * hessian_w_sigma).item()
    return approximate

def speedy_fisher(all_layers):
   
    approximate = torch.tensor(0.0, device=all_layers[0].weight_mu.device)

    l = torch.tensor(0.001, device=all_layers[0].weight_mu.device)
    factor = 1/2 * l * l * 1/1550 * 1/2500 * 1/1550 * 1/8 # hardcode the # of weights in BNN

    for layers in all_layers:
        grad_w_mu = torch.flatten(layers.weight_mu.grad)
        grad_w_log_sigma = torch.flatten(layers.weight_log_sigma.grad)
        grad_b_mu = torch.flatten(layers.bias_mu.grad)
        grad_b_log_sigma = torch.flatten(layers.bias_log_sigma.grad)

        hessian_w_mu = 1/torch.square(torch.flatten(torch.exp(layers.weight_log_sigma)))
        hessian_w_sigma = 1/2 * torch.square(hessian_w_mu)
        hessian_b_mu = 1/torch.square(torch.flatten(torch.exp(layers.bias_log_sigma)))
        hessian_b_sigma = 1/2 * torch.square(hessian_b_mu)

        # Inplace operations
    
        grad_b_mu.mul_(hessian_b_mu).mul_(grad_b_mu)
        grad_w_mu.mul_(hessian_w_mu).mul_(grad_w_mu)
        grad_b_log_sigma.mul_(hessian_b_sigma).mul_(grad_b_log_sigma)
        grad_w_log_sigma.mul_(hessian_w_sigma).mul_(grad_w_log_sigma)

        approximate.add_(factor * torch.sum(grad_b_mu).item())
        approximate.add_(factor * torch.sum(grad_w_mu).item())
        approximate.add_(factor * torch.sum(grad_b_log_sigma).item())
        approximate.add_(factor * torch.sum(grad_w_log_sigma).item())

    return approximate.item()

def build_network(in_dim,out_dim,device):
    model = nn.Sequential(
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=in_dim, out_features=50),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=50, out_features=50),
        nn.ReLU(),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=50, out_features=out_dim),
    )


    return model.to(device=device)

def compute(model,buffer,train_size = 10, device = 'cpu'):
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    kl_weight = 0.01
    batches_num = 1

    lastce = 0
    for _ in range(batches_num):
        samples = buffer.random_batch(400)
        for _ in range(train_size):
            pre = model(torch.from_numpy(samples['observations']).float().to(device=device))
            ce = mse_loss(pre, torch.from_numpy(samples['next_observations']).float().to(device=device))
            kl = kl_loss(model)
            lastce = kl
            cost = ce + kl_weight*kl
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    print('LAST MODEL_CE\n', lastce)

def determine_vime_reward(data,device):
    uav_idx, bnn_model, current_obs, next_obs, trajectory_median = data
    bnn_model = bnn_model.to(device=device)
    current_obs = current_obs.to(device=device)
    next_obs = next_obs.to(device=device)
    KL_preprocess(bnn_model, current_obs, next_obs)
    KL = speedy_fisher(get_all_thetas(bnn_model))
    if len(trajectory_median[uav_idx]) == 0:
        reward = min(KL,1)
    else:
        reward = KL/median(trajectory_median[uav_idx])
    trajectory_median[uav_idx].append(KL)
    if len(trajectory_median[uav_idx]) > 40:
        trajectory_median[uav_idx].pop(0)

    return reward, trajectory_median

def determine_vime_reward_batch(data):
    uav_idx, bnn_model, current_obs, next_obs = data
    KL_preprocess(bnn_model, current_obs, next_obs)
    KL = speedy_fisher(get_all_thetas(bnn_model))
    trajectory_median = [[] for _ in range(current_obs.shape[0])]


    if len(trajectory_median[uav_idx]) == 0:
        reward = min(KL,1)
    else:
        reward = KL/median(trajectory_median[uav_idx])

    trajectory_median[uav_idx].append(KL)
    if len(trajectory_median[uav_idx]) > 40:
        trajectory_median[uav_idx].pop(0)

    return reward, trajectory_median


def convert_to_mean_nn(model):
    # Recall that we assume independence between each nodes, hence by taking the mean for each node, we 
    # essentially get the mean predcition of this BNN(linear combinations!), in the sense of the parameter space.
    nn_model = transform_model(model, bnn.BayesLinear,nn.Linear,args={"in_features" : ".in_features", "out_features" : ".out_features",
                  "bias":".bias"
                 }, 
            attrs={"weight" : ".weight_mu"}, inplace=False)
    return nn_model

def predict_with_mean_nn(model, obs):
    return model(obs)

class SimpleReplayPool(object):
    """Replay pool"""

    def __init__(
            self, max_pool_size, observation_shape, action_dim,
            observation_dtype=np.float32,  # @UndefinedVariable
            action_dtype=np.float32):  # @UndefinedVariable
        self._observation_shape = observation_shape
        self._action_dim = action_dim
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._max_pool_size = max_pool_size

        self._observations = np.zeros(
            (max_pool_size,) + observation_shape,
            dtype=observation_dtype
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
            dtype=action_dtype
        )
        self._rewards = np.zeros(max_pool_size, dtype='float32')
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal):
        self._observations[self._top] = observation
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size = self._size + 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(
                self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    def mean_obs_act(self):
        if self._size >= self._max_pool_size:
            obs = self._observations
            act = self._actions
        else:
            obs = self._observations[:self._top + 1]
            act = self._actions[:self._top + 1]
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0)
        act_mean = np.mean(act, axis=0)
        act_std = np.std(act, axis=0)
        return obs_mean, obs_std, act_mean, act_std

    @property
    def size(self):
        return self._size


if __name__ == "__main__":
    '''for test cases'''
    pool = SimpleReplayPool(100,(10,3),28)
    for i in range(50):
        pool.add_sample(np.zeros([10,3]),9,9,1)
    print(pool.random_batch(7)['observations'].shape)
    print(pool.size)


