import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym.spaces import Discrete
from gym.spaces import Box

from lagom import BaseAgent

from lagom.envs import flatdim

from lagom.networks import Module
from lagom.networks import make_fc
from lagom.networks import ortho_init
from lagom.networks import CategoricalHead
from lagom.networks import DiagGaussianHead
from lagom.networks import StateValueHead
from lagom.networks import linear_lr_scheduler

from lagom.metric import get_bootstrapped_returns
from lagom.metric import get_gae

from lagom.transform import explained_variance as ev

from torch.utils.data import DataLoader
from dataset import Dataset


class MLP(Module):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.env = env
        self.device = device
        
        self.feature_layers = make_fc(flatdim(env.observation_space), config['nn.sizes'])
        for layer in self.feature_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(1/layer.out_features))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in config['nn.sizes']])
        
        self.to(self.device)
        
    def forward(self, x):
        for layer, layer_norm in zip(self.feature_layers, self.layer_norms):
            x = F.selu(layer_norm(layer(x)))
        return x


class Agent(BaseAgent):
    def __init__(self, config, env, device, **kwargs):
        super().__init__(config, env, device, **kwargs)
        
        if config['nn.recurrent']:
            pass
        else:
            self.feature_network = MLP(config, env, device, **kwargs)
        feature_dim = config['nn.sizes'][-1]
        
        if isinstance(env.action_space, Discrete):
            self.action_head = CategoricalHead(feature_dim, env.action_space.n, device, **kwargs)
        elif isinstance(env.action_space, Box):
            self.action_head = DiagGaussianHead(feature_dim, 
                                                flatdim(env.action_space), 
                                                device, 
                                                config['agent.std0'], 
                                                config['agent.std_style'], 
                                                config['agent.std_range'],
                                                config['agent.beta'], 
                                                **kwargs)
        self.V_head = StateValueHead(feature_dim, device, **kwargs)
        
        self.total_T = 0
        
        self.optimizer = optim.Adam(self.parameters(), lr=config['agent.lr'])
        if 'train.timestep' in config:
            N = config['train.timestep']
        elif 'train.iter' in config:
            N = config['train.iter']
        self.lr_scheduler = linear_lr_scheduler(self.optimizer, N, config['agent.min_lr'])
        
    def choose_action(self, obs, **kwargs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        out = {}
        features = self.feature_network(obs)
        
        action_dist = self.action_head(features)
        out['action_dist'] = action_dist
        out['entropy'] = action_dist.entropy()
        out['perplexity'] = action_dist.perplexity()
        
        action = action_dist.sample()
        out['action'] = action
        out['raw_action'] = action.detach().cpu().numpy()
        out['action_logprob'] = action_dist.log_prob(action.detach())
        
        # TODO: independent critic
        V = self.V_head(features)
        out['V'] = V
        
        # sanity check for NaN
        if torch.any(torch.isnan(action)):
            raise ValueError('NaN!')
        return out
    
    def learn_one_update(self, data):
        data = [d.to(self.device) for d in data]
        observations, old_actions, old_logprobs, old_entropies, old_Vs, old_Qs, old_As = data
        # TODO: independent critic
        out = self.choose_action(observations)
        logprobs = out['action_dist'].log_prob(old_actions).squeeze(-1)
        entropies = out['entropy'].squeeze(-1)
        Vs = out['V'].squeeze(-1)
        
        ratio = torch.exp(logprobs - old_logprobs)
        eps = self.config['agent.clip_range']
        policy_loss = -torch.min(ratio*old_As, 
                                 torch.clamp(ratio, 1.0 - eps, 1.0 + eps)*old_As)
        policy_loss = policy_loss.mean()
        entropy_loss = -entropies
        entropy_loss = entropy_loss.mean()
        clipped_Vs = old_Vs + torch.clamp(Vs - old_Vs, -eps, eps)
        value_loss = torch.max(F.mse_loss(Vs, old_Qs, reduction='none'), 
                               F.mse_loss(clipped_Vs, old_Qs, reduction='none'))
        value_loss = value_loss.mean()
        
        loss = policy_loss + self.config['agent.value_coef']*value_loss + self.config['agent.entropy_coef']*entropy_loss
        
        # TODO: independent critic
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config['agent.max_grad_norm'])
        if self.config['agent.use_lr_scheduler']:
            if 'train.timestep' in self.config:
                self.lr_scheduler.step(self.total_T)
            elif 'train.iter' in self.config:
                self.lr_scheduler.step()
        self.optimizer.step()
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['policy_entropy'] = -entropy_loss.item()
        out['value_loss'] = value_loss.item()
        out['explained_variance'] = ev(y_true=old_Qs.detach().cpu().numpy(), y_pred=Vs.detach().cpu().numpy())
        approx_kl = torch.mean(old_logprobs - logprobs)
        out['approx_kl'] = approx_kl.item()
        clip_frac = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)).float().mean()
        out['clip_frac'] = clip_frac.item()
        return out
        
    def learn(self, D, **kwargs):
        # mask-out values when its environment already terminated for episodes
        validity_masks = torch.from_numpy(D.batch_validity_masks).to(self.device)
        logprobs = torch.stack(D.get_batch_info('action_logprob'), 1).squeeze(-1)
        logprobs *= validity_masks
        entropies = torch.stack(D.get_batch_info('entropy'), 1).squeeze(-1)
        entropies *= validity_masks
        Vs = torch.stack(D.get_batch_info('V'), 1).squeeze(-1)
        Vs *= validity_masks
        with torch.no_grad():
            # TODO: independent critic 
            last_Vs = self.V_head(self.feature_network(torch.from_numpy(D.last_observations).to(self.device))).squeeze(-1)
        Qs = get_bootstrapped_returns(D, last_Vs, self.config['agent.gamma'])
        Qs = torch.from_numpy(Qs.copy()).to(self.device)
        if self.config['agent.standardize_Q']:
            Qs = (Qs - Qs.mean(1, keepdim=True))/(Qs.std(1, keepdim=True) + 1e-8)
        As = get_gae(D, Vs, last_Vs, self.config['agent.gamma'], self.config['agent.gae_lambda'])
        As = torch.from_numpy(As.copy()).to(self.device)
        if self.config['agent.standardize_adv']:
            As = (As - As.mean(1, keepdim=True))/(As.std(1, keepdim=True) + 1e-8)
        assert all([x.ndimension() == 2 for x in [logprobs, entropies, Vs, Qs, As]])
        
        dataset = Dataset(D, logprobs, entropies, Vs, Qs, As)
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.config['cuda'] else {}
        dataloader = DataLoader(dataset, self.config['train.batch_size'], shuffle=True, **kwargs)
        for epoch in range(self.config['train.num_epochs']):
            logs = [self.learn_one_update(data) for data in dataloader]
            
            approx_kl = np.mean([item['approx_kl'] for item in logs])
            if approx_kl > self.config['agent.target_kl']:
                break
        
        self.total_T += sum([sum(Ts) for Ts in D.Ts])
        out = {}
        if self.config['agent.use_lr_scheduler']:
            out['current_lr'] = self.lr_scheduler.get_lr()
        out['loss'] = np.mean([item['loss'] for item in logs])
        out['policy_loss'] = np.mean([item['policy_loss'] for item in logs])
        out['entropy_loss'] = np.mean([item['entropy_loss'] for item in logs])
        out['policy_entropy'] = np.mean([item['policy_entropy'] for item in logs])
        out['value_loss'] = np.mean([item['value_loss'] for item in logs])
        out['explained_variance'] = np.mean([item['explained_variance'] for item in logs])
        out['approx_kl'] = np.mean([item['approx_kl'] for item in logs])
        out['clip_frac'] = np.mean([item['clip_frac'] for item in logs])
        out['finished_inner_epochs'] = f'{epoch+1}/{self.config["train.num_epochs"]}'
        return out
