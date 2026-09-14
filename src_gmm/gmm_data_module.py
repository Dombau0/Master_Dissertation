import torch
import numpy as np
import torch.distributions as dist

from torch.utils.data import DataLoader
from omegaconf import DictConfig


class GMM_Data_Module():

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.device = None

        self.helper = self._initialize_helper()
        self.initial = self._initialize_initial()
        self.target = self._initialize_target()

        self.current_sample = self.sample_initial(self.config.train_size)

    
    def set_training_device(self, device):
        self.device = device


    def _initialize_helper(self):
        return dist.MultivariateNormal(
            loc=torch.zeros(self.config.dims, device=self.device),
            covariance_matrix=torch.eye(self.config.dims, device=self.device) * self.config.helper_var
        )
    

    def _initialize_initial(self):
        return dist.MultivariateNormal(
            loc=torch.zeros(self.config.dims, device=self.device),
            covariance_matrix=torch.eye(self.config.dims, device=self.device) * self.config.initial_var
        )
    

    def _initialize_target(self):
        if self.config.gmm_comps == 1:
            return dist.MultivariateNormal(
                loc=torch.randn(self.config.dims, device=self.device),
                covariance_matrix=torch.from_numpy(self._generate_spd_covariance()).float()
            )
        else:
            cat = dist.Categorical(torch.ones(self.config.gmm_comps, device=self.device))
            comp = dist.Independent(
                base_distribution=dist.Normal(
                    loc=torch.empty(
                        self.config.gmm_comps, 
                        self.config.dims,
                        device=self.device
                    ).uniform_(self.config.target_bound[0], self.config.target_bound[1]),
                    scale=torch.ones(self.config.gmm_comps, self.config.dims, device=self.device)), 
                reinterpreted_batch_ndims=1
            )
            return dist.MixtureSameFamily(cat, comp)

    
    def train_data_loader(self):
        return DataLoader(
            dataset=self.current_sample,
            batch_size=self.config.batch_size,
            shuffle=True
        )
    
    
    def update_current_sample(self, new_current_sample: torch.Tensor):
        self.current_sample = new_current_sample.to(self.device)


    def update_helper(self, new_current_sample: torch.Tensor):
        helper_mean = new_current_sample.mean(axis=0)
        helper_cov = torch.cov(new_current_sample.T)
        self.helper = dist.MultivariateNormal(helper_mean, helper_cov)


    def sample_helper(self, n: int):
        return self.helper.sample((n, )).to(self.device)
    
    
    def sample_initial(self, n: int):
        return self.initial.sample((n, )).to(self.device)
    
    
    def sample_target(self, n: int):
        return self.target.sample((n, )).to(self.device)
    

    def log_prob_helper(self, data: torch.Tensor):
        return self.helper.log_prob(data)
    

    def log_prob_initial(self, data: torch.Tensor):
        return self.initial.log_prob(data)
    

    def log_prob_target(self, data: torch.Tensor):
        return self.target.log_prob(data)
    

    def _generate_spd_covariance(self):
        a = np.random.rand(self.config.dims, self.config.dims)
        u, _, v = np.linalg.svd(np.dot(a.T, a))
        b = np.diag(self.config.eig_val_range[0] + np.random.rand(self.config.dims) 
                    * (self.config.eig_val_range[1] - self.config.eig_val_range[0]))
        return np.dot(np.dot(u, b), v)


