import copy
import torch
import torch.optim as optim

from pathlib import Path
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR

from gmm_data_module import GMM_Data_Module
from gmm_network_module import RMLP
from gmm_logger_module import GMM_Logger_Module

import gmm_utils_module as utils


class GMM_Train_Module():

    def __init__(self, config: DictConfig):
        super().__init__()        
        self.config = config

        self.relevant_variables = ["alpha", "crank_nicolson", "device", "dims", "seed"]
        self.output_parameters = {k: getattr(config, k) for k in self.relevant_variables}

        self.seed = utils.set_global_seed(self.config.seed)
        self.device = utils.set_training_device(self.config.device)
        self.directory = utils.set_output_directory(self.config.output_directory, self.output_parameters)

        self.data_module = GMM_Data_Module(self.config.data_module)
        self.logger_module = GMM_Logger_Module(self.config.logger_module)
        self.transport_net = RMLP(self.config.transport_net).to(self.device)
        self.dual_net = RMLP(self.config.dual_net).to(self.device)

        self.data_module.set_training_device(self.device)
        self.logger_module.set_output_directory(self.directory)

        self._initialize_optimizers()
        self._initialize_schedulers()

        self.last_dual_net = None
        self.last_helper = None       

        self.transport_net.train()
        self.dual_net.train()

        if self.config.verbose:
            print(self.directory)


    def _initialize_optimizers(self):
        self.transport_optim = optim.Adam(
            self.transport_net.parameters(),
            lr=self.config.transport_learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.dual_optim = optim.Adam(
            self.dual_net.parameters(),
            lr=self.config.dual_learning_rate,
            weight_decay=self.config.weight_decay
        )

    
    def _initialize_schedulers(self):
        self.transport_scheduler = StepLR(
            optimizer=self.transport_optim,
            step_size=self.config.transport_step_size,
            gamma=self.config.transport_gamma
        )

        self.dual_scheduler = StepLR(
            optimizer=self.dual_optim,
            step_size=self.config.dual_step_size,
            gamma=self.config.dual_gamma
        )


    def train(self):
        with utils.preserve_rng_state():
            self._log_initial_step()

        for iteration in range(1, self.config.jko_steps + 1):

            self.logger_module.start_timer()
            dual_loss_sum, dual_loss_count = 0.0, 0
            transport_loss_sum, transport_loss_count = 0.0, 0

            for _ in range(self.config.epochs_per_jko_step):
                for sample_data in self.data_module.train_data_loader():

                    sample_data = sample_data.to(self.device)
                    helper_data = self.data_module.sample_helper(n=sample_data.shape[0])

                    for _ in range(self.config.dual_steps):
                        self.dual_optim.zero_grad()
                        dual_loss = self.dual_loss(sample_data=sample_data, helper_data=helper_data)
                        dual_loss.backward()
                        self.dual_optim.step()
                        dual_loss_sum += dual_loss.item()
                        dual_loss_count += 1

                    for _ in range(self.config.primal_steps):
                        self.transport_optim.zero_grad()
                        transport_loss = self.transport_loss(iteration=iteration, sample_data=sample_data)
                        transport_loss.backward()
                        self.transport_optim.step()
                        transport_loss_sum += transport_loss.item()
                        transport_loss_count += 1

            self.transport_net.eval()
            self.dual_net.eval()

            if self.config.crank_nicolson:
                self.last_helper = copy.deepcopy(self.data_module.helper)
                self.last_dual_net = copy.deepcopy(self.dual_net)
                for p in self.last_dual_net.parameters():
                    p.requires_grad = False

            # THIS NEEDS TO BE CLEANED AT SOME POINT
            sample_data = self.data_module.current_sample
            helper_data = self.data_module.sample_helper(self.config.data_module.train_size)
            target_data = self.data_module.sample_target(self.config.data_module.train_size)

            if iteration % 5 == 0:
                print(sample_data.mean(0))
                print(target_data.mean(0))
                print(torch.diag(torch.cov(sample_data.T)))
                print(torch.diag(torch.cov(target_data.T)))
                cov_sample = torch.cov(sample_data.T)
                cov_target = torch.cov(target_data.T)
                print(torch.norm(cov_sample - cov_target))
                print(torch.linalg.eigvalsh(cov_sample).cpu())
                print(torch.linalg.eigvalsh(cov_target).cpu())

            kl = self.estimate_variational_kl(sample_data, helper_data)
            print(f"JKO {iteration}: estimated KL = {kl.item():.5f}")
            # END OF WHAT STILL NEEDS TO BE CLEANED INCLUDING KL METHOD

            with torch.no_grad():
                transport_net_fixed = copy.deepcopy(self.transport_net)
                transport_cost = (sample_data - transport_net_fixed(sample_data)).pow(2).sum(-1).mean().item()

            self.logger_module.log_networks(self.transport_net, self.dual_net, iteration)

            new_current_sample = self.pushforward_samples(transport_net_fixed, iteration)
            self.data_module.update_current_sample(new_current_sample=new_current_sample)
            self.data_module.update_helper(new_current_sample=new_current_sample)            
            
            self.logger_module.log_data(self.data_module.current_sample, iteration)

            self.logger_module.log_step(
                iteration=iteration,
                current_sample=new_current_sample,
                target_sample=target_data,
                target_log_prob_fn=self.data_module.log_prob_target,
                estimated_kl=kl.item(),
                mean_dual_loss=dual_loss_sum / max(dual_loss_count, 1),
                mean_transport_loss=transport_loss_sum / max(transport_loss_count, 1),
                transport_cost=transport_cost,
            )

            self.transport_scheduler.step()
            self.dual_scheduler.step()

            self.transport_net.train()
            self.dual_net.train()

        self.logger_module.save_logs(str(self.directory) + "/metrics.csv")

        self.logger_module.save_plots(
            target_samples=target_data,
            current_sample_dir=str(self.directory),
            num_iterations=self.config.jko_steps
        )

        self.logger_module.delete_things()


    def print_step_results(self):
        pass


    def _log_initial_step(self):
        self.logger_module.start_timer()

        self.transport_net.eval()
        self.dual_net.eval()

        sample_data = self.data_module.current_sample
        helper_data = self.data_module.sample_helper(self.config.data_module.train_size)
        target_data = self.data_module.sample_target(self.config.data_module.train_size)

        kl = self.estimate_variational_kl(sample_data, helper_data)

        with torch.no_grad():
            transport_cost = (sample_data - self.transport_net(sample_data)).pow(2).sum(-1).mean().item()

        self.logger_module.log_data(sample_data, 0)

        self.logger_module.log_step(
            iteration=0,
            current_sample=sample_data,
            target_sample=target_data,
            target_log_prob_fn=self.data_module.log_prob_target,
            estimated_kl=kl.item(),
            mean_dual_loss=float("nan"),
            mean_transport_loss=float("nan"),
            transport_cost=transport_cost,
        )

        self.transport_net.train()
        self.dual_net.train()

    
    def estimate_variational_kl(self, sample_data, helper_data):
        with torch.no_grad():
            transported = self.transport_net(sample_data)
            
            coupling = self.coupling_loss(transported)

            helper = self.data_module.log_prob_helper(data=transported).mean()

            target = - self.data_module.log_prob_target(data=transported).mean()

            conjugate = self.conjugate_loss(helper_data)

            wasserstein = (sample_data - transported).pow(2).flatten(start_dim=1).sum(-1).mean() / (2 * self.config.alpha)

            print("coupling: " + str(coupling) + ", helper: " + str(helper) + ", target: " + str(target) + ", conjugate: " + str(-conjugate))
            print("dual loss: " + str(- (coupling - conjugate)) + " transport loss: " + str(wasserstein + coupling + helper + target), " wasserstein loss: " + str(wasserstein))

            return 1 + coupling + helper + target - conjugate
    

    def dual_loss(self, sample_data, helper_data):
        with torch.no_grad():
            transported_samples = self.transport_net(sample_data).detach()

        coupling_loss = self.coupling_loss(transported_samples=transported_samples)
        conjugate_loss = self.conjugate_loss(helper_data=helper_data)

        return - (coupling_loss - conjugate_loss)


    def transport_loss(self, iteration, sample_data):
        transported_samples = self.transport_net(sample_data)

        if iteration < self.config.alpha_step_size:
            wasserstein_loss = (sample_data - transported_samples).pow(2).flatten(start_dim=1).sum(-1).mean() / (2 * self.config.alpha)
        else:
            wasserstein_loss = (sample_data - transported_samples).pow(2).flatten(start_dim=1).sum(-1).mean() / (2 * self.config.alpha * self.config.alpha_gamma)

        for param in self.dual_net.parameters():
            param.requires_grad = False
        
        coupling_loss = self.coupling_loss(transported_samples=transported_samples)
        helper_loss = self.data_module.log_prob_helper(transported_samples).mean()
        target_loss = - self.data_module.log_prob_target(transported_samples).mean()
        
        for param in self.dual_net.parameters():
            param.requires_grad = True

        if not self.config.crank_nicolson:
            return wasserstein_loss + coupling_loss + helper_loss + target_loss
        else:
            crank_nicolson_loss = self.crank_nicolson_loss(iteration=iteration, transported_samples=transported_samples)
            return wasserstein_loss + 0.5 * coupling_loss + 0.5 * helper_loss + target_loss + crank_nicolson_loss
    

    def coupling_loss(self, transported_samples):
        if self.config.ratio:
            return self.dual_net(transported_samples).mean()
        elif self.config.smooth:
            return torch.clamp((self.dual_net(transported_samples)).exp() - 1, min=1e-5).log().mean()
        else:
            return torch.clamp(self.dual_net(transported_samples), min=1e-10).log().mean()
        

    def conjugate_loss(self, helper_data):
        if self.config.ratio or self.config.smooth:
            return self.dual_net(helper_data).exp().mean()
        else:
            return self.dual_net(helper_data).mean()
        

    # def crank_nicolson_loss(self, iteration, transported_samples):
    #     if iteration == 1:
    #         crank_nicolson_loss = self.data_module.log_prob_initial(transported_samples).mean()
    #     else:
    #         crank_nicolson_loss = self.last_dual_net(transported_samples).mean() + self.last_helper.log_prob(transported_samples).mean()

    #     return 0.5 * crank_nicolson_loss


    def log_ratio(self, net, x):
        if self.config.ratio:
            return net(x)
        elif self.config.smooth:
            return torch.clamp(net(x).exp() - 1, min=1e-5).log()
        else:
            return torch.clamp(net(x), min=1e-10).log()

    def crank_nicolson_loss(self, iteration, transported_samples):
        if iteration == 1:
            cn = self.data_module.log_prob_initial(transported_samples).mean()
        else:
            cn = self.log_ratio(self.last_dual_net, transported_samples).mean() + self.last_helper.log_prob(transported_samples).mean()
        return 0.5 * cn
        
    
    def pushforward_samples(self, transport_net, iteration):
        samples = self.data_module.sample_initial(self.config.data_module.train_size)

        for k in range(1, iteration + 1):
            try:
                path = Path(self.directory) / "transport_net" / f"transport_net_{k}.pth"
                state_dict = torch.load(path, map_location=self.device)
            except FileNotFoundError:
                print("There has been a problem loading the current transport map.")
            
            transport_net.load_state_dict(state_dict)

            with torch.no_grad():
                samples = transport_net(samples)
        
        return samples   
    

