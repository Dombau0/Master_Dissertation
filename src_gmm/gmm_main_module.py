import hydra

from omegaconf import DictConfig

from gmm_train_module import GMM_Train_Module


@hydra.main(version_base=None, config_path="../config", config_name="gmm_config")
def main(config: DictConfig):

    trainer = GMM_Train_Module(config)
    trainer.train()


if __name__ == "__main__":
    main()


