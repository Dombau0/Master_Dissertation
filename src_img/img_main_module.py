import hydra

from omegaconf import DictConfig

from img_train_module import IMG_Train_Module


@hydra.main(version_base=None, config_path="../config", config_name="img_config")
def main(config: DictConfig):

    trainer = IMG_Train_Module(config)
    trainer.train()


if __name__ == "__main__":
    main()


