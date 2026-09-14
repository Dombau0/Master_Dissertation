import torch

from omegaconf import DictConfig
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class IMG_Data_Module():

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.train_data, self.test_data = self._load_data()
        self.current_data = self.sample_initial()


    def _load_data(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor()
        ])

        train = MNIST(
            root="./data",
            train=True,
            transform=transform,
            download=True
        )

        test = MNIST(
            root="./data",
            train=False,
            transform=transform,
            download=True
        )

        return train, test
    

    def sample_initial(self):
        initial_data = torch.rand((2 * self.config.samples, self.config.channel, self.config.image_size, self.config.image_size))

        initial_data = self.data_preprocessing(initial_data)

        return initial_data
    

    def data_preprocessing(self, data: torch.Tensor):
        if self.config.uniform_noise:
            data = data / 256.0 * 255.0 + torch.rand_like(data) / 256.0
        elif self.config.gaussian_noise:
            data = data + torch.randn_like(data) * 0.01

        if self.config.center:
            data = 2.0 * data - 1.0
        elif self.config.logit:
            data = 1e-6 + (1.0 - 2.0 * 1e-6) * data
            data = torch.log(data) - torch.log1p(- data)
        
        if self.config.mean is not None and self.config.deviation is not None:
            data = (data - torch.FloatTensor(self.config.mean)[:, None, None]) / torch.FloatTensor(self.config.deviation)[:, None, None]

        return data
    

    def data_postprocessing(self, data: torch.Tensor):
        if self.config.mean is not None and self.config.deviation is not None:
            data = data * torch.FloatTensor(self.config.deviation)[:, None, None] + torch.FloatTensor(self.config.mean)[:, None, None] 
        
        if self.config.center:
            data = (data + 1.0) / 2.0
        elif self.config.logit:
            data = torch.sigmoid(data)

        return data
    

    def shuffle_current_data(self):
        self.current_data = self.current_data[torch.randperm(self.current_data.shape[0]), :, :, :]

    
    def train_data_loader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True
        )
    

    def test_data_loader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.config.batch_size,
            shuffle=False
        )
    

    def current_data_loader(self, index: int):
        if index is not None:
            start_index = index * self.config.batch_size
            return self.current_data[start_index:(start_index + self.config.batch_size)]
        else:
            return self.current_data[torch.randperm(self.current_data.shape[0])[:(2 * self.config.batch_size)], :, :, :]
        

    # add to IMG_Data_Module for the image saving process
    def sample_new_initial(self, num_samples: int):
        data = torch.rand((num_samples, self.config.channel, self.config.image_size, self.config.image_size))
        return self.data_preprocessing(data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from omegaconf import OmegaConf

    config = OmegaConf.create({
        "channel": 1,
        "image_size": 32,
        "batch_size": 64,
        "samples": 60000,
        "uniform_noise": False,
        "gaussian_noise": False,
        "center": True,
        "logit": False,
        "mean": None,
        "deviation": None,
    })

    dm = IMG_Data_Module(config)

    image, label = dm.train_data[0]

    processed = dm.data_preprocessing(image.clone())

    print(processed[:, :, 10])
    print(processed.shape)
    print(dm.current_data.shape)
    print(dm.current_data.shape[0])
    print(dm.current_data[0, 0, :, 10])

    # Convert back for visualization
    processed_display = dm.data_postprocessing(dm.current_data[0, :, :, :].clone())

    fig, ax = plt.subplots(1, 2, figsize=(8,4))

    ax[0].imshow(image.squeeze(), cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(processed_display.squeeze(), cmap="gray")
    ax[1].set_title("Preprocessed")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    plt.imsave("original.png", image.squeeze().numpy(), cmap="gray")
    plt.imsave("preprocessed.png", processed_display.squeeze().numpy(), cmap="gray")


