import time
import copy
import shutil
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image

from img_data_module import IMG_Data_Module
from img_network_module import Unet, Discriminator

import img_utils_module as utils

try:
    from google.colab import files as _colab_files
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False


class IMG_Train_Module():

    def __init__(self, config: DictConfig):
        super().__init__()
        utils.seed_everything(0)
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Local disk (not Drive) — reliable target for repeated writes during training.
        self.local_out_dir = Path("/content/out_img")
        # Best-effort Drive backup destination; override via config.drive_backup_dir if set.
        self.drive_backup_dir = Path(
            self.config.get("drive_backup_dir", "/content/drive/MyDrive/Thesis/out_img_backup")
        )
        self._in_colab = _IN_COLAB

        self.data_module = IMG_Data_Module(self.config.data_module)
        self.transport_net = Unet(dim=32, out_dim=None, dim_mults=(1, 2, 4, 8), in_channel=1).to(self.device)
        self.dual_net = Discriminator(df_dim=128, in_channel=1, d_spectral_norm=True, activation=nn.PReLU()).to(self.device)

        self.transport_net.apply(self._weight_initialization_function)
        self.dual_net.apply(self._weight_initialization_function)

        self.transport_net.train()
        self.dual_net.train()

        self.transport_optim = optim.Adam(
            self.transport_net.parameters(),
            lr=self.config.transport_learning_rate,
            betas=(self.config.beta_one, self.config.beta_two),
            weight_decay=self.config.weight_decay
        )

        self.dual_optim = optim.Adam(
            self.dual_net.parameters(),
            lr=self.config.dual_learning_rate,
            betas=(self.config.beta_one, self.config.beta_two),
            weight_decay=self.config.weight_decay
        )

        self.current_dual_steps = 0
        self.dual_net_unstable = 0
        self.transport_net_unstable = 0
        self.skip_dual_optim = False

        self.last_dual_net = None
        self.last_cn_loss_value = None

        print("start")


    def train(self):
        for k in range(1, self.config.jko_steps + 1):

            for epoch_idx in range(self.config.epochs_per_jko_step):

                self.current_dual_steps = 0
                epoch_start = time.time()

                dual_loss_sum = 0.0
                dual_loss_count = 0
                transport_loss_sum = 0.0
                transport_loss_count = 0
                cn_loss_sum = 0.0
                cn_loss_count = 0
                skip_dual_epoch_count = 0
                batch_count = 0

                for sample_images in self.data_module.train_data_loader():

                    sample_images, _ = sample_images

                    sample_images = self.data_module.data_preprocessing(sample_images).to(self.device)

                    batch_count += 1
                    if self.skip_dual_optim:
                        skip_dual_epoch_count += 1

                    if not self.skip_dual_optim:
                        self.dual_optim.zero_grad()
                        dual_loss = self.dual_loss(sample_images=sample_images)
                        dual_loss.backward()
                        nn.utils.clip_grad_value_(parameters=self.dual_net.parameters(), clip_value=1.0)
                        self.dual_optim.step()
                        self.current_dual_steps += 1

                        dual_loss_sum += dual_loss.item()
                        dual_loss_count += 1

                    if self.skip_dual_optim or self.current_dual_steps % self.config.dual_steps_per_transport == 0:
                        self.transport_optim.zero_grad()
                        transport_loss = self.transport_loss()
                        transport_loss.backward()
                        nn.utils.clip_grad_value_(parameters=self.transport_net.parameters(), clip_value=1.0)
                        self.transport_optim.step()

                        transport_loss_sum += transport_loss.item()
                        transport_loss_count += 1

                        if self.last_cn_loss_value is not None:
                            cn_loss_sum += self.last_cn_loss_value
                            cn_loss_count += 1

                self.data_module.shuffle_current_data()

                epoch_time = time.time() - epoch_start
                avg_dual_loss = dual_loss_sum / dual_loss_count if dual_loss_count > 0 else float("nan")
                avg_transport_loss = transport_loss_sum / transport_loss_count if transport_loss_count > 0 else float("nan")
                avg_cn_loss = cn_loss_sum / cn_loss_count if cn_loss_count > 0 else float("nan")

                print(
                    f"Finished epoch {epoch_idx + 1}/{self.config.epochs_per_jko_step} "
                    f"(JKO step {k}) | time: {epoch_time:.2f}s | "
                    f"avg_dual_loss: {avg_dual_loss:.4f} | avg_transport_loss: {avg_transport_loss:.4f} | "
                    f"avg_cn_loss: {avg_cn_loss:.4f} | "
                    f"dual_updates: {dual_loss_count}/{batch_count} | skipped_dual_batches: {skip_dual_epoch_count} | "
                    f"dual_unstable_ctr: {self.dual_net_unstable} | transport_unstable_ctr: {self.transport_net_unstable}"
                )

            self.transport_net.eval()
            self.dual_net.eval()

            self.log_networks(k)
            self.visualize_samples(jko_step=k)

            with torch.no_grad():
                transport_net_fixed = copy.deepcopy(self.transport_net).eval()

            if self.config.crank_nicolson:
                self.last_dual_net = self._clone_dual_net()

            self.data_module.current_data = self.pushforward_samples(transport_net=transport_net_fixed, iteration=k)

            self.transport_net.train()
            self.dual_net.train()

            print("Finished " + str(k) + " JKO steps")


    def dual_loss(self, sample_images: torch.Tensor):
        with torch.no_grad():
            fake_inputs = self.data_module.current_data_loader(
                self.current_dual_steps
            )

            fake_inputs = fake_inputs.to(self.device)

            fake_images = self.transport_net(fake_inputs)

        real_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - self.dual_net(sample_images)))
        fake_loss = torch.mean(nn.ReLU(inplace=True)(1.0 + self.dual_net(fake_images)))

        self.skip_dual_optim = self._update_counter(condition=(real_loss + fake_loss).item() < 1e-4, counter="dual_net_unstable", threshold=5)

        return real_loss + fake_loss


    def transport_loss(self):
        fake_inputs = self.data_module.current_data_loader(index=None)
        fake_inputs = fake_inputs.to(self.device)
        fake_images = self.transport_net(fake_inputs)

        fake_loss = - torch.mean(self.dual_net(fake_images))
        wasserstein_loss = torch.mean((fake_inputs - fake_images).pow(2).flatten(start_dim=1).sum(dim=-1)) / (2 * self.config.alpha)

        self.skip_dual_optim = self._update_counter(condition=fake_loss.item() > 10, counter="transport_net_unstable", threshold=5)

        if not self.config.crank_nicolson:
            self.last_cn_loss_value = None
            return wasserstein_loss + fake_loss

        else:
            cn_loss = torch.tensor(0.0, device=fake_images.device)
            if self.last_dual_net is not None:
                cn_loss = - torch.mean(self.last_dual_net(fake_images))
                self.last_cn_loss_value = cn_loss.item()
                return wasserstein_loss + 0.5 * fake_loss + 0.5 * cn_loss
            else:
                self.last_cn_loss_value = None
                return wasserstein_loss + fake_loss


    def pushforward_samples(self, transport_net, iteration, batch_size=256):
        samples = self.data_module.current_data.to(self.device)

        with torch.no_grad():
            output_batches = []

            for start in range(0, samples.size(0), batch_size):
                end = start + batch_size
                batch = samples[start:end].to(self.device)
                output_batches.append(transport_net(batch))

            samples = torch.cat(output_batches, dim=0)

        return samples


    def _clone_dual_net(self):
        clone = Discriminator(df_dim=128, in_channel=1, d_spectral_norm=True, activation=nn.PReLU()).to(self.device)
        clone.load_state_dict(self.dual_net.state_dict())
        clone.eval()

        for p in clone.parameters():
            p.requires_grad_(False)

        return clone


    def _update_counter(self, condition, counter, threshold):
        if condition:
            setattr(self, counter, getattr(self, counter) + 1)
        else:
            setattr(self, counter, 0)

        return getattr(self, counter) > threshold


    def _weight_initialization_function(self, m):
        if isinstance(m, nn.Conv2d):
            if self.config.weight_initialization == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif self.config.weight_initialization == "normal":
                nn.init.normal_(m.weight, 0.00, 0.02)
            elif self.config.weight_initialization == "orthogonal":
                nn.init.orthogonal_(m.weight)
            else:
                raise ValueError(f"Unknown weight initialization function: {self.config.weight_initialization}")

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.00, 0.02)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.GroupNorm):
            nn.init.normal_(m.weight, 1.00, 0.02)
            nn.init.zeros_(m.bias)


    def log_networks(self, iteration):
        # 1) Save locally first — this is the write that must not fail mid-run.
        transport_dir = self.local_out_dir / "transport_net"
        dual_dir = self.local_out_dir / "dual_net"
        transport_dir.mkdir(parents=True, exist_ok=True)
        dual_dir.mkdir(parents=True, exist_ok=True)

        transport_path = transport_dir / f"transport_net_{iteration}.pth"
        dual_path = dual_dir / f"dual_net_{iteration}.pth"

        torch.save(self.transport_net.state_dict(), transport_path)
        torch.save(self.dual_net.state_dict(), dual_path)

        # 2) Best-effort backup to Drive. Never let this crash the run.
        self._backup_to_drive(transport_path, dual_path, iteration)

        # 3) Best-effort download to your own machine. Never let this crash the run.
        self._download_to_local_machine(transport_path, dual_path, iteration)


    def _backup_to_drive(self, transport_path: Path, dual_path: Path, iteration: int):
        try:
            drive_transport_dir = self.drive_backup_dir / "transport_net"
            drive_dual_dir = self.drive_backup_dir / "dual_net"
            drive_transport_dir.mkdir(parents=True, exist_ok=True)
            drive_dual_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(transport_path, drive_transport_dir / transport_path.name)
            shutil.copy2(dual_path, drive_dual_dir / dual_path.name)
        except OSError as e:
            print(f"WARNING: Drive backup failed for JKO step {iteration}: {e}")


    def _download_to_local_machine(self, transport_path: Path, dual_path: Path, iteration: int):
        if not self._in_colab:
            return
        try:
            zip_path = self.local_out_dir / f"jko_step_{iteration}_checkpoints.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(transport_path, arcname=transport_path.name)
                zf.write(dual_path, arcname=dual_path.name)
            _colab_files.download(str(zip_path))
        except Exception as e:
            print(f"WARNING: local-machine download failed for JKO step {iteration}: {e}")


    def visualize_samples(self, jko_step: int, num_samples: int = 100, save_every: int = 1):
        out_root = self.local_out_dir / "viz" / f"jko_{jko_step:03d}"

        samples = self.data_module.sample_new_initial(num_samples).to(self.device)

        def save_step(tensor_batch, step_idx):
            for j in range(tensor_batch.size(0)):
                sample_dir = out_root / f"sample_{j:02d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                img = self.data_module.data_postprocessing(tensor_batch[j].detach().cpu())
                save_image(img, sample_dir / f"image_{step_idx:03d}.png")

        save_step(samples, 0)

        nets = []
        for k in range(1, jko_step + 1):
            ckpt_path = self.local_out_dir / "transport_net" / f"transport_net_{k}.pth"
            if not ckpt_path.exists():
                print(f"Warning: missing checkpoint {ckpt_path}, skipping.")
                continue
            net = Unet(dim=32, out_dim=None, dim_mults=(1, 2, 4, 8), in_channel=1).to(self.device)
            net.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            net.eval()
            nets.append(net)

        total_steps = len(nets)
        keep_steps = {i for i in range(1, total_steps + 1) if i % save_every == 0}
        keep_steps.add(total_steps)

        with torch.no_grad():
            current = samples
            for i, net in enumerate(nets, start=1):
                current = net(current)
                if i in keep_steps:
                    save_step(current, i)