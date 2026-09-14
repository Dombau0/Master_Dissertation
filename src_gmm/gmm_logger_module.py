import ot
import time
import torch
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from omegaconf import DictConfig
from sklearn.decomposition import PCA


class GMM_Logger_Module():

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.directory = None
        self.start_time = None

        self.rows = []
        self.ksd_mmd_sample_size = self.config.evaluation_sample_size
        self.sliced_w2_projections = self.config.num_projections


    def set_output_directory(self, directory: Path):
        self.directory = directory


    def start_timer(self):
        self.start_time = time.time()


    def save_plots(
            self, 
            target_samples: torch.Tensor,
            current_sample_dir: str,
            num_iterations: int,
            fig_dir: str = "fig"
        ):
            plot_kde_history(
                target_samples=target_samples, 
                current_sample_dir=current_sample_dir, 
                num_iterations=num_iterations, 
                fig_dir=fig_dir,
                crank_nicolson=self.config.crank_nicolson
            )   


    def log_step(
        self,
        iteration: int,
        current_sample: torch.Tensor,
        target_sample: torch.Tensor,
        target_log_prob_fn,
        estimated_kl: float,
        mean_dual_loss: float,
        mean_transport_loss: float,
        transport_cost: float,
    ):
        elapsed = time.time() - self.start_time

        mean_err, cov_err = mean_cov_error(current_sample, target_sample)

        ksd_input = subsample(current_sample, self.ksd_mmd_sample_size)

        ksd_rbf_val = compute_ksd(ksd_input, target_log_prob_fn, kernel="rbf")
        ksd_imq_val = compute_ksd(ksd_input, target_log_prob_fn, kernel="imq", imq_c=1.0, imq_beta=-0.5)

        mmd_val = compute_mmd_rbf(
            subsample(current_sample, self.ksd_mmd_sample_size),
            subsample(target_sample, self.ksd_mmd_sample_size),
        )

        sw2_val = compute_sliced_w2(
            current_sample, target_sample, num_projections=self.sliced_w2_projections
        )

        self.rows.append({
            "jko_step": iteration,
            "time_sec": elapsed,
            "estimated_kl": estimated_kl,
            "mean_error": mean_err,
            "cov_error_fro": cov_err,
            "ksd_rbf": ksd_rbf_val,
            "ksd_imq": ksd_imq_val,
            "mmd_rbf": mmd_val,
            "sliced_w2": sw2_val,
            "transport_cost": transport_cost,
            "mean_dual_loss": mean_dual_loss,
            "mean_transport_loss": mean_transport_loss,
        })


    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    
    def save_logs(self, path: str):
        df = self.to_dataframe()

        print(df)

        # Standard CSV (comma-separated)
        df.to_csv(path, index=False)

        # Excel-friendly CSV (semicolon-separated)
        excel_path = path.replace(".csv", "_excel.csv")
        df.to_csv(excel_path, index=False, sep=";", decimal=",")

    
    def log_networks(self, transport_net, dual_net, iteration):
        out_dir = self.directory
        # out_dir = Path("out")

        transport_dir = out_dir / "transport_net"
        transport_dir.mkdir(parents=True, exist_ok=True)

        dual_dir = out_dir / "dual_net"       
        dual_dir.mkdir(parents=True, exist_ok=True)        

        torch.save(
            transport_net.state_dict(),
            transport_dir / f"transport_net_{iteration}.pth"
        )

        torch.save(
            dual_net.state_dict(),
            dual_dir / f"dual_net_{iteration}.pth"
        )

    
    def log_data(self, current_sample, iteration):
        # data_dir = Path("out") / "data"
        data_dir = self.directory / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            current_sample,
            data_dir / f"current_sample_{iteration}.pt"
        )

    
    def delete_things(self):
        if self.config.delete_networks:
            folder = self.directory / "transport_net"
            if folder.exists():
                shutil.rmtree(folder)

            folder = self.directory / "dual_net"
            if folder.exists():
                shutil.rmtree(folder)

        if self.config.delete_data:
            folder = self.directory / "data"
            if folder.exists():
                shutil.rmtree(folder)


def subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    return x[idx]


def mean_cov_error(samples: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        mean_err = torch.norm(samples.mean(0) - target.mean(0)).item()
        cov_err = torch.norm(torch.cov(samples.T) - torch.cov(target.T)).item()
        return mean_err, cov_err


def _median_bandwidth(sqdist: torch.Tensor) -> torch.Tensor:
    # median heuristic on squared pairwise distances (excluding self-pairs is fine to skip here)
    med = torch.median(sqdist)
    return torch.clamp(med, min=1e-6)


def _kernel_derivatives(sqdist: torch.Tensor, kernel: str, bandwidth, imq_c, imq_beta):
    """
    Returns f(u), f'(u), f''(u) for the radial kernel k(x,y) = f(||x-y||^2).
    """
    if kernel == "rbf":
        h = bandwidth if bandwidth is not None else _median_bandwidth(sqdist).item()
        f = torch.exp(-sqdist / (2 * h))
        f_prime = -f / (2 * h)
        f_dprime = f / (4 * h ** 2)
    elif kernel == "imq":
        # k(x, y) = (c^2 + ||x-y||^2)^beta, beta in (-1, 0) recommended (Gorham & Mackey, 2017)
        base = imq_c ** 2 + sqdist
        f = base ** imq_beta
        f_prime = imq_beta * base ** (imq_beta - 1)
        f_dprime = imq_beta * (imq_beta - 1) * base ** (imq_beta - 2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return f, f_prime, f_dprime


def compute_ksd(
    samples: torch.Tensor,
    target_log_prob_fn,
    kernel: str = "rbf",
    bandwidth: float = None,
    imq_c: float = 1.0,
    imq_beta: float = -0.5,
) -> float:
    """
    Kernelized Stein Discrepancy (Liu et al., 2016), generalized to support
    both the standard RBF kernel and the IMQ kernel recommended by
    Gorham & Mackey (2017) for weak-convergence-control guarantees.
    """
    samples = samples.detach().clone().requires_grad_(True)
    logp = target_log_prob_fn(samples).sum()
    score = torch.autograd.grad(logp, samples)[0].detach()
    samples = samples.detach()

    n, d = samples.shape
    diff = samples.unsqueeze(1) - samples.unsqueeze(0)   # (n, n, d), x_i - x_j
    sqdist = (diff ** 2).sum(-1)                          # (n, n)

    f, f_prime, f_dprime = _kernel_derivatives(sqdist, kernel, bandwidth, imq_c, imq_beta)

    kxy = f
    grad_x_k = 2 * diff * f_prime.unsqueeze(-1)           # grad_x k(x,y), shape (n, n, d)
    grad_y_k = -grad_x_k
    trace_term = -2 * d * f_prime - 4 * sqdist * f_dprime  # trace(grad_x grad_y k), shape (n, n)

    term1 = (score @ score.T) * kxy
    term2 = (score.unsqueeze(1) * grad_y_k).sum(-1)
    term3 = (grad_x_k * score.unsqueeze(0)).sum(-1)
    term4 = trace_term

    u = term1 + term2 + term3 + term4
    off_diag_sum = u.sum() - u.diagonal().sum()
    ksd_est = off_diag_sum / (n * (n - 1))

    return ksd_est.item()


def compute_mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: float = None) -> float:
    """
    Squared MMD estimate with RBF kernel (unbiased, excludes diagonal within each set).
    """
    n, m = x.shape[0], y.shape[0]

    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    if bandwidth is None:
        h = _median_bandwidth(torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])).item()
    else:
        h = bandwidth

    kxx = torch.exp(-xx / (2 * h))
    kyy = torch.exp(-yy / (2 * h))
    kxy = torch.exp(-xy / (2 * h))

    kxx_sum = (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1))
    kyy_sum = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
    kxy_sum = kxy.sum() / (n * m)

    mmd2 = kxx_sum + kyy_sum - 2 * kxy_sum
    return mmd2.item()


def compute_sliced_w2(x: torch.Tensor, y: torch.Tensor, num_projections: int = 100) -> float:
    """
    Sliced 2-Wasserstein distance: average 1D W2 over random projections.
    Implemented without POT to avoid the extra dependency; if you have POT
    installed you can swap this for ot.sliced_wasserstein_distance(x, y, ...).
    """
    d = x.shape[1]
    n, m = x.shape[0], y.shape[0]

    thetas = torch.randn(num_projections, d, device=x.device)
    thetas = thetas / thetas.norm(dim=1, keepdim=True)

    x_proj = x @ thetas.T   # (n, num_projections)
    y_proj = y @ thetas.T   # (m, num_projections)

    x_sorted, _ = torch.sort(x_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)

    if n != m:
        # interpolate the larger set's sorted values down to the smaller size
        k = min(n, m)
        idx_x = torch.linspace(0, n - 1, k, device=x.device).round().long()
        idx_y = torch.linspace(0, m - 1, k, device=x.device).round().long()
        x_sorted = x_sorted[idx_x]
        y_sorted = y_sorted[idx_y]

    sq_diffs = (x_sorted - y_sorted).pow(2).mean(0)   # mean over samples, per projection
    sw2 = sq_diffs.mean().sqrt()                        # average over projections, then sqrt

    # print(ot.sliced_wasserstein_distance(x, y, n_projections=num_projections))

    return sw2.item()


def plot_kde_history(
    target_samples: torch.Tensor,
    current_sample_dir: str,
    num_iterations: int,
    fig_dir: str = "fig",
    crank_nicolson = False
):
    current_sample_dir = Path(current_sample_dir)
    fig_dir = current_sample_dir / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    current_sample_dir = current_sample_dir / "data"

    target = target_samples.detach().cpu().numpy()

    max_points = 20000

    if target.shape[0] > max_points:
        idx = np.random.choice(target.shape[0], max_points, replace=False)
        target = target[idx]

    if target.ndim != 2:
        raise ValueError("Target samples must have shape (N,d).")

    if target.shape[1] > 2:
        pca = PCA(n_components=2)
        target_proj = pca.fit_transform(target)
    else:
        pca = None
        target_proj = target

    colors = {
        "blue": "#0000FF",
        "green": "#009E00",
        "orange": "#FFC400",
    }

    def save_kde(data, filename, title, color):
        fig, ax = plt.subplots(figsize=(6, 6))

        sns.kdeplot(
            x=data[:, 0],
            y=data[:, 1],
            fill=True,
            levels=10,
            color=color,
            alpha=0.7,
            ax=ax,
        )

        # Fixed axis limits
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        # Grid every 5 units
        ticks = np.arange(-15, 16, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)

        # Equal aspect ratio so circles stay circular
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(title, fontsize=25)
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=100)
        plt.close(fig)

    save_kde(target_proj, "target.png", "Target Distribution", color=colors["orange"])

    for k in range(0, num_iterations + 1):
            
        if (k == num_iterations) or (k % 1 == 0):

            file = current_sample_dir / f"current_sample_{k}.pt"

            if not file.exists():
                print(f"Skipping {file} (not found)")
                continue            

            samples = torch.load(file).cpu().numpy()

            max_points = 20000

            if len(samples) > max_points:
                idx = np.random.choice(len(samples), max_points, replace=False)
                samples = samples[idx]

            if pca is not None:
                samples = pca.transform(samples)

            if not crank_nicolson:
                save_kde(samples, f"current_sample_{k}.png", f"Step {k}", color=colors["blue"])
            else:
                save_kde(samples, f"current_sample_{k}.png", f"Step {k}", color=colors["green"])



