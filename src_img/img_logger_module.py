"""
Standalone inference script: pushes a fixed batch of samples through a saved
chain of JKO transport-net checkpoints and dumps the results.

Unlike IMG_Train_Module.visualize_samples (called mid-training, folder-per-sample),
this groups output by step: all samples produced after a given JKO step land in
the same folder, so you can flip through step_000, step_001, ... to watch the
whole batch evolve.

checkpoint_dir / output_dir default relative to THIS FILE's own location on
disk (its parent's parent / "out_img" / ...), not to whatever directory you
happen to launch `python` from and not to Hydra's own outputs/<date>/<time>
cwd. So it doesn't matter whether you run it from Thesis/, Thesis/src_img/,
or anywhere else:

    python generate_test_samples.py \
        checkpoint_dir=../out_img/alpha-05.0_cn-false_device-cuda_dims-1024_seed-0 \
        num_steps=30 num_samples=100

A relative checkpoint_dir/output_dir override is resolved the same way (relative
to this file's parent directory). Pass an absolute path to bypass that entirely.
All of checkpoint_dir / output_dir / num_samples / num_steps / save_every are
optional hydra overrides; sensible defaults are used if omitted.
"""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image

from img_data_module import IMG_Data_Module
from img_network_module import Unet

# Anchor relative paths to where THIS FILE lives on disk, not to whatever
# directory the process happened to be launched from (and not to Hydra's
# own outputs/<date>/<time> cwd either). This makes path resolution
# independent of how/where you invoke `python`.
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_path(path_str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def generate_test_samples(
    config: DictConfig,
    checkpoint_dir: Path,
    output_dir: Path,
    num_samples: int = 100,
    num_steps: int = 30,
    save_every: int = 1,
    device: torch.device = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_module = IMG_Data_Module(config.data_module)
    transport_dir = checkpoint_dir / "transport_net"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fixed batch of initial noise, preprocessed exactly as during training.
    current = data_module.sample_new_initial(num_samples).to(device)

    def save_step(tensor_batch: torch.Tensor, step_idx: int):
        step_dir = output_dir / f"step_{step_idx:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        for j in range(tensor_batch.size(0)):
            img = data_module.data_postprocessing(tensor_batch[j].detach().cpu())
            save_image(img, step_dir / f"sample_{j:03d}.png")

    # step 0 = the raw noise batch, before any transport net is applied
    save_step(current, 0)

    keep_steps = {i for i in range(1, num_steps + 1) if i % save_every == 0}
    keep_steps.add(num_steps)

    with torch.no_grad():
        for step in range(1, num_steps + 1):
            ckpt_path = transport_dir / f"transport_net_{step}.pth"
            if not ckpt_path.exists():
                print(f"Warning: missing checkpoint {ckpt_path}, stopping early.")
                break

            net = Unet(dim=32, out_dim=None, dim_mults=(1, 2, 4, 8), in_channel=1).to(device)
            net.load_state_dict(torch.load(ckpt_path, map_location=device))
            net.eval()

            current = net(current)

            if step in keep_steps:
                save_step(current, step)
                print(f"Saved step {step}/{num_steps} -> {output_dir / f'step_{step:03d}'}")

            # free the net before loading the next checkpoint
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"Done. Samples written under: {output_dir}")


@hydra.main(version_base=None, config_path="../config", config_name="img_config")
def main(config: DictConfig):
    checkpoint_dir = resolve_path(config.get(
        "checkpoint_dir",
        "../out_img/alpha-30.0_cn-true_device-cuda_dims-1024_seed-0",
    ))

    output_dir_cfg = config.get("output_dir", None)
    output_dir = resolve_path(output_dir_cfg) if output_dir_cfg else checkpoint_dir / "test"

    print(f"Script directory:        {SCRIPT_DIR}")
    print(f"Resolved checkpoint_dir: {checkpoint_dir}")
    print(f"Resolved output_dir:     {output_dir}")

    generate_test_samples(
        config=config,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        num_samples=config.get("num_samples", 1000),
        num_steps=config.get("num_steps", 30),
        save_every=config.get("save_every", 1),
    )


if __name__ == "__main__":
    main()