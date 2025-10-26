# --------------------------------------------------------------
# train_vae_slurm.py   (the same file you will submit with sbatch)
# --------------------------------------------------------------
import os, sys, signal, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T

# ------------------------------------------------------------------
# 1️⃣  Imports – adjust to your project layout
# ------------------------------------------------------------------
from dataset import PNGDataset          # your dataset class
from models.model_vae import VAE               # wrapper around VAEGAN (decoder‑only UNet)


# --------------------------------------------------------------
# Auto‑detect device (MPS → GPU → CPU)
# --------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"[INFO] Using device: {device}")


def save_checkpoint(state: dict, ckpt_dir: Path, name: str = "ckpt_latest.pth"):
    """
    Save a checkpoint (model + optimiser + epoch + RNG state).
    The file is written atomically (`tmp` → final) to avoid corrupted files
    when the job is killed while writing.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_dir / f".tmp_{name}"
    torch.save(state, tmp_path)
    tmp_path.rename(ckpt_dir / name)          # atomic move


def load_checkpoint(ckpt_path: Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    """
    Returns the epoch to resume from (0‑based) and the loss value that was
    stored in the checkpoint.  If the file does not exist we start from
    scratch.
    """
    if not ckpt_path.is_file():
        print(f"[INFO] No checkpoint found at {ckpt_path}. Starting from epoch 0.")
        return 0, None

    print(f"[INFO] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    epoch = ckpt["epoch"]
    loss  = ckpt.get("loss")
    # also restore RNG state – useful for exact reproducibility
    torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return epoch, loss

# ------------------------------------------------------------------
# 2️⃣  Basic configuration (you can also read these from env vars)
# ------------------------------------------------------------------
latent_dim   = 128
batch_size   = 16
epochs       = 200          # you can ask for many epochs – the job will be
lr           = 1e-3
img_dir      = '../data/keras_png_slices_train'   # path is relative to $HOME
ckpt_name    = "vae_ckpt.pth"                    # name inside the ckpt dir

# ------------------------------------------------------------------
# 3️⃣  Where to store checkpoints while the job runs
# ------------------------------------------------------------------
# $SLURM_TMPDIR is a fast, node‑local SSD (≈200 GB on GPU nodes).
# It disappears when the job finishes, so we copy the final artefacts
# back to $HOME at the end of the script.
TMP_ROOT = Path(os.getenv("SLURM_TMPDIR", "/tmp"))   # fallback to /tmp
CKPT_DIR = TMP_ROOT / "vae_checkpoints"
OUT_DIR  = TMP_ROOT / "vae_output"

# ------------------------------------------------------------------
# 4️⃣  Helper to copy results back to $HOME after the job ends
# ------------------------------------------------------------------
HOME_OUT = Path(os.getenv("HOME")) / "vae_runs"
def copy_back():
    """Copy everything from the node‑local scratch to a permanent location."""
    HOME_OUT.mkdir(parents=True, exist_ok=True)
    # rsync preserves permissions, skips already‑copied files, and is fast
    os.system(f"rsync -av --progress {OUT_DIR}/ {HOME_OUT}/")
    os.system(f"rsync -av --progress {CKPT_DIR}/ {HOME_OUT}/checkpoints/")

# ------------------------------------------------------------------
# 5️⃣  Signal handling – save a checkpoint if Slurm pre‑empts us
# ------------------------------------------------------------------
def handle_sigterm(signum, frame):
    """Slurm sends SIGTERM ~2 min before the job is killed."""
    print("\n[WARN] Received SIGTERM – saving checkpoint before exit …")
    save_checkpoint(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss.item() if loss is not None else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
        },
        CKPT_DIR,
        name="ckpt_preempted.pth"
    )
    copy_back()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

# ------------------------------------------------------------------
# 6️⃣  Dataset & DataLoader (make sure images are normalised)
# ------------------------------------------------------------------
transform = T.Compose([T.ToTensor()])   # converts to float ∈[0,1] and (C,H,W)
dataset = PNGDataset(img_dir, transform=transform)
loader  = DataLoader(dataset, batch_size=batch_size,
                     shuffle=True, pin_memory=True, num_workers=4)

# ------------------------------------------------------------------
# 7️⃣  Model, optimiser, loss
# ------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = torch.nn.MSELoss()

def vae_loss(recon, x, mu, logvar):
    recon_loss = mse_loss(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ------------------------------------------------------------------
# 8️⃣  Load checkpoint if it exists (resume training)
# ------------------------------------------------------------------
start_epoch, _ = load_checkpoint(CKPT_DIR / ckpt_name, model, optimizer)

# ------------------------------------------------------------------
# 9️⃣  Training loop – checkpoint every N steps / epochs
# ------------------------------------------------------------------
save_every_steps = 200          # you can tune this (e.g. every 5 min of compute)
global_step = 0

for epoch in range(start_epoch, epochs):
    model.train()
    for i, x in enumerate(loader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        # --------------------------------------------------------------
        # Logging & checkpointing
        # --------------------------------------------------------------
        if global_step % save_every_steps == 0:
            # ---- save a *detached* image grid -------------------------
            model.eval()
            with torch.no_grad():
                save_image(
                    recon[:8].cpu().detach(),
                    OUT_DIR / f"recon_e{epoch}_s{global_step}.png"
                )
            model.train()

            # ---- checkpoint -------------------------------------------
            ckpt_state = {
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": loss.item(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None,
            }
            save_checkpoint(ckpt_state, CKPT_DIR, name=ckpt_name)
            print(f"[INFO] checkpoint saved (epoch {epoch}, step {global_step})")

        if i % 100 == 0:
            print(f"[E{epoch:03d} | S{global_step:06d}] loss={loss.item():.4f}")

        global_step += 1

    # --------------------------------------------------------------
    # End‑of‑epoch checkpoint (useful if you stop after a whole epoch)
    # --------------------------------------------------------------
    save_checkpoint(
        {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch + 1,          # next epoch to start from
            "step": global_step,
            "loss": loss.item(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
        },
        CKPT_DIR,
        name=ckpt_name
    )
    print(f"[INFO] End‑of‑epoch checkpoint saved (epoch {epoch})")

# ------------------------------------------------------------------
# 10️⃣  Job finished – copy everything back to $HOME
# ------------------------------------------------------------------
print("[INFO] Training finished – copying results back to $HOME")
copy_back()
print("[INFO] All done!")