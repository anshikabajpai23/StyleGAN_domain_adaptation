#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESS -> PD 3D style-transfer GAN for knee MRI domain adaptation.

This is adapted from a 3D StyleGAN/StarGAN-v2-like harmonization idea, but fixed for
an unpaired two-domain setup:
  source domain A = DESS
  target domain B = PD

Training objective:
  DESS patch + PD style code -> pseudo-PD patch
  Discriminator learns real PD vs generated pseudo-PD
  Structural losses keep DESS anatomy stable.

Recommended first run for ~70 scans:
  patch_size 64,64,16
  batch_size 1 or 2
  max_iterations 3000 for sanity, then 30000-50000 for full run
"""

import argparse
import os
import glob
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_nii_files(folder: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "**", "*.nii"), recursive=True))
    files += sorted(glob.glob(os.path.join(folder, "**", "*.nii.gz"), recursive=True))
    # remove accidental duplicates caused by *.nii matching *.nii.gz in some shells is not an issue here,
    # but keep unique anyway
    return sorted(list(dict.fromkeys(files)))


def normalize_percentile(vol: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    vol = vol.astype(np.float32)
    mask = vol > 0
    if mask.sum() < 10:
        return np.zeros_like(vol, dtype=np.float32)
    lo, hi = np.percentile(vol[mask], [p_low, p_high])
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    vol[~np.isfinite(vol)] = 0
    return vol.astype(np.float32)


def tensor_to_numpy01(x: torch.Tensor) -> np.ndarray:
    # x: [1, 1, D, H, W] or [1, D, H, W]
    x = x.detach().cpu().float().squeeze().numpy()
    x = (x + 1.0) / 2.0
    return np.clip(x, 0, 1)


# -----------------------------
# Dataset
# -----------------------------

class NiftiPatchDataset(Dataset):
    """Random foreground-biased 3D patches from NIfTI volumes."""

    def __init__(
        self,
        volume_paths: List[str],
        patch_size: Tuple[int, int, int] = (64, 64, 16),
        patches_per_volume: int = 16,
        p_low: float = 1,
        p_high: float = 99,
        foreground_threshold: float = 0.02,
        max_tries: int = 25,
    ):
        if len(volume_paths) == 0:
            raise ValueError("No NIfTI files found for this dataset.")
        self.volume_paths = volume_paths
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.p_low = p_low
        self.p_high = p_high
        self.foreground_threshold = foreground_threshold
        self.max_tries = max_tries
        self.cache = {}

    def __len__(self):
        return len(self.volume_paths) * self.patches_per_volume

    def _load_volume(self, path: str) -> np.ndarray:
        if path not in self.cache:
            vol = nib.load(path).get_fdata().astype(np.float32)
            vol = normalize_percentile(vol, self.p_low, self.p_high)
            self.cache[path] = vol
            # Keep cache small-ish for cluster memory. Random eviction.
            if len(self.cache) > 8:
                k = next(iter(self.cache.keys()))
                if k != path:
                    self.cache.pop(k, None)
        return self.cache[path]

    def _pad_if_needed(self, vol: np.ndarray) -> np.ndarray:
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        if pad_d or pad_h or pad_w:
            vol = np.pad(vol, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
        return vol

    def _random_patch(self, vol: np.ndarray) -> np.ndarray:
        vol = self._pad_if_needed(vol)
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size

        best_patch = None
        best_fg = -1.0
        for _ in range(self.max_tries):
            ds = np.random.randint(0, max(1, d - pd + 1))
            hs = np.random.randint(0, max(1, h - ph + 1))
            ws = np.random.randint(0, max(1, w - pw + 1))
            patch = vol[ds:ds + pd, hs:hs + ph, ws:ws + pw]
            fg = float((patch > self.foreground_threshold).mean())
            if fg > best_fg:
                best_fg = fg
                best_patch = patch
            if fg > 0.10:
                break
        return best_patch.astype(np.float32)

    def __getitem__(self, idx):
        vol_idx = idx % len(self.volume_paths)
        path = self.volume_paths[vol_idx]
        vol = self._load_volume(path)
        patch = self._random_patch(vol)
        x = torch.from_numpy(patch).unsqueeze(0).float()  # [1, D, H, W]
        x = x * 2.0 - 1.0
        return x, path


# -----------------------------
# 3D model blocks
# -----------------------------

class AdaIN3D(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s).view(s.size(0), -1, 1, 1, 1)
        gamma, beta = torch.chunk(h, 2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class ResBlock3D(nn.Module):
    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.norm1 = AdaIN3D(style_dim, dim)
        self.norm2 = AdaIN3D(style_dim, dim)
        self.conv1 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, s):
        residual = x
        x = self.conv1(self.act(self.norm1(x, s)))
        x = self.conv2(self.act(self.norm2(x, s)))
        return x + residual


class Generator3D(nn.Module):
    def __init__(self, style_dim=64, dim=32, n_res=3):
        super().__init__()
        self.enc1 = nn.Conv3d(1, dim, 3, 1, 1)
        self.enc2 = nn.Conv3d(dim, dim * 2, 4, 2, 1)
        self.enc3 = nn.Conv3d(dim * 2, dim * 4, 4, 2, 1)
        self.res = nn.ModuleList([ResBlock3D(dim * 4, style_dim) for _ in range(n_res)])
        self.dec1 = nn.ConvTranspose3d(dim * 4, dim * 2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose3d(dim * 2, dim, 4, 2, 1)
        self.out = nn.Conv3d(dim, 1, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, s):
        x = self.act(self.enc1(x))
        x = self.act(self.enc2(x))
        x = self.act(self.enc3(x))
        for block in self.res:
            x = block(x, s)
        x = self.act(self.dec1(x))
        x = self.act(self.dec2(x))
        return torch.tanh(self.out(x))


class StyleEncoder3D(nn.Module):
    def __init__(self, style_dim=64, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, dim, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim * 2, dim * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim * 4, dim * 8, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(dim * 8, style_dim)
        )

    def forward(self, x):
        return self.net(x)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, style_dim)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator3D(nn.Module):
    """PD-domain discriminator: real PD should be real, generated DESS->PD should be fake."""
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, dim, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim, dim * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim * 2, dim * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim * 4, dim * 8, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(dim * 8, 1, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Structural losses
# -----------------------------

class EdgeLoss3D(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.zeros(3, 3, 3)
        ky = torch.zeros(3, 3, 3)
        kz = torch.zeros(3, 3, 3)
        kx[:, :, 0] = -1; kx[:, :, 2] = 1
        ky[:, 0, :] = -1; ky[:, 2, :] = 1
        kz[0, :, :] = -1; kz[2, :, :] = 1
        self.register_buffer("kx", kx.view(1, 1, 3, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3, 3))
        self.register_buffer("kz", kz.view(1, 1, 3, 3, 3))

    def forward(self, a, b):
        def grad(x):
            gx = F.conv3d(x, self.kx, padding=1)
            gy = F.conv3d(x, self.ky, padding=1)
            gz = F.conv3d(x, self.kz, padding=1)
            return torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-6)
        return F.l1_loss(grad(a), grad(b))


class LaplacianLoss3D(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.zeros(3, 3, 3)
        k[1, 1, 1] = -6
        k[0, 1, 1] = k[2, 1, 1] = 1
        k[1, 0, 1] = k[1, 2, 1] = 1
        k[1, 1, 0] = k[1, 1, 2] = 1
        self.register_buffer("k", k.view(1, 1, 3, 3, 3))

    def forward(self, a, b):
        return F.l1_loss(F.conv3d(a, self.k, padding=1), F.conv3d(b, self.k, padding=1))


# -----------------------------
# Trainer
# -----------------------------

class DESSPDStyleGAN3D:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.G = Generator3D(args.style_dim, args.base_channels, args.n_res).to(self.device)
        self.E = StyleEncoder3D(args.style_dim, args.base_channels).to(self.device)
        self.F = MappingNetwork(args.latent_dim, args.style_dim).to(self.device)
        self.D = Discriminator3D(args.base_channels).to(self.device)
        self.edge = EdgeLoss3D().to(self.device)
        self.lap = LaplacianLoss3D().to(self.device)

        self.opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.E.parameters()) + list(self.F.parameters()), lr=args.lr_g, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    def d_step(self, x_dess, x_pd):
        self.opt_d.zero_grad(set_to_none=True)
        with torch.no_grad():
            s_pd = self.E(x_pd)
            fake_pd = self.G(x_dess, s_pd)
        real_logits = self.D(x_pd)
        fake_logits = self.D(fake_pd.detach())
        d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
        d_loss.backward()
        self.opt_d.step()
        return d_loss.item()

    def g_step(self, x_dess, x_pd):
        self.opt_g.zero_grad(set_to_none=True)

        # Reference-guided DESS -> PD
        s_pd = self.E(x_pd)
        fake_pd = self.G(x_dess, s_pd)
        fake_logits = self.D(fake_pd)

        loss_adv = F.softplus(-fake_logits).mean()
        loss_style = F.l1_loss(self.E(fake_pd), s_pd.detach())
        loss_content = F.l1_loss(fake_pd, x_dess)
        loss_edge = self.edge(fake_pd, x_dess)
        loss_lap = self.lap(fake_pd, x_dess)

        # Identity on target domain: PD with its own style should stay PD.
        id_pd = self.G(x_pd, s_pd.detach())
        loss_id = F.l1_loss(id_pd, x_pd)

        # Optional latent style regularization / diversity, low weight by default.
        z1 = torch.randn(x_dess.size(0), self.args.latent_dim, device=self.device)
        z2 = torch.randn(x_dess.size(0), self.args.latent_dim, device=self.device)
        fake1 = self.G(x_dess, self.F(z1))
        fake2 = self.G(x_dess, self.F(z2))
        loss_div = -F.l1_loss(fake1, fake2)

        loss_g = (
            self.args.lambda_adv * loss_adv +
            self.args.lambda_style * loss_style +
            self.args.lambda_content * loss_content +
            self.args.lambda_edge * loss_edge +
            self.args.lambda_lap * loss_lap +
            self.args.lambda_id * loss_id +
            self.args.lambda_div * loss_div
        )
        loss_g.backward()
        self.opt_g.step()

        return {
            "g": loss_g.item(),
            "adv": loss_adv.item(),
            "style": loss_style.item(),
            "content": loss_content.item(),
            "edge": loss_edge.item(),
            "lap": loss_lap.item(),
            "id": loss_id.item(),
            "div": loss_div.item(),
        }

    def save(self, path, iteration):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "iteration": iteration,
            "args": vars(self.args),
            "G": self.G.state_dict(),
            "E": self.E.state_dict(),
            "F": self.F.state_dict(),
            "D": self.D.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.E.load_state_dict(ckpt["E"])
        if "F" in ckpt:
            self.F.load_state_dict(ckpt["F"])
        if "D" in ckpt:
            self.D.load_state_dict(ckpt["D"])
        self.G.eval(); self.E.eval(); self.F.eval(); self.D.eval()


def save_train_preview(out_dir, iteration, x_dess, x_pd, fake_pd):
    preview_dir = Path(out_dir) / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    a = tensor_to_numpy01(x_dess[:1])
    b = tensor_to_numpy01(x_pd[:1])
    c = tensor_to_numpy01(fake_pd[:1])
    z = a.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(a[:, :, z], cmap="gray"); axes[0].set_title("DESS input"); axes[0].axis("off")
    axes[1].imshow(c[:, :, z], cmap="gray"); axes[1].set_title("Generated pseudo-PD"); axes[1].axis("off")
    axes[2].imshow(b[:, :, b.shape[2] // 2], cmap="gray"); axes[2].set_title("Real PD reference"); axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(preview_dir / f"iter_{iteration:06d}.png", dpi=140)
    plt.close(fig)


def train(args):
    set_seed(args.seed)
    dess_files = find_nii_files(args.dess_dir)
    pd_files = find_nii_files(args.pd_dir)
    print(f"Found DESS scans: {len(dess_files)}")
    print(f"Found PD scans:   {len(pd_files)}")
    if len(dess_files) == 0 or len(pd_files) == 0:
        raise RuntimeError("Need both DESS and PD NIfTI scans.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "train_files.json", "w") as f:
        json.dump({"dess": dess_files, "pd": pd_files, "args": vars(args)}, f, indent=2)

    patch_size = tuple(map(int, args.patch_size.split(",")))
    dess_ds = NiftiPatchDataset(dess_files, patch_size, args.patches_per_volume)
    pd_ds = NiftiPatchDataset(pd_files, patch_size, args.patches_per_volume)
    dess_loader = DataLoader(dess_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    pd_loader = DataLoader(pd_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    model = DESSPDStyleGAN3D(args)
    print(f"Device: {model.device}")
    print(f"Patch size: {patch_size}, batch size: {args.batch_size}")

    iteration = 0
    while iteration < args.max_iterations:
        for (x_dess, _), (x_pd, _) in zip(dess_loader, pd_loader):
            x_dess = x_dess.to(model.device, non_blocking=True)
            x_pd = x_pd.to(model.device, non_blocking=True)
            d_loss = model.d_step(x_dess, x_pd)
            g_losses = model.g_step(x_dess, x_pd)

            if iteration % args.log_every == 0:
                msg = f"iter {iteration:06d} | D {d_loss:.4f} | " + " | ".join([f"{k} {v:.4f}" for k, v in g_losses.items()])
                print(msg, flush=True)

            if iteration % args.preview_every == 0:
                with torch.no_grad():
                    fake_pd = model.G(x_dess, model.E(x_pd))
                save_train_preview(args.out_dir, iteration, x_dess, x_pd, fake_pd)

            if iteration > 0 and iteration % args.save_every == 0:
                model.save(Path(args.out_dir) / "checkpoints" / f"ckpt_{iteration:06d}.pth", iteration)

            iteration += 1
            if iteration >= args.max_iterations:
                break

    model.save(Path(args.out_dir) / "checkpoints" / "final.pth", iteration)
    print(f"Saved final model to {Path(args.out_dir) / 'checkpoints' / 'final.pth'}")


# -----------------------------
# Inference
# -----------------------------

def extract_pd_style(model: DESSPDStyleGAN3D, pd_paths: List[str], patch_size: Tuple[int, int, int], max_refs: int = 12):
    styles = []
    chosen = pd_paths[:max_refs]
    for p in chosen:
        vol = normalize_percentile(nib.load(p).get_fdata().astype(np.float32))
        d, h, w = vol.shape
        pd, ph, pw = patch_size
        vol = np.pad(vol, ((0, max(0, pd-d)), (0, max(0, ph-h)), (0, max(0, pw-w))), mode="constant")
        d, h, w = vol.shape
        coords = [
            ((d-pd)//2, (h-ph)//2, (w-pw)//2),
            ((d-pd)//3, (h-ph)//2, (w-pw)//2),
            (max(0, 2*(d-pd)//3), (h-ph)//2, (w-pw)//2),
        ]
        for ds, hs, ws in coords:
            patch = vol[ds:ds+pd, hs:hs+ph, ws:ws+pw]
            x = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float() * 2 - 1
            x = x.to(model.device)
            with torch.no_grad():
                styles.append(model.E(x))
    return torch.stack(styles, dim=0).mean(dim=0)


def harmonize_one(model: DESSPDStyleGAN3D, source_path: str, style_code: torch.Tensor, out_path: str, patch_size: Tuple[int, int, int]):
    img = nib.load(source_path)
    vol_raw = img.get_fdata().astype(np.float32)
    vol = normalize_percentile(vol_raw)
    d, h, w = vol.shape
    pd, ph, pw = patch_size
    sd, sh, sw = max(1, pd//2), max(1, ph//2), max(1, pw//2)

    out = np.zeros_like(vol, dtype=np.float32)
    weight = np.zeros_like(vol, dtype=np.float32)

    d_starts = list(range(0, max(1, d - pd + 1), sd))
    h_starts = list(range(0, max(1, h - ph + 1), sh))
    w_starts = list(range(0, max(1, w - pw + 1), sw))
    if d_starts[-1] != max(0, d-pd): d_starts.append(max(0, d-pd))
    if h_starts[-1] != max(0, h-ph): h_starts.append(max(0, h-ph))
    if w_starts[-1] != max(0, w-pw): w_starts.append(max(0, w-pw))

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                patch = vol[ds:min(ds+pd,d), hs:min(hs+ph,h), ws:min(ws+pw,w)]
                padded = np.zeros(patch_size, dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                x = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float().to(model.device) * 2 - 1
                with torch.no_grad():
                    y = model.G(x, style_code).squeeze().cpu().numpy()
                y = np.clip((y + 1) / 2, 0, 1)
                ad, ah, aw = patch.shape
                out[ds:ds+ad, hs:hs+ah, ws:ws+aw] += y[:ad, :ah, :aw]
                weight[ds:ds+ad, hs:hs+ah, ws:ws+aw] += 1

    out = out / np.maximum(weight, 1e-6)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, img.header), out_path)
    return out


def infer(args):
    patch_size = tuple(map(int, args.patch_size.split(",")))
    pd_files = find_nii_files(args.pd_dir)
    dess_files = find_nii_files(args.dess_dir)
    model = DESSPDStyleGAN3D(args)
    model.load(args.checkpoint)
    style_code = extract_pd_style(model, pd_files, patch_size, args.max_reference_pd)
    out_root = Path(args.out_dir) / "pseudo_pd_nifti"
    out_root.mkdir(parents=True, exist_ok=True)

    for src in dess_files:
        name = Path(src).name.replace(".nii.gz", "").replace(".nii", "")
        out_path = out_root / f"{name}_pseudoPD.nii.gz"
        print(f"Harmonizing {src} -> {out_path}")
        harmonize_one(model, src, style_code, str(out_path), patch_size)


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "infer"], required=True)
    p.add_argument("--dess_dir", required=True, help="Folder containing DESS NIfTI scans")
    p.add_argument("--pd_dir", required=True, help="Folder containing PD NIfTI scans")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", default="", help="Needed for infer mode")

    p.add_argument("--patch_size", default="64,64,16", help="D,H,W patch size. Use 64,64,16 first for anisotropic PD.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--patches_per_volume", type=int, default=16)
    p.add_argument("--max_iterations", type=int, default=3000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--style_dim", type=int, default=64)
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--n_res", type=int, default=3)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)

    # Start conservative for anatomy preservation. Tune after visual sanity checks.
    p.add_argument("--lambda_adv", type=float, default=1.0)
    p.add_argument("--lambda_style", type=float, default=1.0)
    p.add_argument("--lambda_content", type=float, default=20.0)
    p.add_argument("--lambda_edge", type=float, default=30.0)
    p.add_argument("--lambda_lap", type=float, default=10.0)
    p.add_argument("--lambda_id", type=float, default=10.0)
    p.add_argument("--lambda_div", type=float, default=0.0)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--preview_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--max_reference_pd", type=int, default=12)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --mode infer")
        infer(args)
