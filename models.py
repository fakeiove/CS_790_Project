"""
Model architectures for Latent Diffusion v3 (Fixed):
1. VAE - Fixed decoder (Upsample+Conv), added attention, latent scaling
2. Improved Conditional UNet - Fixed AdaGN conditioning order
3. Improved DiffusionScheduler - Cosine schedule, min-SNR weighting
4. EMA - Lower default decay for small datasets

Key fixes over v2:
- VAE decoder: Upsample+Conv replaces ConvTranspose2d (fixes checkerboard)
- VAE: Added self-attention at bottleneck for better global coherence
- VAE: Added latent_scale_factor so diffusion operates on ~unit variance latents
- UNet AdaGN: scale/shift applied BEFORE activation (proper AdaGN)
- EMA: default decay=0.995 (was 0.9999, too high for small datasets)
- DiffusionScheduler: added min-SNR loss weighting support
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# VAE (Fixed: Upsample+Conv decoder, attention, latent scaling)
# ============================================================

class ResBlock(nn.Module):
    """Residual block with GroupNorm."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class VAEAttention(nn.Module):
    """Self-attention for VAE bottleneck - improves global coherence."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class VAEUpsample(nn.Module):
    """Upsample + Conv instead of ConvTranspose2d to avoid checkerboard artifacts."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class VAE(nn.Module):
    """
    VAE v3 - Fixed version.

    Changes from v2:
    1. Decoder uses Upsample+Conv instead of ConvTranspose2d (no checkerboard)
    2. Self-attention at bottleneck (encoder & decoder) for global coherence
    3. latent_scale_factor: scales latents to ~unit variance for better diffusion training
    4. Slightly deeper encoder for better feature extraction
    """
    def __init__(self, in_channels=1, latent_channels=4, base_ch=64):
        super().__init__()

        # Encoder: 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1),       # 128->64
            ResBlock(base_ch, base_ch * 2),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1), # 64->32
            ResBlock(base_ch * 2, base_ch * 4),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1), # 32->16
            ResBlock(base_ch * 4, base_ch * 4),
            VAEAttention(base_ch * 4),  # [NEW] attention at bottleneck
            ResBlock(base_ch * 4, base_ch * 4),  # [NEW] extra res block
        )

        self.to_mu = nn.Conv2d(base_ch * 4, latent_channels, 1)
        self.to_logvar = nn.Conv2d(base_ch * 4, latent_channels, 1)

        self.from_latent = nn.Conv2d(latent_channels, base_ch * 4, 1)

        # Decoder: 16 -> 32 -> 64 -> 128 (using Upsample+Conv, NOT ConvTranspose2d)
        self.decoder = nn.Sequential(
            ResBlock(base_ch * 4, base_ch * 4),
            ResBlock(base_ch * 4, base_ch * 4),  # [NEW] extra res block
            VAEAttention(base_ch * 4),            # [NEW] attention at bottleneck
            VAEUpsample(base_ch * 4, base_ch * 4),  # [FIXED] 16->32
            ResBlock(base_ch * 4, base_ch * 2),
            VAEUpsample(base_ch * 2, base_ch * 2),  # [FIXED] 32->64
            ResBlock(base_ch * 2, base_ch),
            VAEUpsample(base_ch, base_ch),           # [FIXED] 64->128
            ResBlock(base_ch, base_ch),
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Tanh()
        )

        # [NEW] Learnable latent scaling factor
        # Initialized to 1.0, will be set after VAE training to normalize latent variance
        self.register_buffer('latent_scale_factor', torch.tensor(1.0))

    def compute_latent_scale(self, dataloader, device, num_batches=50):
        """
        Compute and set the latent_scale_factor based on training data.
        Call this AFTER VAE training, BEFORE diffusion training.

        The factor scales latents so they have approximately unit variance,
        which is what the diffusion model expects.
        """
        self.eval()
        all_stds = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                images = batch['image'].to(device)
                mu, _ = self.encode(images)
                all_stds.append(mu.std().item())

        avg_std = sum(all_stds) / len(all_stds)
        self.latent_scale_factor = torch.tensor(1.0 / avg_std).to(device)
        print(f"Latent scale factor set to {self.latent_scale_factor.item():.4f} "
              f"(avg latent std was {avg_std:.4f})")
        return self.latent_scale_factor

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(self.from_latent(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def encode_to_latent(self, x):
        """Encode to scaled latent space (for diffusion model input)."""
        mu, _ = self.encode(x)
        return mu * self.latent_scale_factor

    @torch.no_grad()
    def decode_from_latent(self, z):
        """Decode from scaled latent space (for diffusion model output)."""
        return self.decode(z / self.latent_scale_factor)


def vae_loss(recon, target, mu, logvar, kl_weight=1e-4):
    """VAE loss: reconstruction + KL divergence."""
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# ============================================================
# Improved Diffusion UNet (Fixed AdaGN)
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock(nn.Module):
    """
    Residual block with PROPER AdaGN conditioning + dropout.

    [FIXED] Correct order: conv -> norm -> scale/shift -> activation
    (Previously: conv -> norm -> activation -> scale/shift, which weakens conditioning)
    """
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        # [FIXED] Condition projects to scale/shift for norm2 (after conv2)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        # First conv block (unconditional)
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.dropout(h)

        # Second conv block with AdaGN conditioning
        h = self.conv2(h)
        h = self.norm2(h)

        # [FIXED] Apply scale/shift AFTER norm, BEFORE activation
        scale_shift = self.cond_proj(cond)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift

        # Activation after conditioning
        h = F.silu(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class Upsample(nn.Module):
    """Upsample + Conv instead of ConvTranspose2d to avoid checkerboard artifacts."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Downsample(nn.Module):
    """Strided convolution for downsampling."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ImprovedConditionalUNet(nn.Module):
    """
    Improved UNet for latent diffusion.

    Architecture: 16 -> 8 -> 4 -> 8 -> 16
    """
    def __init__(self, in_channels=4, base_ch=128, num_kl_classes=5,
                 time_dim=256, cond_dim=256, dropout=0.1):
        super().__init__()

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # KL grade embedding (0-4, plus 5=unconditioned for CFG)
        self.kl_embed = nn.Embedding(num_kl_classes + 1, cond_dim)

        # Combine time + KL
        self.cond_combine = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        ch = base_ch

        # === Encoder ===
        self.in_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # Level 1: 16x16
        self.down1_res1 = ConditionalResBlock(ch, ch, cond_dim, dropout)
        self.down1_res2 = ConditionalResBlock(ch, ch, cond_dim, dropout)
        self.down1_res3 = ConditionalResBlock(ch, ch, cond_dim, dropout)
        self.down1_attn = SelfAttention(ch)
        self.down1_pool = Downsample(ch)

        # Level 2: 8x8
        self.down2_res1 = ConditionalResBlock(ch, ch * 2, cond_dim, dropout)
        self.down2_res2 = ConditionalResBlock(ch * 2, ch * 2, cond_dim, dropout)
        self.down2_res3 = ConditionalResBlock(ch * 2, ch * 2, cond_dim, dropout)
        self.down2_attn = SelfAttention(ch * 2)
        self.down2_pool = Downsample(ch * 2)

        # === Bottleneck: 4x4 ===
        self.mid_res1 = ConditionalResBlock(ch * 2, ch * 2, cond_dim, dropout)
        self.mid_attn = SelfAttention(ch * 2)
        self.mid_res2 = ConditionalResBlock(ch * 2, ch * 2, cond_dim, dropout)

        # === Decoder ===
        # Level 2: 4x4 -> 8x8
        self.up2_upsample = Upsample(ch * 2)
        self.up2_res1 = ConditionalResBlock(ch * 4, ch * 2, cond_dim, dropout)  # +skip
        self.up2_res2 = ConditionalResBlock(ch * 2, ch * 2, cond_dim, dropout)
        self.up2_res3 = ConditionalResBlock(ch * 2, ch, cond_dim, dropout)
        self.up2_attn = SelfAttention(ch)

        # Level 1: 8x8 -> 16x16
        self.up1_upsample = Upsample(ch)
        self.up1_res1 = ConditionalResBlock(ch * 2, ch, cond_dim, dropout)  # +skip
        self.up1_res2 = ConditionalResBlock(ch, ch, cond_dim, dropout)
        self.up1_res3 = ConditionalResBlock(ch, ch, cond_dim, dropout)
        self.up1_attn = SelfAttention(ch)

        # Output
        self.out_norm = nn.GroupNorm(min(32, ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

        # [NEW] Zero-initialize output conv for stable training start
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, kl_grade):
        # Condition
        t_emb = self.time_mlp(t)
        kl_emb = self.kl_embed(kl_grade)
        cond = self.cond_combine(torch.cat([t_emb, kl_emb], dim=-1))

        # Encoder
        h = self.in_conv(x)

        h = self.down1_res1(h, cond)
        h = self.down1_res2(h, cond)
        h = self.down1_res3(h, cond)
        h = self.down1_attn(h)
        skip1 = h
        h = self.down1_pool(h)

        h = self.down2_res1(h, cond)
        h = self.down2_res2(h, cond)
        h = self.down2_res3(h, cond)
        h = self.down2_attn(h)
        skip2 = h
        h = self.down2_pool(h)

        # Bottleneck
        h = self.mid_res1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, cond)

        # Decoder
        h = self.up2_upsample(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.up2_res1(h, cond)
        h = self.up2_res2(h, cond)
        h = self.up2_res3(h, cond)
        h = self.up2_attn(h)

        h = self.up1_upsample(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.up1_res1(h, cond)
        h = self.up1_res2(h, cond)
        h = self.up1_res3(h, cond)
        h = self.up1_attn(h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    """
    Exponential Moving Average for model parameters.

    [FIXED] Default decay=0.995 instead of 0.9999.
    For small datasets (~10k images), 0.9999 means EMA needs ~10k updates
    to converge, which may not happen in 400 epochs with small batches.
    0.995 converges in ~200 updates while still smoothing.
    """
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ============================================================
# Improved Diffusion Scheduler (with min-SNR weighting)
# ============================================================

class DiffusionScheduler:
    """
    Improved diffusion scheduler with cosine noise schedule and min-SNR weighting.

    [NEW] min-SNR loss weighting: weights each timestep's loss by
    min(SNR(t), gamma) / SNR(t), which prevents high-noise timesteps
    from dominating the loss. This helps the model learn fine details
    at low noise levels. (From "Efficient Diffusion Training via Min-SNR Weighting")
    """
    def __init__(self, num_timesteps=1000, schedule='cosine', device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule == 'cosine':
            alphas_cumprod = self._cosine_schedule(num_timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod = alphas_cumprod.to(device)

        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # [NEW] SNR = alpha_bar / (1 - alpha_bar)
        self.snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

        # Compute betas from alphas_cumprod
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.betas = 1.0 - alphas_cumprod / alphas_cumprod_prev
        self.betas = self.betas.clamp(max=0.999)
        self.alphas = 1.0 - self.betas

        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def _cosine_schedule(self, T, s=0.008):
        """Cosine noise schedule from 'Improved DDPM' paper."""
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f[1:] / f[0]
        alphas_cumprod = alphas_cumprod.clamp(min=1e-5, max=0.9999)
        return alphas_cumprod.float()

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def get_min_snr_weights(self, t, gamma=5.0):
        """
        [NEW] Min-SNR loss weighting.
        Returns per-sample weights for the MSE loss.
        gamma=5.0 is the recommended default from the paper.
        """
        snr_t = self.snr[t]
        weights = torch.clamp(snr_t, max=gamma) / snr_t
        return weights

    @torch.no_grad()
    def ddim_sample(self, model, shape, kl_grade, num_steps=50,
                    cfg_scale=3.0, device='cuda', eta=0.0):
        """Improved DDIM sampling."""
        step_size = max(self.num_timesteps // num_steps, 1)
        timesteps = list(range(step_size - 1, self.num_timesteps, step_size))[::-1]

        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = self._get_noise_pred(model, x, t_batch, kl_grade, cfg_scale)

            alpha_bar_t = self.alphas_cumprod[t]

            # Predict x0
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-3, 3)  # [CHANGED] wider clamp for scaled latents

            # Last step: just return x0_pred
            if i == len(timesteps) - 1:
                x = x0_pred
                break

            alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]]

            # DDIM update
            if eta > 0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8) *
                    (1 - alpha_bar_t / alpha_bar_prev).clamp(min=0)
                )
                noise = torch.randn_like(x)
            else:
                sigma = 0
                noise = 0

            dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0)) * noise_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        return x

    @torch.no_grad()
    def ddim_guided_sample(self, model, z_source, kl_grade, noise_strength=0.5,
                           num_steps=50, cfg_scale=3.0, device='cuda'):
        """SDEdit-style guided generation."""
        B = z_source.shape[0]

        start_t = int(self.num_timesteps * noise_strength)
        start_t = min(start_t, self.num_timesteps - 1)
        step_size = max(self.num_timesteps // num_steps, 1)

        all_timesteps = list(range(step_size - 1, self.num_timesteps, step_size))[::-1]
        timesteps = [t for t in all_timesteps if t <= start_t]

        if len(timesteps) == 0:
            return z_source

        # Add noise to source
        t_start = torch.full((B,), start_t, device=device, dtype=torch.long)
        x, _ = self.add_noise(z_source, t_start)

        # Denoise
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self._get_noise_pred(model, x, t_batch, kl_grade, cfg_scale)

            alpha_bar_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-3, 3)

            if i == len(timesteps) - 1:
                x = x0_pred
                break

            alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]]
            dir_xt = torch.sqrt((1 - alpha_bar_prev).clamp(min=0)) * noise_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

        return x

    def _get_noise_pred(self, model, x_t, t, kl_grade, cfg_scale):
        """CFG noise prediction."""
        if cfg_scale <= 0:
            return model(x_t, t, kl_grade)

        noise_cond = model(x_t, t, kl_grade)
        uncond_label = torch.full_like(kl_grade, 5)
        noise_uncond = model(x_t, t, uncond_label)
        return noise_uncond + cfg_scale * (noise_cond - noise_uncond)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test VAE
    vae = VAE().to(device)
    x = torch.randn(2, 1, 128, 128, device=device)
    recon, mu, logvar = vae(x)
    print(f"VAE: input {x.shape} -> latent {mu.shape} -> recon {recon.shape}")
    print(f"VAE params: {sum(p.numel() for p in vae.parameters()):,}")

    # Test improved UNet
    unet = ImprovedConditionalUNet().to(device)
    z = torch.randn(2, 4, 16, 16, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    kl = torch.randint(0, 5, (2,), device=device)
    out = unet(z, t, kl)
    print(f"\nImproved UNet: input {z.shape} -> output {out.shape}")
    print(f"UNet params: {sum(p.numel() for p in unet.parameters()):,}")

    # Test EMA
    ema = EMA(unet)
    print(f"EMA initialized (decay={ema.decay})")

    # Test min-SNR weights
    scheduler = DiffusionScheduler(device=device)
    t_test = torch.randint(0, 1000, (4,), device=device)
    weights = scheduler.get_min_snr_weights(t_test)
    print(f"Min-SNR weights for t={t_test.tolist()}: {weights.tolist()}")
