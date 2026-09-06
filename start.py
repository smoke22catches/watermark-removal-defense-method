import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, padding=k // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """E(x, m) -> x_w  (HiDDeN-style)"""
    def __init__(self, msg_len=64, ch=64):
        super().__init__()
        self.msg_len = msg_len
        self.msg_fc = nn.Linear(msg_len, 32 * 32)  # проекція повідомлення у просторову карту
        self.pre = nn.Sequential(ConvBNReLU(3, ch), ConvBNReLU(ch, ch), ConvBNReLU(ch, ch))
        self.fuse = ConvBNReLU(ch + 1, ch)
        self.post = nn.Sequential(ConvBNReLU(ch, ch), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, x, m):
        b, _, h, w = x.shape
        feat = self.pre(x)
        m_map = self.msg_fc(m).view(b, 1, 32, 32)
        m_map = F.interpolate(m_map, size=(h, w), mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([feat, m_map], dim=1))
        residual = self.post(fused)
        x_w = torch.clamp(x + residual, -1.0, 1.0)   # адитивне вбудовування, обмежене за LPIPS-бюджетом
        return x_w


class Decoder(nn.Module):
    """D(x_w') -> m_hat"""
    def __init__(self, msg_len=64, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(3, ch), ConvBNReLU(ch, ch),
            ConvBNReLU(ch, ch, s=2), ConvBNReLU(ch, ch, s=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(ch, msg_len)

    def forward(self, x_w_prime):
        feat = self.net(x_w_prime).flatten(1)
        logits = self.fc(feat)
        return logits  # BCEWithLogits на виході

class DiffJPEG(nn.Module):
    """Диференційоване наближення JPEG-стиснення (straight-through DCT quantization)."""
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality

    def forward(self, x):
        # апроксимація: додаємо квантизаційний шум замість недиференційованого round()
        # (у реальній імплементації використовується блокова DCT + STE, тут спрощено для ілюстрації)
        noise = (torch.rand_like(x) - 0.5) * (1.0 / max(self.quality, 1))
        return torch.clamp(x + noise, -1.0, 1.0)


class DistortionBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.jpeg = DiffJPEG(quality=50)

    def forward(self, x):
        choice = torch.randint(0, 3, (1,)).item()
        if choice == 0:
            return self.jpeg(x)
        elif choice == 1:
            return torch.clamp(x + torch.randn_like(x) * 0.03, -1, 1)   # Gaussian noise
        else:
            return F.interpolate(F.avg_pool2d(x, 2), size=x.shape[-2:], mode="bilinear")  # downsample-upsample

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

class RegenerationProxy(nn.Module):
    """
    Диференційований проксі регенераційної атаки:
    x_w -> latent -> +noise(t) -> few-step DDIM denoise -> x_regen
    Це навчальний сурогат реальних атак (Zhao et al. regeneration attack; DiffPure).
    """
    def __init__(self, vae: AutoencoderKL, unet: UNet2DConditionModel,
                 scheduler: DDIMScheduler, n_steps=4, t_start=0.3):
        super().__init__()
        self.vae = vae.eval()
        self.unet = unet.eval()
        self.scheduler = scheduler
        self.n_steps = n_steps
        self.t_start = t_start
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

    def forward(self, x_w, text_embeds):
        latents = self.vae.encode(x_w).latent_dist.sample() * 0.18215
        t_idx = int(self.t_start * self.scheduler.config.num_train_timesteps)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, torch.tensor([t_idx], device=x_w.device))

        self.scheduler.set_timesteps(self.n_steps, device=x_w.device)
        lat = noisy_latents
        for t in self.scheduler.timesteps:
            with torch.enable_grad():  # свідомо не no_grad — щоб градієнт міг протікати до E під час adv-тренування
                noise_pred = self.unet(lat, t, encoder_hidden_states=text_embeds).sample
                lat = self.scheduler.step(noise_pred, t, lat).prev_sample

        x_regen = self.vae.decode(lat / 0.18215).sample
        return torch.clamp(x_regen, -1.0, 1.0)

def pgd_attack_on_decoder(x_w, m, decoder, eps=0.02, alpha=0.005, steps=5):
    x_adv = x_w.clone().detach().requires_grad_(True)
    for _ in range(steps):
        logits = decoder(x_adv)
        loss = F.binary_cross_entropy_with_logits(logits, m)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x_w - eps, x_w + eps).clamp(-1, 1)
        x_adv.requires_grad_(True)
    return x_adv.detach()

class AttackSampler(nn.Module):
    def __init__(self, distortion_bank, regen_proxy, decoder, probs=(0.4, 0.4, 0.2)):
        super().__init__()
        self.distortion_bank = distortion_bank
        self.regen_proxy = regen_proxy
        self.decoder = decoder
        self.probs = probs  # p(dist), p(regen), p(adv) — відповідає 𝒜_dist, 𝒜_regen, 𝒜_adv з моделі

    def forward(self, x_w, m, text_embeds):
        r = torch.rand(1).item()
        if r < self.probs[0]:
            return self.distortion_bank(x_w), "dist"
        elif r < self.probs[0] + self.probs[1]:
            return self.regen_proxy(x_w, text_embeds), "regen"
        else:
            return pgd_attack_on_decoder(x_w, m, self.decoder), "adv"

def train_step(x, encoder, decoder, attack_sampler, text_embeds,
                opt_ED, msg_len=64, lambda_perc=1.0, device="cuda"):
    b = x.size(0)
    m = torch.randint(0, 2, (b, msg_len), device=device).float()

    # --- Крок "лідера": вбудовування ---
    x_w = encoder(x, m)

    # --- Крок "послідовника": семплована атака з поточного простору 𝒜 ---
    with torch.no_grad():
        x_attacked, attack_type = attack_sampler(x_w, m, text_embeds)
        # для adv-гілки потрібен градієнт через x_w -> тому PGD рахується окремо вище на живому графі

    if attack_type == "adv":
        x_attacked = pgd_attack_on_decoder(x_w, m, decoder)  # перерахунок із градієнтом до x_w

    # --- декодування та функція втрат (відповідає формулі з Кроку 4 моделі) ---
    logits = decoder(x_attacked)
    loss_decode = F.binary_cross_entropy_with_logits(logits, m)
    loss_perc = F.mse_loss(x_w, x) + lpips_loss(x_w, x)  # lpips_loss - зовнішня функція (пакет lpips)
    loss = loss_decode + lambda_perc * loss_perc

    opt_ED.zero_grad()
    loss.backward()
    opt_ED.step()

    return {"loss": loss.item(), "loss_decode": loss_decode.item(),
            "loss_perc": loss_perc.item(), "attack_type": attack_type}

def guided_regen_attack(x_w, decoder, m, regen_proxy, text_embeds, guidance_scale=2.0):
    """
    Керована регенераційна атака: денойзинг ведеться не лише за prior дифузійної моделі,
    а й у напрямку, що явно псує вихід декодувальника (аналог guided diffusion removal).
    """
    x_regen = regen_proxy(x_w, text_embeds)
    x_regen = x_regen.clone().detach().requires_grad_(True)
    logits = decoder(x_regen)
    loss = -F.binary_cross_entropy_with_logits(logits, m)  # максимізуємо похибку
    grad = torch.autograd.grad(loss, x_regen)[0]
    x_regen = torch.clamp(x_regen - guidance_scale * 0.01 * grad.sign(), -1, 1)
    return x_regen.detach()

def train(encoder, decoder, dataloader, val_loader, epochs=100, device="cuda"):
    encoder, decoder = encoder.to(device), decoder.to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    distortion_bank = DistortionBank().to(device)
    # regen_proxy ініціалізується заздалегідь навченими vae/unet/scheduler (Stable Diffusion checkpoint)
    attack_sampler = AttackSampler(distortion_bank, regen_proxy, decoder)

    for epoch in range(epochs):
        # curriculum: збільшуємо частку 𝒜_regen з епохами
        p_regen = min(0.1 + 0.01 * epoch, 0.5)
        attack_sampler.probs = (max(0.5 - p_regen, 0.2), p_regen, 0.2)

        for x, in dataloader:
            x = x.to(device)
            stats = train_step(x, encoder, decoder, attack_sampler, text_embeds, opt)

        if epoch % 5 == 0:
            evaluate(encoder, decoder, val_loader, device)  # Крок 9

@torch.no_grad()
def evaluate(encoder, decoder, val_loader, device, attacks=("clean", "jpeg", "regen", "guided_regen")):
    results = {a: [] for a in attacks}
    for x, in val_loader:
        x = x.to(device)
        m = torch.randint(0, 2, (x.size(0), 64), device=device).float()
        x_w = encoder(x, m)

        for a in attacks:
            if a == "clean":
                x_test = x_w
            elif a == "jpeg":
                x_test = DiffJPEG(quality=50)(x_w)
            elif a == "regen":
                x_test = regen_proxy(x_w, text_embeds)
            elif a == "guided_regen":
                x_test = guided_regen_attack(x_w, decoder, m, regen_proxy, text_embeds)

            logits = decoder(x_test)
            bit_acc = ((torch.sigmoid(logits) > 0.5).float() == m).float().mean().item()
            results[a].append(bit_acc)

    for a in attacks:
        print(f"{a}: bit-accuracy = {sum(results[a]) / len(results[a]):.4f}")
    return results