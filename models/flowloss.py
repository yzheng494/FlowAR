import torch
import torch.nn as nn
import numpy as np

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))


class SBLoss(nn.Module):
    """
    Image-to-Image Schrödinger Bridge loss.

    Forward process (eq. 11):
      q(X_t | X_0, X_1) = N(X_t; μ_t, Σ_t)
      μ_t = (1−t)·X_0 + t·X_1          (linear schedule, β_max=1)
      Σ_t = β_max · t(1−t)

    prediction='x0': model predicts X_0 directly.
    prediction='v':  model predicts the velocity v = X_1 − X_0
                     (time-derivative of the mean path μ_t).
                     At inference, X_0 is recovered as: X_0 ≈ X_t − t · v_pred.

    use_condition: if False, the learned null_cond embedding is broadcast over
                   the condition sequence instead of zeros.  This avoids the
                   degenerate cross-attention behaviour that arises when all K/V
                   tokens are identical bias-only vectors.  The null_cond vector
                   is trained alongside the rest of the model, just like the
                   null-class token used in CFG.
    """

    def __init__(self, prediction='x0', beta_max=1.0, weighting='uniform',
                 use_condition=True, z_channels=16):
        super().__init__()
        assert prediction in ('x0', 'v'), \
            f"prediction must be 'x0' or 'v', got {prediction!r}"
        self.prediction    = prediction
        self.beta_max      = beta_max
        self.weighting     = weighting
        self.use_condition = use_condition

        # Learnable null embedding used when use_condition=False.
        # Shape [1, 1, z_channels] — broadcast to [B, N, z_channels].
        if not use_condition:
            self.null_cond = nn.Parameter(torch.zeros(1, 1, z_channels))

    def forward(self, model, x0, x1, condition):
        if self.weighting == 'uniform':
            t = torch.rand((x0.shape[0], 1, 1), device=x0.device, dtype=x0.dtype)

        sigma_t_sq     = self.beta_max * t
        sigma_bar_t_sq = self.beta_max * (1.0 - t)
        denom          = sigma_t_sq + sigma_bar_t_sq   # = beta_max (constant)

        mu_t    = (sigma_bar_t_sq / denom) * x0 + (sigma_t_sq / denom) * x1
        Sigma_t = sigma_t_sq * sigma_bar_t_sq / denom  # β_max · t(1−t)

        noise = torch.randn_like(x0)
        x_t   = mu_t + torch.sqrt(Sigma_t.clamp(min=0)) * noise

        if self.prediction == 'x0':
            target = x0
        else:  # 'v'
            target = x1 - x0   # velocity of the mean path: dμ_t/dt = X_1 − X_0

        if self.use_condition:
            cond = condition
        else:
            # Broadcast learned null token across the full condition sequence
            cond = self.null_cond.expand(condition.shape[0], condition.shape[1], -1)

        output = model(x_t, t.flatten(), cond)
        loss = mean_flat((output - target) ** 2)
        return loss

    # Keep __call__ routing through forward (nn.Module handles this automatically).


class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, condition):
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1))
                
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
            
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        model_output  = model(model_input, time_input.flatten(), condition)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        return denoising_loss