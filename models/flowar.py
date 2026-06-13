from functools import partial


import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn as nn
from timm.layers import SwiGLU
from timm.models.vision_transformer import DropPath
from typing import Optional
import torch.nn.functional as F
from models.flowmodel import SimpleTransformerAdaLN, SimpleTransformerTimeOnly
from .rope import *
from .flowloss import SILoss, SBLoss
import models.sampler as sampler
import torch.nn as nn
import torch.utils.checkpoint


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6, weight=False):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight=None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is None:
            return output
        else:
            return output * self.weight




class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            scale=None
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        half_head_dim = dim // num_heads // 2
        hw_seq_len = 16
        self.rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
        )
        self.resolusion = scale
        self.k,self.v=None,None
    def clear_cache(self):
        self.k,self.v=None,None

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        sequence = [0,]+[i**2 for i in self.resolusion]
        sequence = torch.cumsum(torch.tensor(sequence),dim=0)
        if self.training:
            q = torch.cat([self.rope(q[:, :, sequence[i]:sequence[i+1]]) for i in range(len(self.resolusion))], dim=2)
            k = torch.cat([self.rope(k[:, :, sequence[i]:sequence[i+1]]) for i in range(len(self.resolusion))], dim=2)
            x = F.scaled_dot_product_attention(
                q, k, v,attn_mask=mask,
                dropout_p=self.attn_drop if self.training else 0.,
            )
        else:
            q= self.rope(q)
            k = self.rope(k)
            if self.k is None or self.v is None:
                self.k = k
                self.v = v
            else:
                self.k = torch.cat([self.k, k], dim=2)
                self.v = torch.cat([self.v, v], dim=2)
            x = F.scaled_dot_product_attention(
                q, self.k, self.v,
            )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = SwiGLU,
            scale=None,
            use_checkpoint=False
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            scale=scale,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = RMSNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio*2/3.),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity() 
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(dim, 6*dim))
        self.dim=dim
        self.use_checkpoint=use_checkpoint

    def forward(self, x: torch.Tensor, condition, mask) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, condition, mask)
        else:
            return self._forward(x, condition, mask)
    def _forward(self, x: torch.Tensor, condition, mask) -> torch.Tensor:
        gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(condition).view(-1, 1, 6, self.dim).unbind(2)
        x = x + self.drop_path1(self.attn(self.norm1(x).mul(scale1.add(1)).add_(shift1), mask).mul_(gamma1))
        x = x + self.drop_path2(self.mlp(self.norm2(x).mul(scale2.add(1)).add_(shift2)).mul_(gamma2))
        return x


class FlowAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.,
                 proj_dropout=0.,
                 buffer_size=0,
                 diffloss_d=3,
                 diffloss_w=1024,
                 scale=(1, 2, 4, 8, 16),
                 cross=False,
                 use_checkpoint=False,
                 use_sb=False,
                 sb_mode='i2i',
                 sb_prediction='x0',
                 sb_use_condition=True,
                 sb_beta_max=1.0,
                 flownet_type='default',
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.scale = list(scale)
        self.seq_len = sum([pz * pz for pz in self.scale])
        self.token_embed_dim = vae_embed_dim * patch_size**2
        

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(1000+1, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob


        # --------------------------------------------------------------------------
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        
        self.z_proj_ln = RMSNorm(encoder_embed_dim, weight=True)#nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.mask_ratio_generator = stats.truncnorm((0.7 - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len+self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout, scale=scale, use_checkpoint=use_checkpoint) for _ in range(encoder_depth)])
        self.encoder_norm =  RMSNorm(encoder_embed_dim, weight=True)

        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len+self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout, scale=scale,use_checkpoint=use_checkpoint) for _ in range(decoder_depth)])

        self.decoder_norm =  RMSNorm(decoder_embed_dim, weight=True) 
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        assert flownet_type in ('default', 'time_only'), \
            f"flownet_type must be 'default' or 'time_only', got {flownet_type!r}"
        flownet_cls = (SimpleTransformerTimeOnly
                       if flownet_type == 'time_only'
                       else SimpleTransformerAdaLN)
        self.flownet = flownet_cls(
            in_channels=self.token_embed_dim,
            model_channels=diffloss_w,
            out_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            num_res_blocks=diffloss_d,
            cross=cross,
        )

        self.initialize_weights()

        # --------------------------------------------------------------------------

        self.use_sb = use_sb
        # 'i2i':        (version 1) scale 0 bridges Gaussian->x1 via sb_sampler,
        #               then each subsequent scale bridges prev->next via I2I SB.
        # 'i2i_refine': (version 2) scale 0 is bypassed entirely (no flownet loss,
        #               no iterative sampling); x1 is taken directly (randn or given),
        #               then x1->x2->x3->... all via I2I SB.
        assert sb_mode in ('i2i', 'i2i_refine'), \
            f"sb_mode must be 'i2i' or 'i2i_refine', got {sb_mode!r}"
        self.sb_mode          = sb_mode
        self.sb_prediction    = sb_prediction
        self.sb_use_condition = sb_use_condition
        self.sb_beta_max      = sb_beta_max
        self.flow_loss_fn = SILoss()
        self.sb_loss_fn = (SBLoss(prediction=sb_prediction,
                                  beta_max=sb_beta_max,
                                  use_condition=sb_use_condition,
                                  z_channels=decoder_embed_dim)
                           if use_sb else None)

        attention_mask = []
        start=0
        total_length = sum([pz * pz for pz in self.scale])+self.buffer_size
        for idx, pz in enumerate(self.scale):
            pz = pz ** 2
            if idx==0:
                pz+=self.buffer_size
            start += pz
            attention_mask.append(torch.cat([torch.ones((pz, start)),
                                             torch.zeros((pz, total_length - start))], dim=-1))
        # self.variable('constant', 'attention_mask', lambda :jnp.concatenate(attention_mask, axis=0))
        attention_mask = torch.cat(attention_mask, dim=0)
        attention_mask = torch.where(attention_mask == 0, -torch.inf, attention_mask)
        attention_mask = torch.where(attention_mask == 1, 0, attention_mask)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', attention_mask.contiguous())


    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]


    def forward_mae_encoder(self, x, condition, mask, start=None, end=None):
        if self.training:
            encoder_pos_embed_learned=self.encoder_pos_embed_learned
        else:
            encoder_pos_embed_learned =self.encoder_pos_embed_learned[:, start:end]
        
        # encoder position embedding
        x = x + encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x, condition, mask)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, condition, mask, start=None, end=None):
        x = self.decoder_embed(x)
        # decoder position embedding
        if self.training:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned
        else:
            decoder_pos_embed_learned=self.decoder_pos_embed_learned[:, start:end]
        x = x + decoder_pos_embed_learned

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, condition, mask)
        x = self.decoder_norm(x)[:, self.buffer_size:]
        if self.training:
            diffusion_pos_embed_learned=self.diffusion_pos_embed_learned
        else:
            diffusion_pos_embed_learned=self.diffusion_pos_embed_learned[:, start:end]

        x = x + diffusion_pos_embed_learned
        return x


    def forward(self, imgs, labels):
        
        label_drop = torch.rand(imgs.shape[0],).cuda()<self.label_drop_prob
        fake_label = torch.ones(imgs.shape[0],).cuda()*1000
        labels = torch.where(label_drop, fake_label, labels)
        gt_latents = [imgs.detach()]
        for i in self.scale[::-1][1:]:
            gt_latents.append(F.interpolate(imgs.detach(), (i,i), mode='area'))
        gt_latents=gt_latents[::-1]
        next_scale=self.scale[1:]
        B,C,H,W = imgs.shape
        x_input = [F.interpolate(F.interpolate(gt_latents[idx].detach(), (H,W), mode='bicubic'), (scale,scale), mode='area') for idx,scale in enumerate(next_scale)]

        # class embed
        class_embedding = self.class_emb(labels.long())

        # patchify and mask (drop) tokens
        gt_latents_patched = [self.patchify(i) for i in gt_latents]
        x_input_patched    = [self.patchify(i) for i in x_input]

        # x_input_patched[k] is the upsampled version of scale k (i.e. the SB x1 for scale k+1)
        # Keep them separate for per-scale SB loss before concatenating for the encoder.
        x_input_cat  = torch.cat(x_input_patched, dim=1)
        gt_latents_cat = torch.cat(gt_latents_patched, dim=1)

        x_proj = self.z_proj(x_input_cat)
        x_proj = torch.cat([class_embedding.unsqueeze(1), x_proj], dim=1)
        x_proj = torch.cat([class_embedding.unsqueeze(1).repeat(1, self.buffer_size, 1), x_proj], dim=1)

        # mae encoder
        x = self.forward_mae_encoder(x_proj, class_embedding, self.mask)

        # mae decoder
        z = self.forward_mae_decoder(x, class_embedding, self.mask)

        # diffloss
        loss = []
        start = self.buffer_size
        for scale_idx, i in enumerate(self.scale):
            z_cond = z[:, start:start + i**2]
            x0_tokens = gt_latents_cat[:, start:start + i**2]
            if self.use_sb:
                if scale_idx == 0 and self.sb_mode == 'i2i_refine':
                    # Version 2: scale 0 is bypassed — no flownet loss at the coarsest scale.
                    # The AR encoder still sees the GT scale-0 tokens for conditioning.
                    continue
                elif scale_idx > 0:
                    # Both versions: scales 1+ bridge from upsampled previous scale → X₀
                    x1_tokens = x_input_patched[scale_idx - 1]
                else:
                    # Version 1 (i2i), scale 0: bridge from Gaussian → X₀
                    x1_tokens = torch.randn_like(x0_tokens)
                l = self.sb_loss_fn(self.flownet, x0_tokens, x1_tokens, z_cond).mean()
            else:
                l = self.flow_loss_fn(self.flownet, x0_tokens, z_cond).mean()
            start += i**2
            loss.append(l / self.scale[-1]**2 * i**2)
        return sum(loss)

    def _sb_cond(self, z):
        """Return the condition for sb_sampler: actual z or broadcast null_cond."""
        if self.sb_use_condition or self.sb_loss_fn is None:
            return z
        null = self.sb_loss_fn.null_cond.expand(z.shape[0], z.shape[1], -1)
        return null.to(z.dtype)

    def sample_tokens(self, num_steps=25, guidance=0.9, cfg=1.0, labels=None, progress=False):
        
        if labels is not None:
            class_embedding = self.class_emb(labels)
        else:
            class_embedding = self.class_emb(torch.ones_like(labels).cuda()*1000)
        if not cfg == 1.0:
            class_embedding = torch.cat([class_embedding, self.class_emb(torch.ones_like(labels).cuda()*1000)], dim=0)
 
        x = class_embedding.unsqueeze(1)
        indices = list(range(len(self.scale)))
        if progress:
            indices = tqdm(indices)
        # generate latents
        sequence = [i**2 for i in self.scale]
        sequence = torch.cumsum(torch.tensor(sequence),dim=0)
        starts = torch.cat([torch.tensor([0]), sequence],dim=0)
        for blk in self.encoder_blocks:
            blk.attn.clear_cache()
        for blk in self.decoder_blocks:
            blk.attn.clear_cache()
        prev_scale_latent = None  # upsampled output from previous scale (for SB)
        for step in indices:
            start = starts[step]
            end = sequence[step]
            z = self.forward_mae_encoder(x, class_embedding, None, start, end)
            z = self.forward_mae_decoder(z, class_embedding, None, start, end)
            scaled_cfg = (cfg - 1) * step / (len(self.scale) - 1) + 1

            if self.use_sb and self.sb_mode == 'i2i_refine' and step == 0:
                # Version 2: scale 0 is not sampled by the flownet.
                # Take x1 = randn directly (or an externally provided coarse latent).
                z_sample = torch.randn([z.shape[0], z.shape[1], 16]).cuda()
            elif self.use_sb:
                # Both SB versions for step > 0, or version 1 at step 0
                x1_init = (prev_scale_latent.cuda()
                           if (step > 0 and prev_scale_latent is not None)
                           else torch.randn([z.shape[0], z.shape[1], 16]).cuda())
                sampled_token_latent = sampler.sb_sampler(
                    self.flownet,
                    x1_init,
                    self._sb_cond(z),
                    num_steps=num_steps,
                    cfg_scale=scaled_cfg,
                    guidance_high=guidance,
                    beta_max=self.sb_beta_max,
                    prediction=self.sb_prediction,
                ).float()
                if not cfg == 1.0:
                    z_sample, _ = sampled_token_latent.chunk(2, dim=0)
                else:
                    z_sample = sampled_token_latent
            else:
                sampled_token_latent = sampler.euler_sampler(
                    self.flownet,
                    torch.randn([z.shape[0], z.shape[1], 16]).cuda(),
                    z,
                    num_steps=num_steps,
                    cfg_scale=scaled_cfg,
                    guidance_high=guidance,
                ).float()
                if not cfg == 1.0:
                    z_sample, _ = sampled_token_latent.chunk(2, dim=0)
                else:
                    z_sample = sampled_token_latent

            if step == len(self.scale) - 1:
                break
            if not cfg == 1.0:
                z_sample = z_sample.repeat(2, 1, 1)
            x_ = z_sample.detach()
            B, N, C = x_.shape
            x_ = x_.permute(0, 2, 1).reshape(B, C, self.scale[step], self.scale[step])
            x_ = F.interpolate(
                F.interpolate(x_, (16, 16), mode='bicubic'),
                (self.scale[step + 1], self.scale[step + 1]),
                mode='area',
            ).reshape(B, C, -1).permute(0, 2, 1)
            prev_scale_latent = x_.clone()  # save for SB starting point at next step
            x = self.z_proj(x_)
        tokens = self.unpatchify(z_sample)
        return tokens


    @torch.no_grad()
    def reconstruct_from_latent(self, latent, labels, num_steps=25):
        """
        Reconstruction evaluation with teacher-forced GT inputs.

        For each scale k:
          - Encoder input  = GT scale_{k-1} upsampled to scale_k  (teacher forcing)
          - SB X₁          = same upsampled GT latent              (ground-truth bridge start)
          - SB X₀ prediction compared against GT scale_k

        This gives meaningful pixel-level metrics because output and GT
        correspond to the same image.

        Returns:
            final_img   : unpatchified finest-scale output  [B, C, H, W]
            per_scale   : list of generated token tensors   [B, scale_k², 16]
        """
        B, C, H, W = latent.shape

        # GT latents at every scale (coarsest→finest)
        gt_latents = [latent.detach()]
        for i in self.scale[::-1][1:]:
            gt_latents.append(F.interpolate(latent.detach(), (i, i), mode='area'))
        gt_latents = gt_latents[::-1]

        # x_input_p[k] = GT scale_k area-downsampled from bicubic-upsampled scale_{k-1}
        #                shape [B, scale[k+1]², 16]  — used as encoder input AND as X₁
        next_scale  = self.scale[1:]
        x_input_p   = [
            self.patchify(
                F.interpolate(
                    F.interpolate(gt_latents[idx].detach(), (H, W), mode='bicubic'),
                    (s, s), mode='area',
                )
            )
            for idx, s in enumerate(next_scale)
        ]

        class_embedding = self.class_emb(labels.long())

        # Reset KV caches — same as sample_tokens
        for blk in self.encoder_blocks:
            blk.attn.clear_cache()
        for blk in self.decoder_blocks:
            blk.attn.clear_cache()

        sequence = [i ** 2 for i in self.scale]
        sequence = torch.cumsum(torch.tensor(sequence), dim=0)
        starts   = torch.cat([torch.tensor([0]), sequence], dim=0)

        # Step 0 encoder input is the class embedding (identical to sample_tokens)
        x = class_embedding.unsqueeze(1)

        per_scale = []
        for step, i in enumerate(self.scale):
            start = starts[step]
            end   = sequence[step]

            z = self.forward_mae_encoder(x, class_embedding, None, start, end)
            z = self.forward_mae_decoder(z, class_embedding, None, start, end)

            # Pick the bridge starting point X₁
            if self.use_sb:
                if step > 0 and self.sb_mode == 'i2i':
                    x1_init = x_input_p[step - 1]   # GT upsampled from scale_{step-1}
                else:
                    x1_init = torch.randn([B, i ** 2, 16], device=latent.device)
                gen_tokens = sampler.sb_sampler(
                    self.flownet, x1_init, self._sb_cond(z),
                    num_steps=num_steps,
                    prediction=self.sb_prediction,
                    beta_max=self.sb_beta_max,
                ).float()
            else:
                gen_tokens = sampler.euler_sampler(
                    self.flownet,
                    torch.randn([B, i ** 2, 16], device=latent.device),
                    z,
                    num_steps=num_steps,
                ).float()

            per_scale.append(gen_tokens)

            if step == len(self.scale) - 1:
                break

            # Teacher-forced: next encoder input = GT scale_{step} upsampled
            x = self.z_proj(x_input_p[step])

        return self.unpatchify(per_scale[-1]), per_scale

    @torch.no_grad()
    def sample_from_gt_coarse(self, latent, labels, num_steps=25, guidance=0.9, cfg=1.0):
        """
        End-to-end generation anchored to the GT coarsest-scale latent (Option B eval).

        Step 0: GT scale[0] tokens are injected directly — flownet is bypassed.
        Steps 1+: free autoregressive generation, model's own previous output feeds
                  both the encoder and the SB starting point X₁.

        Because the coarsest scale pins the content to the source image, comparing
        the final output to the GT gives a meaningful PSNR/SSIM.
        """
        B, C, H, W = latent.shape

        class_embedding = self.class_emb(labels)
        if cfg != 1.0:
            class_embedding = torch.cat(
                [class_embedding,
                 self.class_emb(torch.ones_like(labels) * 1000)], dim=0
            )

        # GT coarsest-scale tokens — injected at step 0
        gt_coarse   = F.interpolate(latent.detach(),
                                    (self.scale[0], self.scale[0]), mode='area')
        z_gt_coarse = self.patchify(gt_coarse)   # [B, scale[0]², C]
        if cfg != 1.0:
            z_gt_coarse = z_gt_coarse.repeat(2, 1, 1)

        for blk in self.encoder_blocks:
            blk.attn.clear_cache()
        for blk in self.decoder_blocks:
            blk.attn.clear_cache()

        sequence = [i ** 2 for i in self.scale]
        sequence  = torch.cumsum(torch.tensor(sequence), dim=0)
        starts    = torch.cat([torch.tensor([0]), sequence], dim=0)

        x = class_embedding.unsqueeze(1)
        prev_scale_latent = None

        for step in range(len(self.scale)):
            start = starts[step]
            end   = sequence[step]

            z = self.forward_mae_encoder(x, class_embedding, None, start, end)
            z = self.forward_mae_decoder(z, class_embedding, None, start, end)
            scaled_cfg = (cfg - 1) * step / (len(self.scale) - 1) + 1

            if step == 0:
                # Bypass flownet — use GT coarsest scale directly
                z_sample = z_gt_coarse
                if cfg != 1.0:
                    z_sample, _ = z_sample.chunk(2, dim=0)
            elif self.use_sb:
                x1_init = (prev_scale_latent.cuda()
                           if prev_scale_latent is not None
                           else torch.randn([z.shape[0], z.shape[1], 16]).cuda())
                sampled = sampler.sb_sampler(
                    self.flownet, x1_init, self._sb_cond(z),
                    num_steps=num_steps,
                    cfg_scale=scaled_cfg,
                    guidance_high=guidance,
                    beta_max=self.sb_beta_max,
                    prediction=self.sb_prediction,
                ).float()
                z_sample = sampled if cfg == 1.0 else sampled.chunk(2)[0]
            else:
                sampled = sampler.euler_sampler(
                    self.flownet,
                    torch.randn([z.shape[0], z.shape[1], 16]).cuda(),
                    z,
                    num_steps=num_steps,
                    cfg_scale=scaled_cfg,
                    guidance_high=guidance,
                ).float()
                z_sample = sampled if cfg == 1.0 else sampled.chunk(2)[0]

            if step == len(self.scale) - 1:
                break

            if cfg != 1.0:
                z_sample = z_sample.repeat(2, 1, 1)
            x_ = z_sample.detach()
            Bc, Nc, Cc = x_.shape
            x_ = x_.permute(0, 2, 1).reshape(Bc, Cc, self.scale[step], self.scale[step])
            x_ = F.interpolate(
                F.interpolate(x_, (16, 16), mode='bicubic'),
                (self.scale[step + 1], self.scale[step + 1]), mode='area',
            ).reshape(Bc, Cc, -1).permute(0, 2, 1)
            prev_scale_latent = x_.clone()
            x = self.z_proj(x_)

        return self.unpatchify(z_sample)


def flowar_small(**kwargs):
    model = FlowAR(
        encoder_embed_dim=768, encoder_depth=6, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def flowar_large(**kwargs):
    model = FlowAR(
        encoder_embed_dim=1024, encoder_depth=8, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), cross=True, **kwargs)
    return model


def flowar_huge(**kwargs):
    model = FlowAR(
        encoder_embed_dim=1536, encoder_depth=15, encoder_num_heads=16,
        decoder_embed_dim=1536, decoder_depth=15, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), cross=True, use_checkpoint=True, **kwargs)
    return model


# Time-only flownet (no AR context conditioning)
def flowar_time_only_small(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('flownet_type', None)
    return flowar_small(use_sb=True, flownet_type='time_only', **kwargs)

def flowar_time_only_large(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('flownet_type', None)
    return flowar_large(use_sb=True, flownet_type='time_only', **kwargs)

def flowar_time_only_huge(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('flownet_type', None)
    return flowar_huge(use_sb=True, flownet_type='time_only', **kwargs)


# Version 1: Gaussian->x1 via sb_sampler, then x1->x2->... via I2I SB
def flowar_sb_i2i_small(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('sb_mode', None)
    return flowar_small(use_sb=True, sb_mode='i2i', **kwargs)

def flowar_sb_i2i_large(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('sb_mode', None)
    return flowar_large(use_sb=True, sb_mode='i2i', **kwargs)

def flowar_sb_i2i_huge(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('sb_mode', None)
    return flowar_huge(use_sb=True, sb_mode='i2i', **kwargs)

# Version 2: x1 taken directly (no flownet at scale 0), then x1->x2->... via I2I SB
def flowar_sb_refine_small(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('sb_mode', None)
    return flowar_small(use_sb=True, sb_mode='i2i_refine', **kwargs)

def flowar_sb_refine_large(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('sb_mode', None)
    return flowar_large(use_sb=True, sb_mode='i2i_refine', **kwargs)

def flowar_sb_refine_huge(**kwargs):
    kwargs.pop('use_sb', None); kwargs.pop('sb_mode', None)
    return flowar_huge(use_sb=True, sb_mode='i2i_refine', **kwargs)

