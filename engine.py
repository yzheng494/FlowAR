import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution
import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------

def _psnr_batch(pred, target):
    """Mean PSNR over a batch of [B,C,H,W] float tensors in [0,1]."""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return (20.0 * torch.log10(1.0 / mse.clamp(min=1e-8).sqrt())).mean().item()


def _ssim_batch(pred, target, data_range=1.0):
    """Mean SSIM over a [B,C,H,W] batch. Uses torchmetrics when available,
    otherwise falls back to an avg-pool approximation."""
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
        return structural_similarity_index_measure(
            pred, target, data_range=data_range).item()
    except Exception:
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        k = 11
        mu_p = F.avg_pool2d(pred,    k, stride=1, padding=k // 2)
        mu_t = F.avg_pool2d(target,  k, stride=1, padding=k // 2)
        sp = F.avg_pool2d(pred ** 2,         k, 1, k // 2) - mu_p ** 2
        st = F.avg_pool2d(target ** 2,       k, 1, k // 2) - mu_t ** 2
        spt = F.avg_pool2d(pred * target,    k, 1, k // 2) - mu_p * mu_t
        num = (2 * mu_p * mu_t + C1) * (2 * spt + C2)
        den = (mu_p ** 2 + mu_t ** 2 + C1) * (sp + st + C2)
        return (num / den).mean().item()


def _make_img_grid(tensors, nrow=8):
    """tensors: list of [B,C,H,W] float in [0,1].  Returns a PIL Image grid."""
    import PIL.Image
    from torchvision.utils import make_grid
    cat = torch.cat([t[:nrow] for t in tensors], dim=0)
    grid = make_grid(cat.clamp(0, 1), nrow=nrow, padding=2, normalize=False)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return PIL.Image.fromarray(arr)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    wandb_run=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_cached:
                moments = samples
                posterior = DiagonalGaussianDistribution(moments)
            else:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    posterior = vae.encode(samples)

            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            x = posterior.sample().mul_(0.2325)

        # forward
        with torch.cuda.amp.autocast(False):
            loss = model(x, labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        update_ema(ema_params, model_params, rate=args.ema_rate)
        # if data_iter_step%2==0:
        #     loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        #     optimizer.zero_grad()
        #     update_ema(ema_params, model_params, rate=args.ema_rate)
        # else:
        #     loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
        if log_writer is not None:
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        if wandb_run is not None and misc.is_main_process():
            wandb_run.log({'train/loss': loss_value_reduce,
                           'train/lr':   lr,
                           'step':        epoch_1000x})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = os.path.join(args.output_dir, "ariter{}cfg{}-image{}".format(args.num_step, cfg, args.num_images))
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)


    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()


        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                sampled_tokens = model_without_ddp.sample_tokens(num_steps=args.num_step, guidance=args.guidance, cfg=cfg,labels=labels_gen,)
            with torch.cuda.amp.autocast():
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()
    time.sleep(10)
    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if use_ema:
           postfix = postfix + "_ema"
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

        # persist to the output directory so results survive without tensorboard/wandb
        import json
        fid_metrics = {
            'fid{}'.format(postfix): fid,
            'is{}'.format(postfix): inception_score,
            'epoch': epoch,
            'num_images': args.num_images,
            'num_step': args.num_step,
            'cfg': cfg,
            'guidance': args.guidance,
        }
        fid_metrics_path = os.path.join(args.output_dir, 'fid_is_metrics.json')
        with open(fid_metrics_path, 'w') as f:
            json.dump(fid_metrics, f, indent=2, sort_keys=True)
        print(f"Saved FID/IS metrics to {fid_metrics_path}")

        # remove temporal saving folder
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)


def _decode_scale_tokens(tokens, full_h, full_w, vae, scale_hw):
    """Upsample latent tokens from scale_hw to (full_h, full_w) then VAE-decode."""
    lat = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[2],
                                           scale_hw, scale_hw)
    if scale_hw != full_h or scale_hw != full_w:
        lat = F.interpolate(lat, (full_h, full_w), mode='bicubic',
                            align_corners=False)
    with torch.cuda.amp.autocast():
        imgs = vae.decode(lat / 0.2325)
    return ((imgs.float() + 1) / 2).clamp(0, 1)


@torch.no_grad()
def evaluate_reconstruction(model_without_ddp, vae, val_loader, args, epoch,
                             wandb_run=None, num_samples=256, nrow_vis=8,
                             use_cached=False):
    """
    Two complementary reconstruction evaluations:

    Option A — per-scale independent (teacher forcing):
        For each scale k, the model is given the TRUE GT scale k-1 upsampled
        latent as both SB starting point X₁ and encoder input.  Each scale's
        output is decoded and compared to the GT image independently.
        Reports val/psnr_A_s{scale} and val/ssim_A_s{scale}.

    Option B — end-to-end from GT coarsest scale (no teacher forcing):
        Step 0: GT 1×1 latent is injected directly (flownet bypassed).
        Steps 1+: model generates freely using its own previous outputs.
        The final output is compared to the GT image.
        Reports val/psnr_B and val/ssim_B.

    Both yield meaningful PSNR/SSIM because the output and GT refer to the
    same image.  Only executes on rank 0.
    """
    if not misc.is_main_process():
        return {}

    model_without_ddp.eval()
    scales = model_without_ddp.scale   # e.g. [1, 2, 4, 8, 16]

    lpips_fn = None
    if _LPIPS_AVAILABLE:
        lpips_fn = _lpips_lib.LPIPS(net='alex').cuda().eval()

    # Accumulators — Option A: one list per scale; Option B: single list +
    # per-scale lists for the actual (non-teacher-forced) generation trajectory
    psnr_A = {s: [] for s in scales}
    ssim_A = {s: [] for s in scales}
    psnr_B, ssim_B, lpips_B = [], [], []
    psnr_B_scale  = {s: [] for s in scales}
    ssim_B_scale  = {s: [] for s in scales}
    lpips_B_scale = {s: [] for s in scales}

    # Visuals: coarse_ref | A_finest | B_e2e | ground_truth
    vis_rows = {'coarse': [], 'A': [], 'B': [], 'gt': []}
    collected = 0

    for samples, labels in val_loader:
        if collected >= num_samples:
            break

        samples = samples.cuda()
        labels  = labels.cuda()

        if use_cached:
            gt_latent = DiagonalGaussianDistribution(samples.float()).mode().mul_(0.2325)
        else:
            with torch.cuda.amp.autocast():
                posterior = vae.encode(samples)
            gt_latent = posterior.sample().mul_(0.2325)   # [B, C, H_lat, W_lat]
        B, C, H_lat, W_lat = gt_latent.shape

        with torch.cuda.amp.autocast():
            gt_images = vae.decode(gt_latent / 0.2325)
        gt_01 = ((gt_images.float() + 1) / 2).clamp(0, 1).cpu()

        # ── Option A: teacher-forced per-scale reconstruction ──────────────
        _, per_scale = model_without_ddp.reconstruct_from_latent(
            gt_latent, labels, num_steps=args.num_step,
        )
        for k, (s, tok) in enumerate(zip(scales, per_scale)):
            img_01 = _decode_scale_tokens(
                tok.cuda(), H_lat, W_lat, vae, s).cpu()
            psnr_A[s].append(_psnr_batch(img_01, gt_01))
            ssim_A[s].append(_ssim_batch(img_01, gt_01))

        # ── Option B: end-to-end from GT coarsest scale ────────────────────
        b_tokens, b_per_scale = model_without_ddp.sample_from_gt_coarse(
            gt_latent, labels, num_steps=args.num_step,
        )
        with torch.cuda.amp.autocast():
            b_images = vae.decode(b_tokens / 0.2325)
        b_01 = ((b_images.float() + 1) / 2).clamp(0, 1).cpu()

        psnr_B.append(_psnr_batch(b_01, gt_01))
        ssim_B.append(_ssim_batch(b_01, gt_01))

        if lpips_fn is not None:
            lp = lpips_fn(b_01.cuda() * 2 - 1,
                          gt_01.cuda() * 2 - 1).mean().item()
            lpips_B.append(lp)

        # Per-scale metrics along the actual (non-teacher-forced) generation
        # trajectory — used to measure improvement from one scale to the next.
        for s, tok in zip(scales, b_per_scale):
            img_01 = _decode_scale_tokens(tok.cuda(), H_lat, W_lat, vae, s).cpu()
            psnr_B_scale[s].append(_psnr_batch(img_01, gt_01))
            ssim_B_scale[s].append(_ssim_batch(img_01, gt_01))
            if lpips_fn is not None:
                lp_s = lpips_fn(img_01.cuda() * 2 - 1,
                                gt_01.cuda() * 2 - 1).mean().item()
                lpips_B_scale[s].append(lp_s)

        # Collect first batch for visualisation
        if not vis_rows['gt']:
            coarse_lat = F.interpolate(
                F.interpolate(gt_latent, (scales[0], scales[0]), mode='area'),
                (H_lat, W_lat), mode='bicubic', align_corners=False,
            )
            with torch.cuda.amp.autocast():
                coarse_imgs = vae.decode(coarse_lat / 0.2325)
            vis_rows['coarse'].append(((coarse_imgs.float() + 1) / 2)
                                      .clamp(0, 1).cpu())
            # Option A: show finest scale output
            a_finest_01 = _decode_scale_tokens(
                per_scale[-1].cuda(), H_lat, W_lat, vae, scales[-1]).cpu()
            vis_rows['A'].append(a_finest_01)
            vis_rows['B'].append(b_01)
            vis_rows['gt'].append(gt_01)

        collected += samples.shape[0]

    # ── Build metrics dict ─────────────────────────────────────────────────
    metrics = {}
    for s in scales:
        if psnr_A[s]:
            metrics[f'val/psnr_A_s{s}'] = float(np.mean(psnr_A[s]))
            metrics[f'val/ssim_A_s{s}'] = float(np.mean(ssim_A[s]))
    metrics['val/psnr_B'] = float(np.mean(psnr_B))
    metrics['val/ssim_B'] = float(np.mean(ssim_B))
    if lpips_B:
        metrics['val/lpips_B'] = float(np.mean(lpips_B))

    # Option B per-scale metrics (actual, non-teacher-forced generation
    # trajectory), plus the improvement going from each scale to the next.
    # PSNR/SSIM: higher is better, so improvement = value_next - value_prev.
    # LPIPS:     lower is better,  so improvement = value_prev - value_next.
    prev_s = None
    for s in scales:
        if not psnr_B_scale[s]:
            continue
        metrics[f'val/psnr_B_s{s}'] = float(np.mean(psnr_B_scale[s]))
        metrics[f'val/ssim_B_s{s}'] = float(np.mean(ssim_B_scale[s]))
        if lpips_B_scale[s]:
            metrics[f'val/lpips_B_s{s}'] = float(np.mean(lpips_B_scale[s]))
        if prev_s is not None:
            metrics[f'val/psnr_B_improve_s{prev_s}_to_s{s}'] = (
                metrics[f'val/psnr_B_s{s}'] - metrics[f'val/psnr_B_s{prev_s}'])
            metrics[f'val/ssim_B_improve_s{prev_s}_to_s{s}'] = (
                metrics[f'val/ssim_B_s{s}'] - metrics[f'val/ssim_B_s{prev_s}'])
            if f'val/lpips_B_s{s}' in metrics and f'val/lpips_B_s{prev_s}' in metrics:
                metrics[f'val/lpips_B_improve_s{prev_s}_to_s{s}'] = (
                    metrics[f'val/lpips_B_s{prev_s}'] - metrics[f'val/lpips_B_s{s}'])
        prev_s = s

    print("Val metrics —")
    print("  Option A (teacher-forced per-scale):")
    for s in scales:
        k_psnr = f'val/psnr_A_s{s}'
        if k_psnr in metrics:
            print(f"    scale {s:2d}: PSNR={metrics[k_psnr]:.3f}  "
                  f"SSIM={metrics[f'val/ssim_A_s{s}']:.4f}")
    print(f"  Option B (end-to-end from GT coarsest): "
          f"PSNR={metrics['val/psnr_B']:.3f}  "
          f"SSIM={metrics['val/ssim_B']:.4f}"
          + (f"  LPIPS={metrics['val/lpips_B']:.4f}" if lpips_B else ""))
    print("  Option B per-scale (generated vs ground truth) + scale-to-scale improvement:")
    prev_s = None
    for s in scales:
        if f'val/psnr_B_s{s}' not in metrics:
            continue
        line = (f"    scale {s:2d}: PSNR={metrics[f'val/psnr_B_s{s}']:.3f}  "
                f"SSIM={metrics[f'val/ssim_B_s{s}']:.4f}")
        if f'val/lpips_B_s{s}' in metrics:
            line += f"  LPIPS={metrics[f'val/lpips_B_s{s}']:.4f}"
        if prev_s is not None:
            line += (f"   |  Δ vs scale {prev_s}: "
                      f"PSNR={metrics[f'val/psnr_B_improve_s{prev_s}_to_s{s}']:+.3f}  "
                      f"SSIM={metrics[f'val/ssim_B_improve_s{prev_s}_to_s{s}']:+.4f}")
            if f'val/lpips_B_improve_s{prev_s}_to_s{s}' in metrics:
                line += f"  LPIPS={metrics[f'val/lpips_B_improve_s{prev_s}_to_s{s}']:+.4f}"
        print(line)
        prev_s = s

    # ── Persist metrics to the output directory ─────────────────────────────
    if getattr(args, 'output_dir', None):
        import json
        metrics_path = os.path.join(args.output_dir, 'reconstruction_metrics.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"Saved reconstruction metrics to {metrics_path}")

    if wandb_run is not None and _WANDB_AVAILABLE:
        log_dict = {**metrics, 'epoch': epoch}

        if vis_rows['gt']:
            log_dict.update({
                'val/img_coarse_ref': wandb.Image(
                    _make_img_grid(vis_rows['coarse'], nrow=nrow_vis),
                    caption='Coarse ref (GT 1×1 → upsampled)'),
                'val/img_A_teacher': wandb.Image(
                    _make_img_grid(vis_rows['A'], nrow=nrow_vis),
                    caption='Option A — teacher-forced finest scale'),
                'val/img_B_e2e': wandb.Image(
                    _make_img_grid(vis_rows['B'], nrow=nrow_vis),
                    caption='Option B — end-to-end from GT coarsest'),
                'val/img_groundtruth': wandb.Image(
                    _make_img_grid(vis_rows['gt'], nrow=nrow_vis),
                    caption='Ground truth'),
            })

        wandb_run.log(log_dict)

    return metrics


def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            moments = posterior.parameters
            posterior_flip = vae.encode(samples.flip(dims=[3]))
            moments_flip = posterior_flip.parameters

        for i, path in enumerate(paths):
            save_path = os.path.join(args.cached_path, path + '.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    return
