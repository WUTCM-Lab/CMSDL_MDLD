import argparse
import os
import time
import random
import json
import math
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import cv2
from tqdm import tqdm
from sklearn import metrics
from skimage.filters.thresholding import threshold_otsu
import scipy.io as io
from utils.data import Data_Loader, to_normalization
from utils.loss import *
from utils.Evaluation import Evaluation

from model.CMSDL_MDLD import (
    DINOStudentTeacher,
    dino_loss,
    weighted_dino_loss,
    layer_token_ce_loss,
)



def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def minmax01(arr: np.ndarray):
    arr = arr.astype(np.float32)
    amin, amax = float(arr.min()), float(arr.max())
    if amax - amin < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)

def to_uint8_01(arr01: np.ndarray):
    arr01 = np.clip(arr01, 0.0, 1.0)
    return (arr01 * 255.0).astype(np.uint8)

class Timer:
    def __init__(self):
        self.t0 = None
    def start(self):
        self.t0 = time.time()
    def stop(self):
        return 0.0 if self.t0 is None else (time.time() - self.t0)

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_augment_pipelines():
    global_aug1 = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    global_aug2 = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    return global_aug1, global_aug2

def apply_batch_aug(batch: torch.Tensor, aug) -> torch.Tensor:
    return torch.stack([aug(img) for img in batch], dim=0)


def cosine_rampup(current_epoch: int, total_epochs: int, start_value: float, end_value: float):
    if total_epochs <= 1:
        return end_value
    ratio = (current_epoch - 1) / (total_epochs - 1)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    value = end_value + 0.5 * (start_value - end_value) * (1.0 + math.cos(math.pi * ratio))
    return value

def get_dynamic_distill_weights(epoch: int, total_epochs: int, base_w_sm: float, base_w_cm: float):
    w_sm_epoch = base_w_sm
    w_cm_epoch = cosine_rampup(
        current_epoch=epoch,
        total_epochs=total_epochs,
        start_value=base_w_cm * 0.35,
        end_value=base_w_cm,
    )
    return w_sm_epoch, w_cm_epoch

def get_dynamic_token_weight(epoch: int, total_epochs: int, base_w_token: float):
    return cosine_rampup(
        current_epoch=epoch,
        total_epochs=total_epochs,
        start_value=base_w_token * 0.25,
        end_value=base_w_token,
    )


@torch.no_grad()
def symmetric_kl_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p, q: (B, K)
    return: (B,)
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    kl_qp = torch.sum(q * (torch.log(q) - torch.log(p)), dim=-1)
    return 0.5 * (kl_pq + kl_qp)

@torch.no_grad()
def js_divergence_from_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p, q: (B, K)
    return: (B,)
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return 0.5 * (kl_pm + kl_qm)

@torch.no_grad()
def normalized_l2_per_sample(x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x1, x2: (B, D)
    return: (B,)
    """
    x1 = x1 / (x1.norm(dim=-1, keepdim=True) + eps)
    x2 = x2 / (x2.norm(dim=-1, keepdim=True) + eps)
    return torch.norm(x1 - x2, p=2, dim=-1)

@torch.no_grad()
def pooled_feature_distance(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """
    feat1, feat2: (B,C,H,W)
    """
    p1 = feat1.mean(dim=[2, 3])
    p2 = feat2.mean(dim=[2, 3])
    return normalized_l2_per_sample(p1, p2)

@torch.no_grad()
def flattened_feature_distance(feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
    """
    feat1, feat2: (B,C,H,W)
    """
    B = feat1.size(0)
    f1 = feat1.view(B, -1)
    f2 = feat2.view(B, -1)
    return normalized_l2_per_sample(f1, f2)


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    epoch: int,
    total_epochs: int,
    t_teacher: float,
    t_student: float,
    dino_w_cm: float = 1.0,
    dino_w_sm: float = 0.5,
    dino_w_token: float = 0.1,
    center_m: float = 0.9,
    teacher_m: float = 0.996,
    student_chunk_size: int = 128,
    conf_power: float = 1.0,
    token_layer_weights=None,
):
    model.train()
    global_aug1, global_aug2 = build_augment_pipelines()

    w_sm_epoch, w_cm_epoch = get_dynamic_distill_weights(
        epoch=epoch,
        total_epochs=total_epochs,
        base_w_sm=dino_w_sm,
        base_w_cm=dino_w_cm,
    )
    w_token_epoch = get_dynamic_token_weight(
        epoch=epoch,
        total_epochs=total_epochs,
        base_w_token=dino_w_token,
    )

    running = 0.0
    running_cm = 0.0
    running_sm = 0.0
    running_tok = 0.0
    running_conf = 0.0

    with tqdm(total=len(loader), desc=f"Train (CMSDL-MSDL++) [Epoch {epoch}]", ncols=180) as t:
        for batch_idx, (t1, t2, _) in enumerate(loader):
            t1 = t1.float()
            t2 = t2.float()

            t1_v1 = apply_batch_aug(t1, global_aug1)
            t1_v2 = apply_batch_aug(t1, global_aug2)
            t2_v1 = apply_batch_aug(t2, global_aug1)
            t2_v2 = apply_batch_aug(t2, global_aug2)

            t1_v1 = t1_v1.to(device, non_blocking=True)
            t1_v2 = t1_v2.to(device, non_blocking=True)
            t2_v1 = t2_v1.to(device, non_blocking=True)
            t2_v2 = t2_v2.to(device, non_blocking=True)

            B = t1_v1.size(0)
            num_chunks = math.ceil(B / student_chunk_size)

            with torch.no_grad():
                teacher_out_t1 = model.teacher_forward(
                    t1_v1,
                    T=t_teacher,
                    return_logits=True,
                    return_features=True,
                    return_layer_tokens=True,
                    update_codebook=True,
                )
                teacher_out_t2 = model.teacher_forward(
                    t2_v1,
                    T=t_teacher,
                    return_logits=True,
                    return_features=True,
                    return_layer_tokens=True,
                    update_codebook=True,
                )

                q_t1 = teacher_out_t1["probs"]
                q_t2 = teacher_out_t2["probs"]
                logit_t1 = teacher_out_t1["logits"]
                logit_t2 = teacher_out_t2["logits"]

                token_ids_t1 = teacher_out_t1["layer_token_ids"]
                token_ids_t2 = teacher_out_t2["layer_token_ids"]

                conf_t1 = model.batch_confidence_from_probs(q_t1)
                conf_t2 = model.batch_confidence_from_probs(q_t2)

                cm_w_t1_from_t2 = conf_t2.pow(conf_power).detach()
                cm_w_t2_from_t1 = conf_t1.pow(conf_power).detach()

                mean_conf = 0.5 * (conf_t1.mean() + conf_t2.mean())

            opt.zero_grad(set_to_none=True)

            batch_loss_value = 0.0
            batch_loss_cm_value = 0.0
            batch_loss_sm_value = 0.0
            batch_loss_tok_value = 0.0

            for start in range(0, B, student_chunk_size):
                end = min(start + student_chunk_size, B)

                s_t1_v1 = t1_v1[start:end]
                s_t1_v2 = t1_v2[start:end]
                s_t2_v1 = t2_v1[start:end]
                s_t2_v2 = t2_v2[start:end]

                q1 = q_t1[start:end]
                q2 = q_t2[start:end]
                w_cm_from_q2 = cm_w_t1_from_t2[start:end]
                w_cm_from_q1 = cm_w_t2_from_t1[start:end]

                s_out_t1_v1 = model.student_forward_logits(
                    s_t1_v1, T=t_student, return_feat=True, return_layer_token_logits=True
                )
                s_out_t1_v2 = model.student_forward_logits(
                    s_t1_v2, T=t_student, return_feat=True, return_layer_token_logits=True
                )
                s_out_t2_v1 = model.student_forward_logits(
                    s_t2_v1, T=t_student, return_feat=True, return_layer_token_logits=True
                )
                s_out_t2_v2 = model.student_forward_logits(
                    s_t2_v2, T=t_student, return_feat=True, return_layer_token_logits=True
                )

                p_t1_v1 = s_out_t1_v1["log_probs"]
                p_t1_v2 = s_out_t1_v2["log_probs"]
                p_t2_v1 = s_out_t2_v1["log_probs"]
                p_t2_v2 = s_out_t2_v2["log_probs"]

                # same-modal anchor
                loss_sm = (
                    dino_loss(p_t1_v2, q1) +
                    dino_loss(p_t2_v2, q2)
                ) * 0.5

                # cross-modal fine alignment
                loss_cm = (
                    weighted_dino_loss(p_t1_v1, q2, w_cm_from_q2) +
                    weighted_dino_loss(p_t1_v2, q2, w_cm_from_q2) +
                    weighted_dino_loss(p_t2_v1, q1, w_cm_from_q1) +
                    weighted_dino_loss(p_t2_v2, q1, w_cm_from_q1)
                ) * 0.25

                # layer-wise token distillation
                tok_sm = (
                    layer_token_ce_loss(
                        s_out_t1_v2["layer_token_logits"],
                        [x[start:end] for x in token_ids_t1],
                        layer_weights=token_layer_weights,
                    ) +
                    layer_token_ce_loss(
                        s_out_t2_v2["layer_token_logits"],
                        [x[start:end] for x in token_ids_t2],
                        layer_weights=token_layer_weights,
                    )
                ) * 0.5

                tok_cm = (
                    layer_token_ce_loss(
                        s_out_t1_v1["layer_token_logits"],
                        [x[start:end] for x in token_ids_t2],
                        layer_weights=token_layer_weights,
                    ) +
                    layer_token_ce_loss(
                        s_out_t2_v1["layer_token_logits"],
                        [x[start:end] for x in token_ids_t1],
                        layer_weights=token_layer_weights,
                    )
                ) * 0.5

                loss_tok = 0.5 * tok_sm + 0.5 * tok_cm

                loss = (
                    w_sm_epoch * loss_sm +
                    w_cm_epoch * loss_cm +
                    w_token_epoch * loss_tok
                )

                (loss / num_chunks).backward()

                batch_loss_value += float(loss.item())
                batch_loss_cm_value += float(loss_cm.item())
                batch_loss_sm_value += float(loss_sm.item())
                batch_loss_tok_value += float(loss_tok.item())

            nn.utils.clip_grad_norm_(model.student_backbone.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.student_head.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.student_layer_projectors.parameters(), max_norm=5.0)
            nn.utils.clip_grad_norm_(model.layer_token_predictors.parameters(), max_norm=5.0)

            opt.step()

            with torch.no_grad():
                model.update_teacher(m=teacher_m)
                model.update_center_logits(torch.cat([logit_t1, logit_t2], dim=0), momentum=center_m)

            batch_loss_value /= max(1, num_chunks)
            batch_loss_cm_value /= max(1, num_chunks)
            batch_loss_sm_value /= max(1, num_chunks)
            batch_loss_tok_value /= max(1, num_chunks)

            running += batch_loss_value
            running_cm += batch_loss_cm_value
            running_sm += batch_loss_sm_value
            running_tok += batch_loss_tok_value
            running_conf += float(mean_conf.item())

            t.set_postfix({
                "loss": f"{batch_loss_value:.5f}",
                "cm": f"{batch_loss_cm_value:.5f}",
                "sm": f"{batch_loss_sm_value:.5f}",
                "tok": f"{batch_loss_tok_value:.5f}",
                "conf": f"{float(mean_conf.item()):.3f}",
                "w_cm": f"{w_cm_epoch:.3f}",
                "w_sm": f"{w_sm_epoch:.3f}",
                "w_tok": f"{w_token_epoch:.3f}",
            })
            t.update(1)

    denom = max(1, len(loader))
    return {
        "loss": running / denom,
        "loss_cm": running_cm / denom,
        "loss_sm": running_sm / denom,
        "loss_tok": running_tok / denom,
        "mean_conf": running_conf / denom,
        "w_cm_epoch": w_cm_epoch,
        "w_sm_epoch": w_sm_epoch,
        "w_token_epoch": w_token_epoch,
    }



def to_3ch_image(img: np.ndarray, fallback_gray: np.ndarray) -> np.ndarray:
    if img is None:
        base = np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)
        return base
    if img.ndim == 2:
        return np.stack([img, img, img], axis=2)
    if img.ndim == 3 and img.shape[2] == 3:
        return img.copy()
    return np.stack([fallback_gray, fallback_gray, fallback_gray], axis=2)

def overlay_fp_fn(base_bgr: np.ndarray, fp_mask255: np.uint8, fn_mask255: np.uint8,
                  alpha_fp: float = 0.45, alpha_fn: float = 0.45):
    fp_layer = np.zeros_like(base_bgr)
    fn_layer = np.zeros_like(base_bgr)
    fp_layer[fp_mask255 > 0] = (0, 0, 255)
    fn_layer[fn_mask255 > 0] = (255, 0, 0)
    fp_vis = cv2.addWeighted(base_bgr, 1.0, fp_layer, alpha_fp, 0)
    fn_vis = cv2.addWeighted(base_bgr, 1.0, fn_layer, alpha_fn, 0)
    combined = cv2.addWeighted(fp_vis, 1.0, fn_layer, alpha_fn, 0)
    return combined, fp_vis, fn_vis

def overlay_tp_fp_fn(base_bgr: np.ndarray,
                     tp_mask255: np.uint8, fp_mask255: np.uint8, fn_mask255: np.uint8,
                     alpha_tp: float = 0.45, alpha_fp: float = 0.45, alpha_fn: float = 0.45):
    tp_layer = np.zeros_like(base_bgr)
    fp_layer = np.zeros_like(base_bgr)
    fn_layer = np.zeros_like(base_bgr)
    tp_layer[tp_mask255 > 0] = (0, 255, 0)
    fp_layer[fp_mask255 > 0] = (0, 0, 255)
    fn_layer[fn_mask255 > 0] = (255, 0, 0)
    tp_vis = cv2.addWeighted(base_bgr, 1.0, tp_layer, alpha_tp, 0)
    fp_vis = cv2.addWeighted(base_bgr, 1.0, fp_layer, alpha_fp, 0)
    fn_vis = cv2.addWeighted(base_bgr, 1.0, fn_layer, alpha_fn, 0)
    combined = cv2.addWeighted(base_bgr, 1.0, tp_layer, alpha_tp, 0)
    combined = cv2.addWeighted(combined, 1.0, fp_layer, alpha_fp, 0)
    combined = cv2.addWeighted(combined, 1.0, fn_layer, alpha_fn, 0)
    return combined, tp_vis, fp_vis, fn_vis



@torch.no_grad()
def validate_change_map(
    model,
    loader,
    o_h,
    o_w,
    gt,
    vis_dir,
    device,
    x1_full=None,
    x2_full=None,
    t_student_infer: float = 0.1,
    feature_layer_weights=None,
    token_layer_weights=None,
    alpha_feat: float = 0.55,
    beta_proto: float = 0.25,
    gamma_tok: float = 0.20,
    threshold_mode: str = "otsu",
    manual_thr: float = 66.3,
    percentile_thr: float = 85.0,
):

    model.eval()

    if feature_layer_weights is None:
        feature_layer_weights = [0.7, 0.9, 1.1, 1.3]
    if token_layer_weights is None:
        token_layer_weights = [0.7, 0.9, 1.1, 1.3]

    feature_scores = []
    proto_scores = []
    token_scores = []
    final_scores = []

    with tqdm(total=len(loader), desc='Validate (CM++)', ncols=170, colour='cyan') as t:
        for (t1, t2, _) in loader:
            t1 = t1.to(device).float()
            t2 = t2.to(device).float()

            out1 = model.inference_forward(t1, T_student=t_student_infer)
            out2 = model.inference_forward(t2, T_student=t_student_infer)

            feat_dist_layers = []
            feat_weight_sum = 0.0

            for i, (f1, f2) in enumerate(zip(out1["feats"], out2["feats"])):
                w = float(feature_layer_weights[i])

                d_pool = pooled_feature_distance(f1, f2)
                d_flat = flattened_feature_distance(f1, f2)
                d = 0.5 * d_pool + 0.5 * d_flat

                feat_dist_layers.append(w * d)
                feat_weight_sum += w

            feat_score = sum(feat_dist_layers) / max(feat_weight_sum, 1e-6)

            proto_score = js_divergence_from_probs(out1["proto_probs"], out2["proto_probs"])

            token_dist_layers = []
            token_weight_sum = 0.0

            for i, (p1, p2, id1, id2) in enumerate(zip(
                out1["layer_token_probs"],
                out2["layer_token_probs"],
                out1["layer_token_ids"],
                out2["layer_token_ids"],
            )):
                w = float(token_layer_weights[i])

                d_soft = js_divergence_from_probs(p1, p2)
                d_hard = (id1 != id2).float()
                d = 0.7 * d_soft + 0.3 * d_hard

                token_dist_layers.append(w * d)
                token_weight_sum += w

            token_score = sum(token_dist_layers) / max(token_weight_sum, 1e-6)

            final_score = alpha_feat * feat_score + beta_proto * proto_score + gamma_tok * token_score

            feature_scores.extend(feat_score.cpu().numpy().tolist())
            proto_scores.extend(proto_score.cpu().numpy().tolist())
            token_scores.extend(token_score.cpu().numpy().tolist())
            final_scores.extend(final_score.cpu().numpy().tolist())

            t.update(1)

    # reshape
    feat_map = minmax01(np.array(feature_scores)).reshape(o_h, o_w)
    proto_map = minmax01(np.array(proto_scores)).reshape(o_h, o_w)
    token_map = minmax01(np.array(token_scores)).reshape(o_h, o_w)
    cmi = minmax01(np.array(final_scores)).reshape(o_h, o_w)

    feat_u8 = to_uint8_01(feat_map)
    proto_u8 = to_uint8_01(proto_map)
    token_u8 = to_uint8_01(token_map)
    cmi_u8 = to_uint8_01(cmi)

    cv2.imwrite(os.path.join(vis_dir, 'CMI_feature_multi.png'), feat_u8)
    cv2.imwrite(os.path.join(vis_dir, 'CMI_proto_js.png'), proto_u8)
    cv2.imwrite(os.path.join(vis_dir, 'CMI_token_multi.png'), token_u8)
    cv2.imwrite(os.path.join(vis_dir, 'CMI_CMSDL_MSDL_PP.png'), cmi_u8)

    if threshold_mode.lower() == "otsu":
        thr = float(threshold_otsu(cmi_u8))
    elif threshold_mode.lower() == "percentile":
        thr = float(np.percentile(cmi_u8, percentile_thr))
    else:
        thr = float(manual_thr)

    bcm = (cmi_u8 > thr).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(vis_dir, 'BCM_CMSDL_MSDL_PP.png'), bcm)

    evaler = Evaluation(gt.astype(np.uint8), bcm)
    OA, KC, AA = evaler.Classification_indicators()
    P, R, F1 = evaler.ObjectExtract_indicators()

    FPR, TPR, _ = metrics.roc_curve((gt > 127).astype(int).flatten(), cmi.flatten())
    AUC = metrics.auc(FPR, TPR)

    pred = (bcm > 127).astype(np.uint8)
    gt_bin = (gt > 127).astype(np.uint8)
    FP = ((pred == 1) & (gt_bin == 0)).astype(np.uint8) * 255
    FN = ((pred == 0) & (gt_bin == 1)).astype(np.uint8) * 255
    TP = ((pred == 1) & (gt_bin == 1)).astype(np.uint8) * 255

    base_raw = x2_full if x2_full is not None else x1_full
    base_bgr = to_3ch_image(base_raw, cmi_u8)

    H, W = base_bgr.shape[:2]
    if (FP.shape[0] != H) or (FP.shape[1] != W):
        FP = cv2.resize(FP, (W, H), interpolation=cv2.INTER_NEAREST)
        FN = cv2.resize(FN, (W, H), interpolation=cv2.INTER_NEAREST)
        TP = cv2.resize(TP, (W, H), interpolation=cv2.INTER_NEAREST)

    overlay_all_2, overlay_fp, overlay_fn = overlay_fp_fn(base_bgr, FP, FN, alpha_fp=0.45, alpha_fn=0.45)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_FN.png"), overlay_all_2)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_only.png"), overlay_fp)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FN_only.png"), overlay_fn)

    overlay_all_3, overlay_tp, overlay_fp2, overlay_fn2 = overlay_tp_fp_fn(
        base_bgr, TP, FP, FN, alpha_tp=0.45, alpha_fp=0.45, alpha_fn=0.45
    )
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_FP_FN.png"), overlay_all_3)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_TP_only.png"), overlay_tp)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FP_only_v2.png"), overlay_fp2)
    cv2.imwrite(os.path.join(vis_dir, "Overlay_FN_only_v2.png"), overlay_fn2)

    print(
        f'[VAL][CMSDL-MSDL++] '
        f'AUC={AUC:.4f} OA={OA:.2f} KC={KC:.2f} F1={F1:.2f} '
        f'thr={thr:.2f} mode={threshold_mode}'
    )

    outputs = {
        "feat_u8": feat_u8,
        "proto_u8": proto_u8,
        "token_u8": token_u8,
        "cmi_u8": cmi_u8,
        "bcm_u8": bcm,
    }
    stats = {
        "AUC": float(AUC),
        "OA": float(OA),
        "KC": float(KC),
        "F1": float(F1),
        "threshold": float(thr),
        "alpha_feat": float(alpha_feat),
        "beta_proto": float(beta_proto),
        "gamma_tok": float(gamma_tok),
    }
    return outputs, stats

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', default='yellow', type=str)
    parser.add_argument('--t1_path', default='./data/Yellow/yellow_A_1.bmp', type=str)
    parser.add_argument('--t2_path', default='./data/Yellow/yellow_A_2.bmp', type=str)
    parser.add_argument('--gt_path', default='./data/Yellow/yellow_A_gt.bmp', type=str)

    parser.add_argument('--t1_nc', default=3, type=int)
    parser.add_argument('--t2_nc', default=1, type=int)
    parser.add_argument('--patch_size', default=11, type=int)
    parser.add_argument('--test_ps', default=11, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--vision_path', default='./dino_vision/yellow_A_512_msdlplus/', type=str)

    # DINO / CMSDL
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--head_hidden', default=512, type=int)
    parser.add_argument('--head_bottleneck', default=256, type=int)
    parser.add_argument('--t_teacher', default=0.04, type=float)
    parser.add_argument('--t_student', default=0.1, type=float)
    parser.add_argument('--teacher_m', default=0.996, type=float)
    parser.add_argument('--center_m', default=0.9, type=float)

    # loss weights
    parser.add_argument('--w_cm', default=1.34, type=float)
    parser.add_argument('--w_sm', default=0.165, type=float)

    # DistilMOS-style layer token distillation
    parser.add_argument('--w_token', default=0.10, type=float)
    parser.add_argument('--token_dim', default=128, type=int)
    parser.add_argument('--token_hidden', default=256, type=int)
    parser.add_argument('--num_layer_tokens', default=64, type=int)
    parser.add_argument('--token_codebook_m', default=0.95, type=float)

    # Train details
    parser.add_argument('--student_chunk_size', default=128, type=int)
    parser.add_argument('--conf_power', default=1.0, type=float)

    # Inference fusion weights
    parser.add_argument('--alpha_feat', default=0.55, type=float)
    parser.add_argument('--beta_proto', default=0.25, type=float)
    parser.add_argument('--gamma_tok', default=0.20, type=float)

    # Threshold strategy
    parser.add_argument('--threshold_mode', default='manual', type=str, choices=['otsu', 'manual', 'percentile'])
    parser.add_argument('--manual_thr', default=66.3, type=float)
    parser.add_argument('--percentile_thr', default=84.0, type=float)

    parser.add_argument('--seed', default=2025, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.vision_path)

    train_dataset = Data_Loader(
        args.data_name, args.t1_path, args.t2_path, args.gt_path,
        patch_size=args.patch_size, mode='train', transform=T.ToTensor()
    )
    test_dataset = Data_Loader(
        args.data_name, args.t1_path, args.t2_path, args.gt_path,
        patch_size=args.test_ps, mode='test', transform=T.ToTensor()
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 6,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    if args.data_name == 'bastrop':
        mat = io.loadmat(args.t1_path)
        x1 = mat['t1_L5'][:, :, 3]
        x2 = mat["t2_ALI"][:, :, 5]
        x1 = to_normalization(x1)[..., np.newaxis]
        x2 = to_normalization(x2)[..., np.newaxis]
        gt = (mat["ROI_1"] * 255).astype(np.uint8)
        x1_full = (x1 * 255).astype(np.uint8)
        x2_full = (x2 * 255).astype(np.uint8)
        x1_full = np.repeat(x1_full, 3, axis=2)
        x2_full = np.repeat(x2_full, 3, axis=2)
    else:
        x1_full = cv2.imread(args.t1_path)
        x2_full = cv2.imread(args.t2_path)
        gt = cv2.imread(args.gt_path)[:, :, 0].astype(np.uint8)

    o_h, o_w = gt.shape[:2]

    model = DINOStudentTeacher(
        in_channels=3,
        feat_dim=64,
        head_hidden=args.head_hidden,
        head_bottleneck=args.head_bottleneck,
        out_dim=args.out_dim,
        teacher_momentum=args.teacher_m,
        token_dim=args.token_dim,
        token_hidden=args.token_hidden,
        num_layer_tokens=args.num_layer_tokens,
        token_codebook_m=args.token_codebook_m,
    ).to(device)

    opt = optim.RMSprop(
        [
            {"params": model.student_backbone.parameters(), "lr": args.lr},
            {"params": model.student_head.parameters(), "lr": args.lr},
            {"params": model.student_layer_projectors.parameters(), "lr": args.lr},
            {"params": model.layer_token_predictors.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=1e-5,
        momentum=0.9,
    )

    token_layer_weights_train = [0.7, 0.9, 1.1, 1.3]

    feature_layer_weights_infer = [0.7, 0.9, 1.1, 1.3]
    token_layer_weights_infer = [0.7, 0.9, 1.1, 1.3]

    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            opt=opt,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            t_teacher=args.t_teacher,
            t_student=args.t_student,
            dino_w_cm=args.w_cm,
            dino_w_sm=args.w_sm,
            dino_w_token=args.w_token,
            center_m=args.center_m,
            teacher_m=args.teacher_m,
            student_chunk_size=args.student_chunk_size,
            conf_power=args.conf_power,
            token_layer_weights=token_layer_weights_train,
        )

        print(
            f"[EPOCH {epoch}] "
            f"loss={train_stats['loss']:.5f} "
            f"cm={train_stats['loss_cm']:.5f} "
            f"sm={train_stats['loss_sm']:.5f} "
            f"tok={train_stats['loss_tok']:.5f} "
            f"conf={train_stats['mean_conf']:.4f} "
            f"w_cm={train_stats['w_cm_epoch']:.4f} "
            f"w_sm={train_stats['w_sm_epoch']:.4f} "
            f"w_tok={train_stats['w_token_epoch']:.4f}"
        )

        ep_dir = os.path.join(args.vision_path, str(epoch))
        ensure_dir(ep_dir)

        outputs, metrics_dict = validate_change_map(
            model=model,
            loader=test_loader,
            o_h=o_h,
            o_w=o_w,
            gt=gt,
            vis_dir=ep_dir,
            device=device,
            x1_full=x1_full,
            x2_full=x2_full,
            t_student_infer=args.t_student,
            feature_layer_weights=feature_layer_weights_infer,
            token_layer_weights=token_layer_weights_infer,
            alpha_feat=args.alpha_feat,
            beta_proto=args.beta_proto,
            gamma_tok=args.gamma_tok,
            threshold_mode=args.threshold_mode,
            manual_thr=args.manual_thr,
            percentile_thr=args.percentile_thr,
        )

        io.savemat(
            os.path.join(ep_dir, 'CMSDL_MSDL_PP_outputs.mat'),
            {
                'feat_u8': outputs['feat_u8'],
                'proto_u8': outputs['proto_u8'],
                'token_u8': outputs['token_u8'],
                'cmi_u8': outputs['cmi_u8'],
                'bcm_u8': outputs['bcm_u8'],
            }
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_loss_cm": float(train_stats["loss_cm"]),
            "train_loss_sm": float(train_stats["loss_sm"]),
            "train_loss_tok": float(train_stats["loss_tok"]),
            "mean_conf": float(train_stats["mean_conf"]),
            "w_cm_epoch": float(train_stats["w_cm_epoch"]),
            "w_sm_epoch": float(train_stats["w_sm_epoch"]),
            "w_token_epoch": float(train_stats["w_token_epoch"]),
            **{k: float(v) for k, v in metrics_dict.items()},
        }
        history.append(record)

        with open(os.path.join(ep_dir, 'metrics_CMSDL_MSDL_PP.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        torch.save(
            model.state_dict(),
            os.path.join(ep_dir, f'cmsdl_msdl_pp_epoch{epoch}.pth')
        )

    with open(os.path.join(args.vision_path, 'history_all.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print('Training finished.')

if __name__ == '__main__':
    main()