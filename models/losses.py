# Local modifications (vs GitHub HEAD e1e2447), for trm_diffusion_adaptive_v2:
#   1. z_cut_loss: weighted by outputs["z_cut_valid"] -> samples without a cut/keep decision
#      (first step of a new puzzle) are excluded from the z loss (gradient already 0 there via detach; fixes the logged value).
#   2. Metrics: train/z_cut_rate (fraction actually cut) and train/z_label_cut_rate (fraction labeled "cut"),
#      over decision samples only; multiplied by count because pretrain.py divides non-loss metrics by count.
#   3. DDP fix (2026-11-16): z_cut_loss starts as 0 * (z_cut_logits - z_keep_logits).sum() whenever the model emits z logits.
#      With z_label_prob<1 the label is drawn by Python random.random() (unseeded, differs per rank); on ranks without a label
#      z_head.grad was None, so pretrain.py's `if param.grad is not None: dist.all_reduce(...)` called all_reduce a different
#      number of times per rank -> hang / mismatched reductions. Now z_head always gets a (zero) grad.
#   Models without z keys (trm, trm_diffusion, ...) are unaffected.
from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
            
        z_cut_loss = 0
        if "z_cut_logits" in outputs:
            # DDP: keep z_head grad non-None even on steps without a z label (z_label_prob<1, decided per rank),
            # otherwise pretrain.py's `if param.grad is not None: all_reduce` desyncs across ranks
            z_cut_loss = 0 * (outputs["z_cut_logits"] - outputs["z_keep_logits"]).sum()
        if "target_z_cut" in outputs:
            # z_cut_valid (adaptive_v2): only samples that continue the same puzzle have a cut/keep decision
            w = outputs["z_cut_valid"].to(outputs["z_cut_logits"].dtype) if "z_cut_valid" in outputs else None
            z_cut_loss = F.binary_cross_entropy_with_logits(outputs["z_cut_logits"] - outputs["z_keep_logits"], outputs["target_z_cut"], weight=w, reduction="sum")
            metrics["z_cut_loss"] = z_cut_loss.detach()
        if "z_cut_valid" in outputs:
            with torch.no_grad():
                v = outputs["z_cut_valid"]; nv = v.sum().clamp_min(1)
                # pretrain.py divides non-loss metrics by `count` -> multiply by count so the logged value is a rate
                metrics["z_cut_rate"] = (outputs["z_cut_taken"] & v).sum() / nv * metrics["count"]
                if "target_z_cut" in outputs:
                    metrics["z_label_cut_rate"] = ((outputs["target_z_cut"] > 0.5) & v).sum() / nv * metrics["count"]
            
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss + z_cut_loss), metrics, detached_outputs, new_carry.halted.all()

