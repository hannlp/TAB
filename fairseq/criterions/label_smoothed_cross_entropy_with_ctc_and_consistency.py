# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.models.speech_to_text.utils import save_to_dict
from fairseq.criterions import register_criterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
from .ctc import CtcCriterion, CtcCriterionConfig

logger = logging.getLogger(__name__)

def sigmoid(x):
  return 1. / (1. + math.exp(-x))

@register_criterion("label_smoothed_cross_entropy_with_ctc_and_consistency")
class LabelSmoothedCrossEntropyCriterionWithCTCAndConsistency(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, label_smoothing,
                 sentence_avg,
                 cfg: CtcCriterionConfig,
                 ctc_weight=0.0,
                 save_dir=None,
                 use_auxiliary_branch=True,
                 auxiliary_branch_loss_weight=1.0,
                 consistency_type='bi-KL',
                 consistency_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing)

        self.report_accuracy = True
        self.ctc_weight = ctc_weight
        self.ctc_criterion = CtcCriterion(cfg, task, ctc_weight, save_dir)
        self.save_dir = save_dir
        self.use_auxiliary_branch = use_auxiliary_branch
        self.auxiliary_branch_loss_weight = auxiliary_branch_loss_weight
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        logger.info("The type of consistency loss is: {}".format(self.consistency_type))

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        CtcCriterion.add_args(parser)

        parser.add_argument(
            "--use-auxiliary-branch",
            default=False,
            action="store_true",
            help="Introduce auxiliary branch during fine-tuning",
        )
        parser.add_argument(
            "--auxiliary-branch-loss-weight",
            default=1.0,
            type=float,
            help="The weight of auxiliary branch loss",
        )
        parser.add_argument(
            "--consistency-type",
            default="bi-KL",
            type=str,
            help="the type of consistency, can be 'bi-KL', 'jsd', 'jsd-detach', 'aux2orig-KL', 'aux2orig-KL-detach', 'orig2aux-KL', 'orig2aux-KL-detach'",
        )
        parser.add_argument(
            "--consistency-weight",
            default=1.0,
            type=float,
            help="The weight of consistency loss",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()

        audio_out, encoder_padding_mask, ctc_out = model.encoder.encode_audio(src_tokens, src_lengths)
        encoder_out_orig = model.encoder.encode_text(audio_out, encoder_padding_mask)
        encoder_out_orig = save_to_dict(encoder_out_orig, encoder_padding_mask, ctc_out)

        # original branch loss
        net_output_orig = model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out_orig)
        loss_orig, nll_loss_orig, lprobs_orig, target = self.compute_loss_with_lprobs(model, net_output_orig, sample, reduce=reduce)

        device = net_output_orig[0].device
        loss_aux, nll_loss_aux, consistency_loss = torch.tensor([0.]).to(device), torch.tensor([0.]).to(device), torch.tensor([0.]).to(device)

        if model.training and self.use_auxiliary_branch:
            probs = self.get_replacement_probs(model, net_output_orig, sample, self.padding_idx)
            metrics.log_scalar("probs", sum(probs) / len(probs), round=3)
            aux = model.encoder.get_auxiliary_branch(audio_out, ctc_out["ctc_alignment"][0], probs)
            encoder_out_aux = model.encoder.encode_text(aux, encoder_padding_mask)
            encoder_out_aux = save_to_dict(encoder_out_aux, encoder_padding_mask)
            # auxiliary branch loss
            net_output_aux = model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out_aux)
            loss_aux, nll_loss_aux, lprobs_aux, _ = self.compute_loss_with_lprobs(model, net_output_aux, sample, reduce=reduce)
            # consistency loss
            consistency_loss = self.compute_consistency_loss(lprobs_orig, lprobs_aux, target, ignore_index=self.padding_idx)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        n_tokens = sample["ntokens"]
        n_sentences = sample["target"].size(0)

        logging_output = {
            "trans_loss": utils.item(loss_orig.data) if reduce else loss_orig.data,
            "nll_loss": utils.item(nll_loss_orig.data) if reduce else nll_loss_orig.data,
            "ntokens": n_tokens,
            "nsentences": n_sentences,
            "sample_size": sample_size,
            "aux_loss": utils.item(loss_aux.data) if reduce else loss_aux.data,
            "aux_nll_loss": utils.item(nll_loss_aux.data) if reduce else nll_loss_aux.data,
            "consistency_loss": utils.item(consistency_loss.data) if reduce else consistency_loss.data
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output_orig, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        if self.ctc_criterion.all_ctc_weight > 0:
            ctc_loss, logging_output = self.ctc_criterion.compute_ctc_loss(model, sample, encoder_out_orig, logging_output)
            loss = ctc_loss + loss_orig + self.auxiliary_branch_loss_weight * loss_aux + self.consistency_weight * consistency_loss

        logging_output["loss"] = utils.item(loss.data) if reduce else loss.data

        return loss, sample_size, logging_output

    def compute_uncertainty(self, lprobs, target, ignore_index):
        entropy = -torch.sum((torch.exp(lprobs) * lprobs), dim=-1)
        idx = ~target.eq(ignore_index)
        entropy = torch.mean(entropy[idx]) / math.log(lprobs.shape[-1])
        return entropy.detach().item()

    def get_replacement_probs(self, model, net_output_orig, sample, ignore_index):
        target = model.get_targets(sample, net_output_orig)
        bsz = len(target)
        if model.encoder.replacement_probability_strategy == 'fix':
            return [model.encoder.replacement_probability] * bsz
        elif model.encoder.replacement_probability_strategy == 'dynamic':
            gamma = model.encoder.uncertainty_gamma
            lprobs = model.get_normalized_probs(net_output_orig, log_probs=True)
            probs = [self.compute_uncertainty(lprobs[i], target[i], ignore_index) * gamma for i in range(bsz)]
            return probs
        else:
            raise NotImplementedError

    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def compute_consistency_loss(self, lprobs_orig, lprobs_aux, target, ignore_index):
        pad_mask = target.eq(ignore_index)
        if self.consistency_type == 'jsd':
            lprobs_avg = torch.logsumexp(torch.stack([lprobs_orig, lprobs_aux], dim=0), dim=0) - math.log(2)
            kl_loss_orig = F.kl_div(lprobs_avg, lprobs_orig, log_target=True, reduction="none").sum(-1)
            kl_loss_aux = F.kl_div(lprobs_avg, lprobs_aux, log_target=True, reduction="none").sum(-1)
            kl_loss_orig.masked_fill_(pad_mask, 0.0)
            kl_loss_aux.masked_fill_(pad_mask, 0.0)
            kl_loss_orig = kl_loss_orig.sum()
            kl_loss_aux = kl_loss_aux.sum()
            consistency_loss = (kl_loss_orig + kl_loss_aux) / 2.0
            return consistency_loss
        elif self.consistency_type == 'jsd-detach':
            lprobs_avg = torch.logsumexp(torch.stack([lprobs_orig, lprobs_aux], dim=0), dim=0) - math.log(2)
            lprobs_avg = lprobs_avg.detach()
            kl_loss_orig = F.kl_div(lprobs_avg, lprobs_orig, log_target=True, reduction="none").sum(-1)
            kl_loss_aux = F.kl_div(lprobs_avg, lprobs_aux, log_target=True, reduction="none").sum(-1)
            kl_loss_orig.masked_fill_(pad_mask, 0.0)
            kl_loss_aux.masked_fill_(pad_mask, 0.0)
            kl_loss_orig = kl_loss_orig.sum()
            kl_loss_aux = kl_loss_aux.sum()
            consistency_loss = (kl_loss_orig + kl_loss_aux) / 2.0
            return consistency_loss
        elif self.consistency_type == 'bi-KL':
            kl_loss_orig = F.kl_div(lprobs_aux, lprobs_orig, log_target=True, reduction="none").sum(-1)
            kl_loss_aux = F.kl_div(lprobs_orig, lprobs_aux, log_target=True, reduction="none").sum(-1)
            kl_loss_orig.masked_fill_(pad_mask, 0.0)
            kl_loss_aux.masked_fill_(pad_mask, 0.0)
            kl_loss_orig = kl_loss_orig.sum()
            kl_loss_aux = kl_loss_aux.sum()
            bi_kl_loss = (kl_loss_orig + kl_loss_aux) / 2.0
            return bi_kl_loss
        elif self.consistency_type == 'aux2orig-KL':
            kl_loss_orig = F.kl_div(lprobs_aux, lprobs_orig, log_target=True, reduction="none").sum(-1)
            kl_loss_orig.masked_fill_(pad_mask, 0.0)
            kl_loss_orig = kl_loss_orig.sum()
            return kl_loss_orig
        elif self.consistency_type == 'aux2orig-KL-detach':
            kl_loss_orig = F.kl_div(lprobs_aux.detach(), lprobs_orig, log_target=True, reduction="none").sum(-1)
            kl_loss_orig.masked_fill_(pad_mask, 0.0)
            kl_loss_orig = kl_loss_orig.sum()
            return kl_loss_orig
        elif self.consistency_type == 'orig2aux-KL':
            kl_loss_aux = F.kl_div(lprobs_orig, lprobs_aux, log_target=True, reduction="none").sum(-1)
            kl_loss_aux.masked_fill_(pad_mask, 0.0)
            kl_loss_aux = kl_loss_aux.sum()
            return kl_loss_aux
        elif self.consistency_type == 'orig2aux-KL-detach':
            kl_loss_aux = F.kl_div(lprobs_orig.detach(), lprobs_aux, log_target=True, reduction="none").sum(-1)
            kl_loss_aux.masked_fill_(pad_mask, 0.0)
            kl_loss_aux = kl_loss_aux.sum()
            return kl_loss_aux
        else:
            raise NotImplementedError

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        trans_loss_sum = utils.item(
            sum(log.get("trans_loss", 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        enc_loss_sum = utils.item(
            sum(log.get("encoder_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if trans_loss_sum != loss_sum:
            metrics.log_scalar(
                "trans_loss", trans_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        if "aux_loss" in logging_outputs[0]:
            aux_loss_sum = utils.item(
                sum(log.get("aux_loss", 0) for log in logging_outputs)
            )
            metrics.log_scalar(
                "aux_loss", aux_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_derived(
                "aux-loss/orig-loss", lambda meters: meters["aux_loss"].avg / meters["trans_loss"].avg
            )
        if "aux_nll_loss" in logging_outputs[0]:
            aux_nll_loss_sum = utils.item(
                sum(log.get("aux_nll_loss", 0) for log in logging_outputs)
            )
            metrics.log_scalar(
                "aux_nll_loss", aux_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        if "consistency_loss" in logging_outputs[0]:
            consistency_loss_sum = utils.item(
                sum(log.get("consistency_loss", 0) for log in logging_outputs)
            )
            metrics.log_scalar(
                "consistency_loss", consistency_loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if enc_loss_sum != 0:
            metrics.log_scalar("enc_loss", enc_loss_sum, sample_size, round=3)

        if "ctc_loss" in logging_outputs[0] or "all_ctc_loss" in logging_outputs[0]:
            CtcCriterion.reduce_metrics(logging_outputs)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
