# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional
import numpy as np
import logging
import editdistance
import os
import sys

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round

logger = logging.getLogger(__name__)


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=True,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="sentencepiece",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
                    "wordpiece, BPE symbols, etc. "
                    "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC loss"},
    )
    ctc_entropy: float = field(
        default=0.0,
        metadata={"help": "weight of CTC entropy"},
    )
    ctc_entropy_cutoff: int = field(
        default=0,
        metadata={"help": "cutoff for CTC entropy computation"},
    )
    interleaved_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of interleaved CTC loss"},
    )

    aligned_target_ctc: bool = field(
        default=False,
        metadata={"help": "calculate target ctc by aligned text"},
    )
    target_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of CTC loss for target sentence"},
    )
    target_interleaved_ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight of interleaved CTC loss for target sentence"},
    )

    cal_all_ctc: bool = field(
        default=False,
        metadata={"help": "calculate all ctc results"},
    )

    ctc_self_distill_weight: float = field(
        default=0.0,
        metadata={"help": "weight of the self distillation CTC loss"},
    )
    target_ctc_self_distill_weight: float = field(
        default=0.0,
        metadata={"help": "weight of the self distillation CTC loss for target sentence"},
    )
    ctc_self_distill_prob: float = field(
        default=0.1,
        metadata={"help": "probability to use distillation loss"},
    )
    ctc_self_distill_temperature: float = field(
        default=1,
        metadata={"help": "temperature for ctc self distillation"},
    )

    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )


@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(self, cfg: CtcCriterionConfig, task: FairseqTask, ctc_weight=1.0, save_dir=None):
        super().__init__(task)

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.blank_idx = task.target_dictionary.index(task.blank_symbol) if hasattr(task, 'blank_symbol') else 0
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process
        self.sentence_avg = cfg.sentence_avg
        self.save_dir = save_dir

        self.cal_all_ctc = cfg.cal_all_ctc
        self.ctc_weight = ctc_weight
        self.interleaved_ctc_weight = cfg.interleaved_ctc_weight
        self.aligned_target_ctc = cfg.aligned_target_ctc
        self.target_ctc_weight = cfg.target_ctc_weight
        self.target_interleaved_ctc_weight = cfg.target_interleaved_ctc_weight

        self.ctc_self_distill_weight = cfg.ctc_self_distill_weight
        self.target_ctc_self_distill_weight = float(cfg.target_ctc_self_distill_weight)
        self.ctc_self_distill_prob = float(cfg.ctc_self_distill_prob)
        self.ctc_self_distill_temperature = float(cfg.ctc_self_distill_temperature)

        self.ctc_entropy = cfg.ctc_entropy
        self.ctc_entropy_cutoff = cfg.ctc_entropy_cutoff

        self.all_ctc_weight = self.ctc_weight + self.interleaved_ctc_weight + \
                              self.target_ctc_weight + self.target_interleaved_ctc_weight + \
                              self.ctc_self_distill_weight + self.target_ctc_self_distill_weight + \
                              self.ctc_entropy

        if self.all_ctc_weight > 0:
            self.ctc_loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction="sum", zero_infinity=True)

        self.ctc_names = []
        self.use_ctc = (self.ctc_weight + self.interleaved_ctc_weight > 0)
        self.use_target_ctc = (self.target_ctc_weight + self.target_interleaved_ctc_weight > 0)
        self.use_source_distill = self.use_target_distill = False
        if self.ctc_self_distill_prob > 0:
            if self.ctc_self_distill_weight:
                self.use_source_distill = True
            if self.target_ctc_self_distill_weight > 0:
                self.use_target_distill = True

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])
        ntokens = sample["ntokens"]
        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens

        logging_output = {
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        loss, logging_output = self.compute_ctc_loss(model, sample, net_output, logging_output)
        return loss, sample_size, logging_output

    def get_ground_truth_alignment(self, model, sample):
        ctc_alignment_oracle = dict()

        try:
            from fairseq.torch_imputer import best_alignment, imputer_loss
        except ImportError:
            logger.warning("Imputer is not available.")

        src_tokens, src_lengths, prev_output_tokens = sample["net_input"].values()
        with torch.no_grad():
            encoder_out = model.encoder(src_tokens, src_lengths)
            ctc_logit = None
            if "ctc_logit" in encoder_out and len(encoder_out["ctc_logit"]) != 0:
                ctc_logit = encoder_out["ctc_logit"][0]
            elif "interleaved_ctc_logits" in encoder_out and len(encoder_out["interleaved_ctc_logits"]) != 0:
                ctc_logit = encoder_out["interleaved_ctc_logits"][-1]

            ctc_alignment_oracle["source"] = None
            if ctc_logit is not None:
                if "transcript" in sample:
                    tokens = sample["transcript"]["tokens"]
                else:
                    tokens = sample["target"]
                pad_mask = (tokens != self.pad_idx) & (tokens != self.eos_idx)
                target_lengths = pad_mask.sum(-1)

                if "ctc_padding_mask" in encoder_out:
                    non_padding_mask = ~encoder_out["ctc_padding_mask"][0]
                else:
                    non_padding_mask = ~encoder_out["encoder_padding_mask"][0]
                input_lengths = non_padding_mask.long().sum(-1)

                best_aligns = best_alignment(ctc_logit.float(), tokens, input_lengths, target_lengths,
                                             self.pad_idx, zero_infinity=True)
                best_aligns_pad = torch.tensor([a + [0] * (ctc_logit.size(0) - len(a)) for a in best_aligns],
                                               device=ctc_logit.device, dtype=tokens.dtype)
                oracle_pos = torch.div(best_aligns_pad, 2, rounding_mode='floor').clip(max=tokens.shape[1] - 1)
                oracle = tokens.gather(-1, oracle_pos)
                source_oracle = oracle.masked_fill(best_aligns_pad % 2 == 0, self.blank_idx)

                ctc_alignment_oracle["source"] = [source_oracle, best_aligns_pad]

            ctc_alignment_oracle["target"] = None
            target_ctc_logit = None
            if "target_ctc_logit" in encoder_out and len(encoder_out["target_ctc_logit"]) != 0:
                target_ctc_logit = encoder_out["ctc_logit"][0]
            elif "target_interleaved_ctc_logits" in encoder_out and len(
                    encoder_out["target_interleaved_ctc_logits"]) != 0:
                target_ctc_logit = encoder_out["target_interleaved_ctc_logits"][-1]

            if target_ctc_logit is not None:
                target_tokens = sample["target"]
                target_pad_mask = (target_tokens != self.pad_idx) & (target_tokens != self.eos_idx)
                target_lengths = target_pad_mask.sum(-1)

                if "ctc_padding_mask" in encoder_out:
                    non_padding_mask = ~encoder_out["ctc_padding_mask"][0]
                else:
                    non_padding_mask = ~encoder_out["encoder_padding_mask"][0]
                input_lengths = non_padding_mask.long().sum(-1)

                best_aligns = best_alignment(target_ctc_logit.float(), target_tokens, input_lengths, target_lengths,
                                             self.pad_idx, zero_infinity=True)
                best_aligns_pad = torch.tensor([a + [0] * (ctc_logit.size(0) - len(a)) for a in best_aligns],
                                               device=target_ctc_logit.device, dtype=target_tokens.dtype)
                oracle_pos = (best_aligns_pad // 2).clip(max=tokens.shape[1] - 1)
                oracle = tokens.gather(-1, oracle_pos)
                target_oracle = oracle.masked_fill(best_aligns_pad % 2 == 0, self.blank_idx)
                ctc_alignment_oracle["target"] = [target_oracle, best_aligns_pad]

        return ctc_alignment_oracle

    def get_ctc_loss(self, model, ctc_logit, targets, input_lengths, target_lengths, loss_coef):
        lprobs = model.get_normalized_probs(
            [ctc_logit], log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        lprobs.batch_first = False

        loss = 0
        with torch.backends.cudnn.flags(enabled=False):
            for item_targets, item_target_lengths, item_coef in zip(targets, target_lengths, loss_coef):
                loss += self.ctc_loss(
                    lprobs,
                    item_targets,
                    input_lengths,
                    item_target_lengths,
                ) * item_coef
        return loss, lprobs

    @staticmethod
    def get_ctc_self_distill_loss(distill_num, teacher_logit, student_logits, non_padding_mask):
        ctc_self_distill_loss = 0
        ctc_self_distill_num = 0
        for i in range(distill_num):
            logit = student_logits[i]
            if type(logit) == list:
                student_logit = logit[0]
                non_padding_mask = ~logit[1]
            else:
                student_logit = logit

            if student_logit.size() != teacher_logit.size():
                continue

            loss = F.kl_div(
                F.log_softmax(student_logit, dim=-1, dtype=torch.float32),
                F.log_softmax(teacher_logit, dim=-1, dtype=torch.float32),
                # F.log_softmax(teacher_logit.detach(), dim=-1, dtype=torch.float32),
                log_target=True,
                reduction="none",
            )
            ctc_self_distill_loss += loss.sum(-1).transpose(0, 1).masked_fill_(~non_padding_mask, 0.0).sum()
            ctc_self_distill_num += 1
        return ctc_self_distill_num, ctc_self_distill_loss

    def get_target_text(self, sample):
        if self.aligned_target_ctc and "aligned_target" in sample:
            return sample["aligned_target"]["tokens"]
        else:
            return sample["target"]

    def compute_ctc_loss(self, model, sample, net_output, logging_output):
        if "transcript" in sample:
            tokens = sample["transcript"]["tokens"]
        else:
            tokens = sample["target"]
        if "ctc_padding_mask" in net_output:
            non_padding_mask = ~net_output["ctc_padding_mask"][0]
        else:
            non_padding_mask = ~net_output["encoder_padding_mask"][0]

        input_lengths = non_padding_mask.long().sum(-1)

        pad_mask = (tokens != self.pad_idx) & (tokens != self.eos_idx)

        if "mixup" in net_output and net_output["mixup"] is not None:
            mixup = True
            mixup_coef = net_output["mixup"]["coef"]
            mixup_idx1 = net_output["mixup"]["index1"]
            mixup_idx2 = net_output["mixup"]["index2"]

            mask1 = pad_mask[mixup_idx1]
            mask2 = pad_mask[mixup_idx2]
            transcripts1 = tokens[[mixup_idx1]].masked_select(mask1)
            transcripts2 = tokens[mixup_idx2].masked_select(mask2)
            transcript_lengths1 = mask1.sum(-1)
            transcript_lengths2 = mask2.sum(-1)
            transcripts = [transcripts1, transcripts2]
            transcript_lengths = [transcript_lengths1, transcript_lengths2]
            loss_coef = [mixup_coef, 1 - mixup_coef]
        else:
            mixup = False
            transcripts = [tokens.masked_select(pad_mask)]
            transcript_lengths = [pad_mask.sum(-1)]
            loss_coef = [1]

        all_ctc_logits = dict()
        self.ctc_names = []
        lprobs = None
        target_lprobs = None

        interleaved_ctc_num = 0
        interleaved_ctc_loss = 0
        if "interleaved_ctc_logits" in net_output:
            interleaved_ctc_num = len(net_output["interleaved_ctc_logits"])

        # calculate the interleaved CTC loss
        if self.interleaved_ctc_weight > 0 and interleaved_ctc_num > 0:
            logits = net_output["interleaved_ctc_logits"]
            for i in range(interleaved_ctc_num):
                logit = logits[i]
                if type(logit) == list:
                    inter_ctc_logit = logit[0]
                    inter_input_lengths = (~logit[1]).long().sum(-1)
                else:
                    inter_ctc_logit = logit
                    inter_input_lengths = input_lengths

                all_ctc_logits["interleaved_ctc_logit%d" % i] = [inter_ctc_logit, inter_input_lengths]
                inter_loss, inter_lprobs = self.get_ctc_loss(
                    model, inter_ctc_logit, transcripts, inter_input_lengths, transcript_lengths, loss_coef)
                interleaved_ctc_loss += inter_loss
                lprobs = inter_lprobs

            interleaved_ctc_loss /= interleaved_ctc_num
            logging_output["interleaved_ctc_loss"] = utils.item(interleaved_ctc_loss.data)

        ctc_loss = 0
        ctc_entropy = 0
        if self.ctc_weight > 0 and "ctc_logit" in net_output and len(net_output["ctc_logit"]) > 0:
            ctc_logit = net_output["ctc_logit"][0]
            all_ctc_logits["ctc_logit"] = [ctc_logit, input_lengths]

            ctc_loss, lprobs = self.get_ctc_loss(
                model, ctc_logit, transcripts, input_lengths, transcript_lengths, loss_coef)

            if self.ctc_entropy > 0:
                if self.ctc_entropy_cutoff != 0:
                    cut_ctc_logit = ctc_logit.sort(dim=-1, descending=True)[0][:, :, 0:self.ctc_entropy_cutoff]
                    cut_ctc_logit = cut_ctc_logit / cut_ctc_logit.sum(dim=-1, keepdim=True)
                    ctc_entropy = Categorical(logits=cut_ctc_logit).entropy().sum()
                else:
                    ctc_entropy = Categorical(logits=ctc_logit).entropy().sum()

                logging_output["ctc_entropy"] = utils.item(ctc_entropy.data)
            logging_output["ctc_loss"] = utils.item(ctc_loss.data)

        # calculate the target CTC loss
        target_ctc_loss = 0
        target_interleaved_ctc_loss = 0
        target_interleaved_ctc_num = 0
        if self.use_target_ctc:
            target_tokens = self.get_target_text(sample)
            target_pad_mask = (target_tokens != self.pad_idx) & (target_tokens != self.eos_idx)
            target_no_padding_mask = ~target_pad_mask

            if mixup:
                mask1 = target_pad_mask[mixup_idx1]
                mask2 = target_pad_mask[mixup_idx2]
                target_tokens1 = target_tokens.masked_select(mask1)
                target_tokens2 = target_tokens.masked_select(mask2)
                target_lengths1 = mask1.sum(-1)
                target_lengths2 = mask2.sum(-1)
                target_tokens = [target_tokens1, target_tokens2]
                target_lengths = [target_lengths1, target_lengths2]
                loss_coef = [mixup_coef, 1 - mixup_coef]
            else:
                target_tokens = [target_tokens.masked_select(target_pad_mask)]
                target_lengths = [target_pad_mask.sum(-1)]
                loss_coef = [1]

            if "target_interleaved_ctc_logits" in net_output:
                target_interleaved_ctc_num = len(net_output["target_interleaved_ctc_logits"])
            if target_interleaved_ctc_num != 0 and self.target_interleaved_ctc_weight > 0:
                logits = net_output["target_interleaved_ctc_logits"]
                for i in range(target_interleaved_ctc_num):
                    logit = logits[i]
                    if type(logit) == list:
                        target_inter_ctc_logit = logit[0]
                        padding = ~logit[1]
                        inter_input_lengths = padding.long().sum(-1)
                    else:
                        target_inter_ctc_logit = logit
                        inter_input_lengths = input_lengths

                    all_ctc_logits["target_interleaved_ctc_logit%d" % i] = [target_inter_ctc_logit, inter_input_lengths]
                    inter_loss, target_inter_lprobs = self.get_ctc_loss(
                        model, target_inter_ctc_logit, target_tokens, inter_input_lengths, target_lengths, loss_coef)
                    target_interleaved_ctc_loss += inter_loss
                    target_lprobs = target_inter_lprobs

                target_interleaved_ctc_loss /= target_interleaved_ctc_num
                logging_output["target_interleaved_ctc_loss"] = utils.item(target_interleaved_ctc_loss.data)

            if self.target_ctc_weight > 0:
                assert "target_ctc_logit" in net_output
                target_ctc_logit = net_output["target_ctc_logit"][0]
                all_ctc_logits["target_ctc_logit"] = [target_ctc_logit, input_lengths]

                target_ctc_loss, target_lprobs = self.get_ctc_loss(
                    model, target_ctc_logit, target_tokens, input_lengths, target_lengths, loss_coef)
                logging_output["target_ctc_loss"] = utils.item(target_ctc_loss.data)

        # calculate the self distillation CTC loss
        ctc_self_distill_loss = 0
        if self.use_source_distill or self.use_target_distill:
            ctc_self_distill_choice = torch.rand(1).uniform_()

            cal_source_distill = cal_target_distill = False
            if not self.training:
                cal_source_distill = True if self.use_source_distill else False
                cal_target_distill = True if self.use_target_distill else False
            else:
                if ctc_self_distill_choice <= self.ctc_self_distill_prob:
                    if self.use_source_distill and self.use_target_distill:
                        cal_source_distill = True if ctc_self_distill_choice > self.ctc_self_distill_prob / 2 else False
                        cal_target_distill = not cal_source_distill
                    else:
                        cal_source_distill = self.use_source_distill
                        cal_target_distill = self.use_target_distill

            # source self distillation
            if cal_source_distill:
                ctc_self_distill_num = 0
                non_padding = non_padding_mask

                # if self.ctc_weight > 0 and self.ctc_self_distill_weight > 0 and interleaved_ctc_num > 0:
                if self.ctc_self_distill_weight > 0 and interleaved_ctc_num > 0:
                    teacher_logit = ctc_logit
                    student_logits = net_output["interleaved_ctc_logits"]
                    ctc_self_distill_num = interleaved_ctc_num
                elif self.ctc_self_distill_weight > 0 and interleaved_ctc_num > 1:
                    teacher_logit = net_output["interleaved_ctc_logits"][-1]
                    student_logits = net_output["interleaved_ctc_logits"][:-1]
                    ctc_self_distill_num = interleaved_ctc_num - 1

                if ctc_self_distill_num != 0:
                    ctc_self_distill_num, source_ctc_self_distill_loss = \
                        self.get_ctc_self_distill_loss(
                            ctc_self_distill_num, teacher_logit, student_logits, non_padding)
                    source_ctc_self_distill_loss /= ctc_self_distill_num
                    logging_output["ctc_self_distill_loss"] = utils.item(source_ctc_self_distill_loss.data)
                    ctc_self_distill_loss += source_ctc_self_distill_loss * self.ctc_self_distill_weight

            # target self distillation
            if cal_target_distill:
                ctc_self_distill_num = 0
                non_padding = non_padding_mask

                if self.target_ctc_weight > 0 and self.target_ctc_self_distill_weight > 0 and target_interleaved_ctc_num > 0:
                    teacher_logit = target_ctc_logit
                    student_logits = net_output["target_interleaved_ctc_logits"]
                    ctc_self_distill_num = target_interleaved_ctc_num
                elif self.target_ctc_self_distill_weight > 0 and target_interleaved_ctc_num > 1:
                    teacher_logit = net_output["target_interleaved_ctc_logits"][-1]
                    student_logits = net_output["target_interleaved_ctc_logits"][:-1]
                    ctc_self_distill_num = target_interleaved_ctc_num - 1

                if ctc_self_distill_num != 0:
                    ctc_self_distill_num, target_ctc_self_distill_loss = \
                        self.get_ctc_self_distill_loss(
                            ctc_self_distill_num, teacher_logit, student_logits, non_padding)

                    target_ctc_self_distill_loss /= ctc_self_distill_num
                    logging_output["target_ctc_self_distill_loss"] = utils.item(target_ctc_self_distill_loss.data)
                    ctc_self_distill_loss += target_ctc_self_distill_loss * self.target_ctc_self_distill_weight

        loss = \
            self.ctc_weight * ctc_loss + \
            self.interleaved_ctc_weight * interleaved_ctc_loss + \
            self.target_ctc_weight * target_ctc_loss + \
            self.target_interleaved_ctc_weight * target_interleaved_ctc_loss + \
            ctc_self_distill_loss + \
            self.ctc_entropy * ctc_entropy

        logging_output["all_ctc_loss"] = utils.item(loss.data)

        if torch.isnan(loss) or torch.isinf(loss) or utils.item(loss.data) < 0:
            # logger.warning("Illegal loss %f!" % loss)
            if self.ctc_weight != 0:
                logger.warning("CTC loss %f!" % ctc_loss)
            if self.interleaved_ctc_weight != 0:
                logger.warning("Intermedia CTC loss %f!" % interleaved_ctc_loss)
            if self.target_ctc_weight != 0:
                logger.warning("Target CTC loss %f!" % target_ctc_loss)

        # CER is not completely accurate and is for reference only.
        if not model.training:
            encoder = model.encoder.encoder if hasattr(model.encoder, "encoder") else model.encoder
            if hasattr(encoder, "ctc_valid"):
                if lprobs is not None:
                    lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
                    if mixup:
                        idx = mixup_idx1 if mixup_coef > 0.5 else mixup_idx2
                        tokens = tokens[idx]

                    c_err, c_len, w_errs, w_len, wv_errs = encoder.ctc_valid(
                        lprobs_t, tokens, input_lengths, self.task.source_dictionary, lang="source")

                    logging_output["wv_errors"] = wv_errs
                    logging_output["w_errors"] = w_errs
                    logging_output["w_total"] = w_len
                    logging_output["c_errors"] = c_err
                    logging_output["c_total"] = c_len

                if target_lprobs is not None:
                    target_lprobs_t = target_lprobs.transpose(0, 1).float().contiguous().cpu()
                    target_tokens = self.get_target_text(sample)
                    if mixup:
                        idx = mixup_idx1 if mixup_coef > 0.5 else mixup_idx2
                        target_tokens = target_tokens[idx]

                    c_err, c_len, w_errs, w_len, wv_errs = model.encoder.ctc_valid(
                        target_lprobs_t, target_tokens, input_lengths, self.task.target_dictionary, lang="target")

                    logging_output["target_wv_errors"] = wv_errs
                    logging_output["target_w_errors"] = w_errs
                    logging_output["target_w_total"] = w_len
                    logging_output["target_c_errors"] = c_err
                    logging_output["target_c_total"] = c_len

                if self.cal_all_ctc:
                    logging_output["save_dir"] = self.save_dir
                    for name, items in all_ctc_logits.items():
                        logit, lengths = items
                        if "target" in name:
                            dictionary = self.task.target_dictionary
                            ctc_tokens = target_tokens
                            lang = "target"
                        else:
                            dictionary = self.task.source_dictionary
                            ctc_tokens = tokens
                            lang = "source"
                        c_err, c_len, w_errs, w_len, wv_errs = model.encoder.ctc_valid(
                            logit, ctc_tokens, lengths, dictionary, lang)
                        cer = c_err * 100 / c_len
                        wer = w_errs * 100 / w_len

                        logging_output["dump_%s_cer" % name] = cer
                        logging_output["dump_%s_wer" % name] = wer

        return loss, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        ctc_loss_sum = utils.item(
            sum(log.get("ctc_loss", 0) for log in logging_outputs)
        )
        ctc_entropy_sum = utils.item(
            sum(log.get("ctc_entropy", 0) for log in logging_outputs)
        )
        inter_ctc_loss_sum = utils.item(
            sum(log.get("interleaved_ctc_loss", 0) for log in logging_outputs)
        )
        target_ctc_loss_sum = utils.item(
            sum(log.get("target_ctc_loss", 0) for log in logging_outputs)
        )
        target_interleaved_ctc_loss_sum = utils.item(
            sum(log.get("target_interleaved_ctc_loss", 0) for log in logging_outputs)
        )
        ctc_self_distill_loss_sum = utils.item(
            sum(log.get("ctc_self_distill_loss", 0) for log in logging_outputs)
        )
        target_ctc_self_distill_loss_sum = utils.item(
            sum(log.get("target_ctc_self_distill_loss", 0) for log in logging_outputs)
        )
        all_ctc_loss_sum = utils.item(
            sum(log.get("all_ctc_loss", 0) for log in logging_outputs)
        )
        # loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        if np.isnan(all_ctc_loss_sum) or np.isinf(all_ctc_loss_sum) or all_ctc_loss_sum < 0:
            logger.warning("Illegal loss %f!" % all_ctc_loss_sum)
        if all_ctc_loss_sum > 0:
            if "loss" not in logging_outputs[0]:
                metrics.log_scalar(
                    "loss",
                    all_ctc_loss_sum / sample_size / math.log(2),
                    sample_size,
                    round=3,
                )
            else:
                if all_ctc_loss_sum != ctc_loss_sum:
                    metrics.log_scalar(
                        "all_ctc_loss",
                        all_ctc_loss_sum / sample_size / math.log(2),
                        sample_size,
                        round=3,
                    )
        if ctc_loss_sum > 0:
            metrics.log_scalar(
                "ctc_loss",
                ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ctc_entropy_sum > 0:
            metrics.log_scalar(
                "ctc_entropy",
                ctc_entropy_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if inter_ctc_loss_sum > 0:
            metrics.log_scalar(
                "interleaved_ctc_loss",
                inter_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if target_ctc_loss_sum > 0:
            metrics.log_scalar(
                "target_ctc_loss",
                target_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if target_interleaved_ctc_loss_sum > 0:
            metrics.log_scalar(
                "target_interleaved_ctc_loss",
                target_interleaved_ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        if ctc_self_distill_loss_sum > 0:
            metrics.log_scalar(
                "ctc_self_distill_loss",
                ctc_self_distill_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if target_ctc_self_distill_loss_sum > 0:
            metrics.log_scalar(
                "target_ctc_self_distill_loss_sum",
                target_ctc_self_distill_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", ctc_loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        # wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        # metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "cer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

        target_c_errors = sum(log.get("target_c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_target_c_errors", target_c_errors)
        target_c_total = sum(log.get("target_c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_target_c_total", target_c_total)
        target_w_errors = sum(log.get("target_w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_target_w_errors", target_w_errors)
        target_w_total = sum(log.get("target_w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_target_w_total", target_w_total)

        if target_c_total > 0:
            metrics.log_derived(
                "target_cer",
                lambda meters: safe_round(
                    meters["_target_c_errors"].sum * 100.0 / meters["_target_c_total"].sum, 3
                )
                if meters["_target_c_total"].sum > 0
                else float("nan"),
            )
        if target_w_total > 0:
            metrics.log_derived(
                "target_wer",
                lambda meters: safe_round(
                    meters["_target_w_errors"].sum * 100.0 / meters["_target_w_total"].sum, 3
                )
                if meters["_target_w_total"].sum > 0
                else float("nan"),
            )

        # save_dir = logging_outputs.get("save_dir", None)
        # if save_dir is not None and os.path.exists(save_dir):
        #     out = open(os.path.join(save_dir, "ctc_results"), "a")
        # else:
        #     out = sys.stdout
        #
        # for key in logging_outputs:
        #     if key.startswith("dump"):
        #         print("%s: %.2f" % (key, logging_outputs[key]), end="\t", file=out)
        # print("", file=out)
        # out.close()
        #
        # out = sys.stdout

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
