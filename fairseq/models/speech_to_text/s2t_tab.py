import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq import checkpoint_utils, utils
from fairseq.models.speech_to_text.utils import save_to_dict
from fairseq.models.transformer import Embedding
from fairseq.models import FairseqEncoder, register_model, register_model_architecture
from fairseq.models.speech_to_text import S2TTransformerModel, S2TTransformerEncoder
from fairseq.modules import FairseqDropout, LayerNorm, PositionalEmbedding, TransformerEncoderLayer
logger = logging.getLogger(__name__)

@register_model("s2t_tab")
class S2TTABModel(S2TTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        S2TTransformerModel.add_args(parser)
        S2TTABModel.add_specific_args(parser)

    @staticmethod
    def add_specific_args(parser):
        # TAB setting
        parser.add_argument(
            "--textual-encoder-no-scale-embedding",
            action="store_true",
            help="if True, dont scale textual encoder embeddings",
        )
        parser.add_argument(
            "--textual-encoder-embed-norm",
            action="store_true",
            help="if True, dont scale embeddings in textual encoder",
        )
        parser.add_argument(
            "--text-encoder-layers",
            default=6,
            type=int,
            help="layers of the text encoder",
        )
        parser.add_argument(
            "--text-attention-type",
            default="selfattn",
            type=str,
            help="attention type of the textual encoder",
        )
        parser.add_argument(
            "--replacement-probability-strategy",
            default="fix",
            type=str,
            help="The strategy of replacement probability p*, which can be fix or dynamic",
        )
        parser.add_argument(
            "--replacement-probability",
            default=0.2,
            type=float,
            help="The replacement probability p*, when 'strategy' is 'fix' ",
        )
        parser.add_argument(
            "--uncertainty-gamma",
            default=0.5,
            type=float,
            help="The gamma of uncertainty based replacement probability, when 'strategy' is 'dynamic' ",
        )
        parser.add_argument(
            "--acoustic-encoder",
            default="transformer",
            type=str,
            help="the architecture of the acoustic encoder",
        )
        parser.add_argument(
            "--load-pretrained-acoustic-encoder-from",
            type=str,
            metavar="STR",
            help="model to take acoustic encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-text-encoder-from",
            type=str,
            metavar="STR",
            help="model to take text encoder weights from (for initialization)",
        )

    @classmethod
    def build_encoder(cls, args, task=None, decoder_embed_tokens=None):
        encoder = S2TTABEncoder(args, task, decoder_embed_tokens)

        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_encoder_from}"
            )
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder,
                checkpoint=args.load_pretrained_encoder_from,
                strict=False
            )

        if getattr(args, "load_pretrained_acoustic_encoder_from", None):
            logger.info(
                f"loaded pretrained acoustic encoder from: "
                f"{args.load_pretrained_acoustic_encoder_from}"
            )
            encoder.acoustic_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.acoustic_encoder,
                checkpoint=args.load_pretrained_acoustic_encoder_from,
                strict=False
            )

        if getattr(args, "load_pretrained_text_encoder_from", None):
            logger.info(
                f"loaded pretrained text encoder from: "
                f"{args.load_pretrained_text_encoder_from}"
            )
            encoder.textual_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder.textual_encoder,
                checkpoint=args.load_pretrained_text_encoder_from,
                strict=False
            )

        return encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args, task, decoder_embed_tokens)
        if getattr(args, "encoder_freeze_module", None):
            utils.freeze_parameters(encoder, args.encoder_freeze_module)
            logging.info("freeze the encoder module: {}".format(args.encoder_freeze_module))

        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        if getattr(args, "decoder_freeze_module", None):
            utils.freeze_parameters(decoder, args.decoder_freeze_module)
            logging.info("freeze the decoder module: {}".format(args.decoder_freeze_module))

        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


class TextualEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens=None):

        super().__init__(None)

        self.register_buffer("version", torch.Tensor([3]))  # for consistent
        embed_dim = args.encoder_embed_dim
        layer_num = args.text_encoder_layers
        self.layer_num = layer_num
        self.embed_tokens = embed_tokens

        self.embed_scale = math.sqrt(embed_dim)
        if args.textual_encoder_no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = dictionary.pad_index

        self.encoder_embed_norm = getattr(args, "textual_encoder_embed_norm", False)
        if self.encoder_embed_norm:
            self.embed_ln = LayerNorm(embed_dim)

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(layer_num)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        if self.encoder_embed_norm:
            x = self.embed_ln(x)
        x = self.embed_scale * x
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x = positions + x
        x = self.dropout_module(x)
        for layer in self.layers:
            x = layer(x, encoder_padding_mask, pos_emb=positions)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        pass


class S2TTABEncoder(FairseqEncoder):
    def __init__(self, args, task=None, decoder_embed_tokens=None):
        super().__init__(None)

        # acoustic encoder
        self.acoustic_encoder = S2TTransformerEncoder(args, task, decoder_embed_tokens)
        self.blank_idx = 0
        self.src_dict = task.source_dictionary
        self.replacement_probability_strategy = args.replacement_probability_strategy

        if self.replacement_probability_strategy == 'fix':
            self.replacement_probability = args.replacement_probability
            logger.info("Copy and replacement by fix ratio: {}".format(self.replacement_probability))
        elif self.replacement_probability_strategy == 'dynamic':
            self.uncertainty_gamma = args.uncertainty_gamma
            logger.info("Copy and replacement by dynamic ratio, gamma is: {}".format(self.uncertainty_gamma))
        else:
            raise NotImplementedError
        self.pt_embeddings = nn.Embedding(
            len(task.source_dictionary), args.encoder_embed_dim, task.source_dictionary.pad_index
        )

        self.pt_embeddings.weight = decoder_embed_tokens.weight

        acoustic_encoder_attention_type = args.encoder_attention_type
        args.encoder_attention_type = args.text_attention_type
        # textual encoder
        self.textual_encoder = TextualEncoder(args, task.source_dictionary, decoder_embed_tokens)

        args.encoder_attention_type = acoustic_encoder_attention_type

    def shrink(self, out, ctc_logit, padding):
        T, B, D = out.shape
        distribution = F.softmax(ctc_logit, dim=-1)
        lengths = (~padding).long().sum(-1)
        with torch.no_grad():
            ctc_alignment = []
            prob_ctc = distribution.transpose(0, 1)  # T x B x V -> B x T x V
            for b in range(B):
                pred, counts = torch.unique_consecutive(
                    prob_ctc[b][: lengths[b]].argmax(-1), return_counts=True)
                ctc_alignment.append([pred, counts])

            new_lengths = [len(a) for a, _ in ctc_alignment]
            # B x T x T'
            weights_matrix = torch.zeros(B, T, max(new_lengths), dtype=prob_ctc.dtype)
            for b, (_, counts) in enumerate(ctc_alignment):
                processed_inputs_cnt = 0
                for t, same in enumerate(counts.tolist()):
                    new_processed_inputs_cnt = processed_inputs_cnt + same
                    weights_matrix[b, processed_inputs_cnt:new_processed_inputs_cnt, t] = 1.0 / same
                    processed_inputs_cnt = new_processed_inputs_cnt
            weights_matrix = weights_matrix.to(prob_ctc.device)

        out = out.permute(1, 2, 0) # T x B x C -> B x C x T
        compressed_output = out.bmm(weights_matrix).type_as(out)  # B x D x T'
        shrunk = compressed_output.permute(2, 0, 1)
        out_lengths = lengths.new(new_lengths)
        padding = lengths_to_padding_mask(out_lengths)
        return shrunk, padding, ctc_alignment

    def encode_audio(self, src_tokens, src_lengths=None, **kwargs):
        acoustic_encoder_out = self.acoustic_encoder(src_tokens, src_lengths, **kwargs)
        encoder_out = acoustic_encoder_out["encoder_out"][0]
        encoder_padding_mask = acoustic_encoder_out["encoder_padding_mask"][0]
        ctc_padding_mask = encoder_padding_mask

        if "ctc_logit" in acoustic_encoder_out and len(acoustic_encoder_out["ctc_logit"]) > 0:
            ctc_logit = acoustic_encoder_out["ctc_logit"][0]
        else:
            ctc_logit = None

        assert ctc_logit is not None
        audio_out, encoder_padding_mask, ctc_alignment = self.shrink(encoder_out, ctc_logit, ctc_padding_mask)

        ctc_out = {
            "ctc_alignment": [ctc_alignment],
            "ctc_logit": [ctc_logit],  # T x B x D
            "ctc_padding_mask": [ctc_padding_mask],  # B x T
            "interleaved_ctc_logits": acoustic_encoder_out.get("interleaved_ctc_logits", []),  # B x T x D
        }

        return audio_out, encoder_padding_mask, ctc_out

    def encode_text(self, x, padding_mask):
        x = self.textual_encoder(x, padding_mask)
        return x

    def get_auxiliary_branch(self, audio_out, ctc_alignment, probs):
        aux = audio_out.clone()
        for b in range(audio_out.size(1)):
            alignment_embeddings = self.pt_embeddings(ctc_alignment[b][0])
            length = alignment_embeddings.size(0)
            selected = (torch.rand(length) < probs[b]) & (ctc_alignment[b][0] != self.blank_idx).cpu()
            aux[:length, b, :][selected] = alignment_embeddings[selected]
        aux = aux.detach()
        return aux

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        audio_out, encoder_padding_mask, ctc_out = self.encode_audio(src_tokens, src_lengths)
        encoder_out = self.encode_text(audio_out, encoder_padding_mask)
        encoder_out = save_to_dict(encoder_out, encoder_padding_mask)
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            [] if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            [] if len(encoder_out["encoder_padding_mask"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]]
        )

        new_encoder_embedding = (
            [] if len(encoder_out["encoder_embedding"]) == 0
            else [x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


@register_model_architecture(model_name="s2t_tab", arch_name="s2t_tab")
def base_architecture(args):
    # Convolutional subsampler
    args.subsampling_type = getattr(args, "subsampling_type", "conv1d")
    args.subsampling_layers = getattr(args, "subsampling_layers", 2)
    args.subsampling_filter = getattr(args, "subsampling_filter", 1024)
    args.subsampling_kernel = getattr(args, "subsampling_kernel", 5)
    args.subsampling_stride = getattr(args, "subsampling_stride", 2)
    args.subsampling_norm = getattr(args, "subsampling_norm", "none")
    args.subsampling_activation = getattr(args, "subsampling_activation", "glu")

    # Transformer
    args.acoustic_encoder = getattr(args, "acoustic_encoder", "transformer")
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_type = getattr(args, "encoder_attention_type", "selfattn")
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_type = getattr(args, "decoder_attention_type", "selfattn")
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", args.dropout)
    args.activation_dropout = getattr(args, "activation_dropout", args.dropout)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.encoder_no_scale_embedding = getattr(args, "encoder_no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)

    args.encoder_embed_linear = getattr(args, "encoder_embed_linear", False)
    args.encoder_embed_norm = getattr(args, "encoder_embed_norm", False)

    # CTC
    args.ctc_layer = getattr(args, "ctc_layer", 0)
    args.share_ctc_and_embed = getattr(args, "share_ctc_and_embed", False)

    # Conformer
    args.encoder_activation_fn = getattr(args, "encoder_activation_fn", "relu")
    args.macaron_style = getattr(args, "macaron_style", False)
    args.use_cnn_module = getattr(args, "use_cnn_module", False)
    args.cnn_module_kernel = getattr(args, "cnn_module_kernel", 31)
    args.cnn_module_norm = getattr(args, "cnn_module_norm", "batch_norm")

    # Relative position encoding
    args.max_encoder_relative_length = getattr(args, 'max_encoder_relative_length', -1)
    args.max_decoder_relative_length = getattr(args, 'max_decoder_relative_length', -1)
    args.k_only = getattr(args, 'k_only', True)

    # local modeling
    args.hard_mask_window = getattr(args, 'hard_mask_window', 0)
    args.gauss_mask_sigma = getattr(args, 'gauss_mask_sigma', 0)
    args.init_mask_weight = getattr(args, 'init_mask_weight', 0)

    # interleaved CTC
    args.interleaved_ctc_layers = getattr(args, "interleaved_ctc_layers", None)
    args.interleaved_ctc_temperature = getattr(args, "interleaved_ctc_temperature", 1)
    args.interleaved_ctc_drop_prob = getattr(args, "interleaved_ctc_drop_prob", 0)

    # Text Enocder
    args.text_encoder_layers = getattr(args, "text_encoder_layers", 6)
    args.text_attention_type = getattr(args, "text_attention_type", "selfattn")
    args.textual_encoder_no_scale_embedding = getattr(args, "textual_encoder_no_scale_embedding", False)
    args.textual_encoder_embed_norm = getattr(args, "textual_encoder_embed_norm", False)
