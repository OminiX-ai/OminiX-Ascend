"""
Export Qwen3-TTS models to GGUF format for C++ inference.

Components exported:
- Talker LLM: Qwen3 backbone (28 layers, hidden=2048)
- Code Predictor: 5-layer transformer for codec token prediction
- Speaker Encoder: ECAPA-TDNN for speaker embedding extraction
- Speech Tokenizer Encoder: Mimi-based audio encoder
- Speech Tokenizer Decoder: RVQ decoder + transformer + vocoder

Usage:
    python export_qwen_tts.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --output_dir gguf/
    python export_qwen_tts.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --speaker_encoder
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    gguf_py_path = Path(__file__).parent.parent.parent / 'gguf-py'
    if gguf_py_path.exists():
        sys.path.insert(1, str(gguf_py_path))

import gguf


# ============================================================================
# Shared utilities
# ============================================================================

def create_gguf_writer(model_name: str, output_dir: str) -> tuple:
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{model_name}.gguf"
    writer = gguf.GGUFWriter(output_path, model_name)
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name(model_name)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    return writer, output_path


def add_config(writer: gguf.GGUFWriter, cfg: dict):
    """Add configuration key-value pairs to GGUF."""
    for k, v in cfg.items():
        if isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)
        elif isinstance(v, int):
            writer.add_uint32(k, v)
        elif isinstance(v, str):
            writer.add_string(k, v)
        elif isinstance(v, (list, tuple)):
            if len(v) > 0:
                writer.add_array(k, list(v))
        else:
            print(f"  [skip config] {k}: unsupported type {type(v)}")


def add_params(writer: gguf.GGUFWriter, state_dict: dict, prefix: str = ""):
    """Add model parameters (tensors) to GGUF."""
    for name, param in state_dict.items():
        full_name = f"{prefix}{name}" if prefix else name

        # Bias and norm layers → float32
        if param.dim() <= 1:
            param = param.to(torch.float32)
        elif name.endswith(("_norm.weight", "layer_norm.weight",
                            "layernorm.weight", "LayerNorm.weight")):
            param = param.to(torch.float32)

        # Conv weights (3D+) keep shape; others squeeze
        is_conv = name.endswith(".weight") and param.dim() >= 3
        if not is_conv:
            param = param.squeeze()

        writer.add_tensor(full_name, param.cpu().numpy())


def finalize_gguf(writer: gguf.GGUFWriter, output_path: str):
    print(f"  Writing {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done: {size_mb:.2f} MB")


# ============================================================================
# Export: Talker LLM (Qwen3 backbone, without code predictor)
# ============================================================================

def export_talker(model, output_dir: str):
    """Export Talker LLM backbone to GGUF.

    Includes: text_embedding, codec_embedding, 28 transformer layers,
    norm, text_projection, codec_head.
    Excludes: code_predictor (exported separately).
    """
    print("\n[1/5] Exporting Talker LLM...")
    writer, path = create_gguf_writer("qwen_tts_talker", output_dir)

    talker = model.talker
    cfg = talker.config

    add_config(writer, {
        "hidden_size":            cfg.hidden_size,           # 2048
        "num_hidden_layers":      cfg.num_hidden_layers,     # 28
        "num_attention_heads":    cfg.num_attention_heads,    # 16
        "num_key_value_heads":    cfg.num_key_value_heads,    # 8
        "intermediate_size":      cfg.intermediate_size,      # 6144
        "head_dim":               cfg.head_dim,               # 128
        "vocab_size":             cfg.vocab_size,              # 3072
        "text_vocab_size":        cfg.text_vocab_size,         # 151936
        "text_hidden_size":       cfg.text_hidden_size,        # 2048
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta":             float(cfg.rope_theta),
        "rms_norm_eps":           float(cfg.rms_norm_eps),
        "hidden_act":             cfg.hidden_act,
        "num_code_groups":        cfg.num_code_groups,         # 16
        "codec_bos_id":           cfg.codec_bos_id,
        "codec_eos_token_id":     cfg.codec_eos_token_id,
        "codec_pad_id":           cfg.codec_pad_id,
    })

    # Add rope_scaling info
    if cfg.rope_scaling:
        rs = cfg.rope_scaling
        if "mrope_section" in rs:
            writer.add_array("rope_scaling.mrope_section", rs["mrope_section"])
            # Also write as qwen3.rope.dimension_sections (4-element array)
            # so llama.cpp enables MRoPE via use_mrope() → CANN acceleration
            sections = list(rs["mrope_section"])
            while len(sections) < 4:
                sections.append(0)
            writer.add_array("qwen3.rope.dimension_sections",
                             [int(s) for s in sections[:4]])
        writer.add_bool("rope_scaling.interleaved",
                        rs.get("interleaved", False))

    # Add codec_language_id mapping
    if hasattr(cfg, "codec_language_id") and cfg.codec_language_id:
        lang_ids = cfg.codec_language_id
        writer.add_string("codec_language_ids_json", json.dumps(lang_ids))

    # Extract talker state dict, excluding code_predictor
    full_sd = talker.state_dict()
    talker_sd = {k: v for k, v in full_sd.items()
                 if not k.startswith("code_predictor.")}

    print(f"  Talker tensors: {len(talker_sd)}")
    add_params(writer, talker_sd)
    finalize_gguf(writer, path)


# ============================================================================
# Export: Code Predictor (5-layer transformer)
# ============================================================================

def export_code_predictor(model, output_dir: str):
    """Export Code Predictor to GGUF.

    Structure: small_to_mtp_projection, 5 transformer layers,
    15 codec_embeddings, 15 lm_heads, norm.
    """
    print("\n[2/5] Exporting Code Predictor...")
    writer, path = create_gguf_writer("qwen_tts_code_predictor", output_dir)

    cp = model.talker.code_predictor
    cfg = model.talker.config.code_predictor_config

    add_config(writer, {
        "hidden_size":            cfg.hidden_size,           # 1024
        "num_hidden_layers":      cfg.num_hidden_layers,     # 5
        "num_attention_heads":    cfg.num_attention_heads,    # 16
        "num_key_value_heads":    cfg.num_key_value_heads,    # 8
        "intermediate_size":      cfg.intermediate_size,      # 3072
        "head_dim":               cfg.head_dim,               # 128
        "vocab_size":             cfg.vocab_size,              # 2048
        "num_code_groups":        cfg.num_code_groups,         # 16
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta":             float(cfg.rope_theta),
        "rms_norm_eps":           float(cfg.rms_norm_eps),
        "hidden_act":             cfg.hidden_act,
        # Parent talker hidden_size for projection
        "talker_hidden_size":     model.talker.config.hidden_size,  # 2048
    })

    sd = cp.state_dict()
    print(f"  Code Predictor tensors: {len(sd)}")
    add_params(writer, sd)
    finalize_gguf(writer, path)


# ============================================================================
# Export: Speaker Encoder (ECAPA-TDNN)
# ============================================================================

def export_speaker_encoder(model, output_dir: str):
    """Export Speaker Encoder to GGUF.

    Structure: TDNN block, 3x SE-Res2Net blocks, MFA, ASP, FC.
    Output: 2048-dim speaker embedding.
    """
    print("\n[3/5] Exporting Speaker Encoder...")
    writer, path = create_gguf_writer("qwen_tts_speaker_encoder", output_dir)

    se = model.speaker_encoder
    cfg = model.config.speaker_encoder_config

    add_config(writer, {
        "enc_dim":      cfg.enc_dim,       # 2048
        "sample_rate":  cfg.sample_rate,   # 24000
    })

    sd = se.state_dict()
    print(f"  Speaker Encoder tensors: {len(sd)}")
    add_params(writer, sd)
    finalize_gguf(writer, path)


# ============================================================================
# Export: Speech Tokenizer Encoder (Mimi-based)
# ============================================================================

def export_tokenizer_encoder(speech_tokenizer, output_dir: str):
    """Export Speech Tokenizer Encoder to GGUF.

    Structure: Conv encoder layers, encoder transformer, quantizer
    (semantic + acoustic RVQ), downsample.
    """
    print("\n[4/5] Exporting Speech Tokenizer Encoder...")
    writer, path = create_gguf_writer("qwen_tts_tokenizer_enc", output_dir)

    enc = speech_tokenizer.encoder
    enc_cfg = speech_tokenizer.config

    add_config(writer, {
        "encoder_valid_num_quantizers": enc_cfg.encoder_valid_num_quantizers,
        "input_sample_rate":            enc_cfg.input_sample_rate,
        "encode_downsample_rate":       enc_cfg.encode_downsample_rate,
    })

    # Also add encoder-specific config from MimiConfig
    mimi_cfg = enc_cfg.encoder_config
    if mimi_cfg:
        mimi_dict = mimi_cfg if isinstance(mimi_cfg, dict) else mimi_cfg.to_dict()
        for k in ["num_filters", "hidden_size", "num_hidden_layers",
                   "num_attention_heads", "num_key_value_heads",
                   "codebook_size", "codebook_dim", "num_quantizers",
                   "frame_rate", "sampling_rate"]:
            if k in mimi_dict:
                v = mimi_dict[k]
                if isinstance(v, float):
                    writer.add_float32(f"encoder.{k}", v)
                elif isinstance(v, int):
                    writer.add_uint32(f"encoder.{k}", v)

    sd = enc.state_dict()
    print(f"  Encoder tensors: {len(sd)}")
    add_params(writer, sd)
    finalize_gguf(writer, path)


# ============================================================================
# Export: Speech Tokenizer Decoder (RVQ + Transformer + Vocoder)
# ============================================================================

def export_tokenizer_decoder(speech_tokenizer, output_dir: str):
    """Export Speech Tokenizer Decoder to GGUF.

    Structure: quantizer (RVQ codebooks), pre_conv, pre_transformer
    (8-layer sliding window), upsample (ConvNeXt), decoder (SnakeBeta vocoder).
    """
    print("\n[5/5] Exporting Speech Tokenizer Decoder...")
    writer, path = create_gguf_writer("qwen_tts_tokenizer_dec", output_dir)

    dec = speech_tokenizer.decoder
    dec_cfg = speech_tokenizer.config.decoder_config

    add_config(writer, {
        "codebook_size":            dec_cfg.codebook_size,       # 2048
        "hidden_size":              dec_cfg.hidden_size,          # 1024
        "latent_dim":               dec_cfg.latent_dim,           # 1024
        "num_hidden_layers":        dec_cfg.num_hidden_layers,    # 8
        "num_attention_heads":      dec_cfg.num_attention_heads,  # 16
        "num_key_value_heads":      dec_cfg.num_key_value_heads,  # 16
        "intermediate_size":        dec_cfg.intermediate_size,    # 3072
        "sliding_window":           dec_cfg.sliding_window,       # 72
        "num_quantizers":           dec_cfg.num_quantizers,       # 16
        "decoder_dim":              dec_cfg.decoder_dim,           # 1536
        "max_position_embeddings":  dec_cfg.max_position_embeddings,
        "rope_theta":               float(dec_cfg.rope_theta),
        "rms_norm_eps":             float(dec_cfg.rms_norm_eps),
        "hidden_act":               dec_cfg.hidden_act,
        "layer_scale_initial_scale": float(dec_cfg.layer_scale_initial_scale),
    })

    # Add upsample rates as arrays
    writer.add_array("upsample_rates", list(dec_cfg.upsample_rates))
    writer.add_array("upsampling_ratios", list(dec_cfg.upsampling_ratios))

    # Also add tokenizer-level config
    tok_cfg = speech_tokenizer.config
    add_config(writer, {
        "output_sample_rate":    tok_cfg.output_sample_rate,
        "decode_upsample_rate":  tok_cfg.decode_upsample_rate,
    })

    sd = dec.state_dict()
    print(f"  Decoder tensors: {len(sd)}")
    add_params(writer, sd)
    finalize_gguf(writer, path)


# ============================================================================
# Main
# ============================================================================

def load_model(model_path: str):
    """Load Qwen3-TTS model (talker + speaker encoder)."""
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path, device_map="cpu", dtype=torch.float16, trust_remote_code=True,
    )
    print("Model loaded.")
    return model


def load_speech_tokenizer(model_path: str):
    """Load speech tokenizer from the speech_tokenizer subdirectory."""
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    tokenizer_path = os.path.join(model_path, "speech_tokenizer")
    print(f"Loading speech tokenizer from {tokenizer_path}...")
    tokenizer_model = AutoModel.from_pretrained(
        tokenizer_path, device_map="cpu", dtype=torch.float32,
        trust_remote_code=True,
    )
    print("Speech tokenizer loaded.")
    return tokenizer_model


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-TTS models to GGUF format")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="HuggingFace repo or local path")
    parser.add_argument("--output_dir", type=str, default="gguf",
                        help="Output directory for GGUF files")
    parser.add_argument("--all", action="store_true",
                        help="Export all components")
    parser.add_argument("--talker", action="store_true")
    parser.add_argument("--code_predictor", action="store_true")
    parser.add_argument("--speaker_encoder", action="store_true")
    parser.add_argument("--tokenizer_enc", action="store_true")
    parser.add_argument("--tokenizer_dec", action="store_true")
    args = parser.parse_args()

    # Default to --all if no specific component selected
    if not any([args.talker, args.code_predictor, args.speaker_encoder,
                args.tokenizer_enc, args.tokenizer_dec]):
        args.all = True

    need_main_model = (args.all or args.talker or args.code_predictor
                       or args.speaker_encoder)
    need_tokenizer = (args.all or args.tokenizer_enc or args.tokenizer_dec)

    model = None
    speech_tokenizer = None

    if need_main_model:
        model = load_model(args.model_path)

    if need_tokenizer:
        speech_tokenizer = load_speech_tokenizer(args.model_path)

    if args.all or args.talker:
        export_talker(model, args.output_dir)

    if args.all or args.code_predictor:
        export_code_predictor(model, args.output_dir)

    if args.all or args.speaker_encoder:
        export_speaker_encoder(model, args.output_dir)

    if args.all or args.tokenizer_enc:
        export_tokenizer_encoder(speech_tokenizer, args.output_dir)

    if args.all or args.tokenizer_dec:
        export_tokenizer_decoder(speech_tokenizer, args.output_dir)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
