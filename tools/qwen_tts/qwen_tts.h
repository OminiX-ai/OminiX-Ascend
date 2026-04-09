#pragma once

#include "bpe_tokenizer.h"
#include "speaker_encoder.h"
#include "speech_tokenizer_encoder.h"
#include "speech_tokenizer_decoder.h"
#include "talker.h"
#include <functional>
#include <string>
#include <vector>

struct QwenTTSParams {
    std::string model_dir;           // Directory containing GGUF files
    std::string tokenizer_dir;       // Directory containing vocab.json + merges.txt
    std::string text;                // Target text to synthesize
    std::string target_lang = "English";
    std::string ref_audio;           // Reference audio path (WAV, 24kHz mono)
    std::string ref_text;            // Reference audio transcript
    std::string ref_lang = "English";
    std::string output = "output.wav";
    std::string device = "CPU";
    std::string talker_model;       // Override Talker GGUF filename (e.g. qwen_tts_talker_llama_q4km.gguf)
    std::string cp_model;           // Override CP llama GGUF (for NPU acceleration)
    std::string ref_cache;          // Pre-computed ref_codes + spk_embedding cache file
    std::string voice;              // Built-in voice id (resolved to ref_cache via voices.json)
    std::string voices_dir;         // Directory containing voices.json + *.bin caches
    std::string xvec;               // x-vector file (xvec-only inference, mutually exclusive with ICL)
    std::string xvec_extract;       // wav path: tool mode — extract spk_embedding into --xvec_out
    std::string xvec_out;           // output .xvec path for --xvec_extract
    int n_threads = 8;
    int n_gpu_layers = 0;            // Number of layers to offload to GPU/NPU (0=CPU only)
    int max_new_tokens = 2048;
    // When true, generate() lowers max_new_tokens to a heuristic per-text
    // cap (~8 frames per English BPE token, ~6 per Chinese, min 30) so a
    // failing-to-EOS talker can't run all the way to the 2048-frame max.
    // main.cpp clears this flag whenever the user passes --max_tokens.
    bool auto_max_tokens = true;
    // When true, generate() splits target_text on sentence terminators
    // (. ! ? for English, 。！？for Chinese) plus em-dash / semicolon /
    // colon clause boundaries, and runs the talker once per sentence.
    // Necessary because Qwen3-TTS was trained on relatively short
    // utterances: for ~100-word inputs the model otherwise compresses
    // speech to ~half the natural rate (162 ms/token vs 305 ms/token
    // measured), slurring words together in the second half. The encoder
    // and spk_embedding load only once per generate() call.
    bool auto_sentence_chunk = true;
    bool profiling = false;
    // Sampling parameters (matching Python defaults)
    TalkerSamplingParams sampling;

    // Streaming output: when stream_chunk_frames > 0 and stream_callback is
    // set, the talker generation runs in streaming mode. Each time
    // stream_chunk_frames new codec frames have been produced, the codec
    // decoder is invoked on those frames in isolation (no cross-chunk
    // warmup, same as the Top1 fix) and stream_callback is called with the
    // resulting PCM samples. The final partial chunk (if any) is flushed at
    // the end with is_final=true. The full audio is also accumulated into
    // the audio_out vector returned by generate(), so callers that don't
    // care about latency can ignore the callback entirely.
    using StreamCallback = std::function<void(
        const float *samples, size_t n_samples, bool is_final)>;
    int stream_chunk_frames = 0;
    StreamCallback stream_callback;
};

class QwenTTS {
public:
    QwenTTS() = default;
    ~QwenTTS() = default;

    bool load(const QwenTTSParams& params);
    bool generate(const QwenTTSParams& params, std::vector<float>& audio_out);

    // Tool mode: extract speaker embedding from a wav and save to .xvec file.
    // Only requires the speaker_encoder to be loaded (not Talker / CP / decoder).
    bool extract_xvec(const std::string &wav_path, const std::string &out_xvec_path);

private:
    QwenTTSParams params_;
    bool loaded_ = false;
    std::string cached_ref_text_;  // ref_text from cache file

    // Sub-components
    BpeTokenizer tokenizer_;
    SpeakerEncoder speaker_encoder_;
    SpeechTokenizerEncoder tokenizer_encoder_;
    SpeechTokenizerDecoder tokenizer_decoder_;
    TalkerLLM talker_;

    // Tokenize text — produces separate ref and target text token vectors
    void tokenize_tts_text(const std::string &ref_text,
                            const std::string &target_text,
                            std::vector<int> &ref_text_tokens,
                            std::vector<int> &target_text_tokens) const;
};
