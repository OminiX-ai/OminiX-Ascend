#pragma once

#include "bpe_tokenizer.h"
#include "speaker_encoder.h"
#include "speech_tokenizer_encoder.h"
#include "speech_tokenizer_decoder.h"
#include "talker.h"
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
    int n_threads = 8;
    int n_gpu_layers = 0;            // Number of layers to offload to GPU/NPU (0=CPU only)
    int max_new_tokens = 2048;
    bool profiling = false;
    // Sampling parameters (matching Python defaults)
    TalkerSamplingParams sampling;
};

class QwenTTS {
public:
    QwenTTS() = default;
    ~QwenTTS() = default;

    bool load(const QwenTTSParams& params);
    bool generate(const QwenTTSParams& params, std::vector<float>& audio_out);

private:
    QwenTTSParams params_;
    bool loaded_ = false;

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
