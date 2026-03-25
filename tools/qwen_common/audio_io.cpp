#define MINIAUDIO_IMPLEMENTATION
#include "../../vendor/miniaudio/miniaudio.h"

#include "audio_io.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace audio_io {

bool load_wav(const std::string &path, AudioData &audio) {
  ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);
  ma_decoder decoder;

  ma_result result = ma_decoder_init_file(path.c_str(), &config, &decoder);
  if (result != MA_SUCCESS) {
    fprintf(stderr, "Failed to open audio file: %s\n", path.c_str());
    return false;
  }

  audio.sample_rate = decoder.outputSampleRate;
  printf("[audio_io] loaded %s: sr=%d, channels=%d\n", path.c_str(), audio.sample_rate, audio.channels);
  audio.channels = decoder.outputChannels;

  // Get total frame count
  ma_uint64 total_frames;
  result = ma_decoder_get_length_in_pcm_frames(&decoder, &total_frames);
  if (result != MA_SUCCESS) {
    // If we can't get length, read in chunks
    total_frames = 0;
  }

  // Read all frames
  std::vector<float> buffer;
  if (total_frames > 0) {
    buffer.resize(total_frames * audio.channels);
    ma_uint64 frames_read;
    result =
        ma_decoder_read_pcm_frames(&decoder, buffer.data(), total_frames, &frames_read);
    buffer.resize(frames_read * audio.channels);
  } else {
    // Read in chunks for streaming formats
    const size_t chunk_size = 4096;
    std::vector<float> chunk(chunk_size * audio.channels);
    ma_uint64 frames_read;
    while (true) {
      result =
          ma_decoder_read_pcm_frames(&decoder, chunk.data(), chunk_size, &frames_read);
      if (frames_read == 0) {
        break;
      }
      buffer.insert(buffer.end(), chunk.begin(),
                    chunk.begin() + frames_read * audio.channels);
    }
  }

  ma_decoder_uninit(&decoder);

  // Convert to mono by averaging channels
  if (audio.channels > 1) {
    size_t num_frames = buffer.size() / audio.channels;
    audio.samples.resize(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
      float sum = 0.0f;
      for (int c = 0; c < audio.channels; ++c) {
        sum += buffer[i * audio.channels + c];
      }
      audio.samples[i] = sum / audio.channels;
    }
    audio.channels = 1;
  } else {
    audio.samples = std::move(buffer);
  }

  return true;
}

bool resample(const AudioData &input, int target_sr,
              std::vector<float> &output) {
  printf("[audio_io] resample: input_sr=%d, target_sr=%d, input_samples=%zu\n",
         input.sample_rate, target_sr, input.samples.size());
  if (input.sample_rate == target_sr) {
    output = input.samples;
    return true;
  }

  ma_resampler_config config =
      ma_resampler_config_init(ma_format_f32, 1, input.sample_rate, target_sr,
                               ma_resample_algorithm_linear);
  // Enable low-pass filtering with maximum order for better quality
  // This helps reduce aliasing artifacts during resampling
  config.linear.lpfOrder = MA_MAX_FILTER_ORDER;  // 8

  ma_resampler resampler;
  ma_result result = ma_resampler_init(&config, nullptr, &resampler);
  if (result != MA_SUCCESS) {
    fprintf(stderr, "Failed to initialize resampler\n");
    return false;
  }

  // Calculate expected output size
  ma_uint64 input_frames = input.samples.size();
  ma_uint64 expected_output_frames;
  result = ma_resampler_get_expected_output_frame_count(&resampler, input_frames,
                                                        &expected_output_frames);
  if (result != MA_SUCCESS) {
    // Estimate output size
    expected_output_frames =
        (input_frames * target_sr + input.sample_rate - 1) / input.sample_rate;
  }

  output.resize(expected_output_frames + 1024); // Add some padding

  ma_uint64 frames_in = input_frames;
  ma_uint64 frames_out = output.size();

  result = ma_resampler_process_pcm_frames(&resampler, input.samples.data(),
                                           &frames_in, output.data(), &frames_out);

  ma_resampler_uninit(&resampler, nullptr);

  if (result != MA_SUCCESS) {
    fprintf(stderr, "Resampling failed\n");
    return false;
  }

  output.resize(frames_out);
  return true;
}

bool load_audio(const std::string &path, int target_sr,
                std::vector<float> &output) {
  AudioData audio;
  if (!load_wav(path, audio)) {
    return false;
  }

  if (!resample(audio, target_sr, output)) {
    return false;
  }

  return true;
}

void normalize_volume(std::vector<float> &audio) {
  if (audio.empty()) {
    return;
  }

  float max_val = 0.0f;
  for (const float &sample : audio) {
    float abs_val = std::abs(sample);
    if (abs_val > max_val) {
      max_val = abs_val;
    }
  }

  // If max > 1, divide by min(2, max) to prevent clipping
  if (max_val > 1.0f) {
    float divisor = std::min(2.0f, max_val);
    for (float &sample : audio) {
      sample /= divisor;
    }
  }
}

void pad_zeros(std::vector<float> &audio, int num_samples) {
  if (num_samples <= 0) {
    return;
  }
  audio.resize(audio.size() + num_samples, 0.0f);
}

bool validate_duration(const std::vector<float> &audio, int sample_rate,
                       float min_sec, float max_sec) {
  float duration = get_duration(audio, sample_rate);
  return duration >= min_sec && duration <= max_sec;
}

bool save_wav(const std::string &path, const std::vector<int16_t> &samples,
              int sample_rate, int channels) {
  ma_encoder_config config = ma_encoder_config_init(
      ma_encoding_format_wav, ma_format_s16, channels, sample_rate);
  ma_encoder encoder;

  ma_result result = ma_encoder_init_file(path.c_str(), &config, &encoder);
  if (result != MA_SUCCESS) {
    fprintf(stderr, "Failed to open file for writing: %s\n", path.c_str());
    return false;
  }

  ma_uint64 frames_written;
  result = ma_encoder_write_pcm_frames(&encoder, samples.data(),
                                       samples.size() / channels, &frames_written);

  ma_encoder_uninit(&encoder);

  if (result != MA_SUCCESS) {
    fprintf(stderr, "Failed to write audio data\n");
    return false;
  }

  return true;
}

bool save_wav(const std::string &path, const std::vector<float> &samples,
              int sample_rate, int channels) {
  // Convert float [-1, 1] to int16
  std::vector<int16_t> int_samples(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    float clamped = std::max(-1.0f, std::min(1.0f, samples[i]));
    int_samples[i] = static_cast<int16_t>(clamped * 32767.0f);
  }
  return save_wav(path, int_samples, sample_rate, channels);
}

} // namespace audio_io
