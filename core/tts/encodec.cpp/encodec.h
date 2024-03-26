/**
 * @file encodec.h
 * @brief Header file for the encodec library.
 *
 * This file contains the declarations of the structs and functions used in the encodec library.
 * The library provides functionality for audio compression and decompression using a custom model.
 * The model consists of an encoder, a quantizer and a decoder, each with their own set of parameters.
 * The library also provides functions for loading and freeing the model, as well as compressing and decompressing audio data.
 *
 */
#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#define ENCODEC_FILE_MAGIC   'ggml'

static const size_t MB = 1024*1024;

enum class encodec_verbosity_level {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
};

struct encodec_hparams {
    // The number of input channels is always 1 (mono).
    int32_t in_channels          = 1;
    // The hidden dimension for the codebook.
    int32_t hidden_dim           = 128;
    // The number of filters for the first convolution.
    int32_t n_filters            = 32;
    // The filter size for upsampling and downsampling.
    int32_t ratios[4]            = {8, 5, 4, 2};
    // The kernel size for the first convolution.
    int32_t kernel_size          = 7;
    // The kernel size for the residual blocks.
    int32_t residual_kernel_size = 3;
    // Compression
    int32_t compress             = 2;
    // The number of layers in the LSTM modules.
    int32_t n_lstm_layers        = 2;
    // The stride of the first convolution.
    int32_t stride               = 1;

    // The dimension of the codebook.
    int32_t n_bins    = 1024;
    // The sample rate of the model.
    int32_t sr        = 24000;
    // The bandwidth.
    int32_t bandwidth = 24;

    // The number of codebooks.
    int32_t n_q = 32;
    // The product of the ratios.
    int32_t hop_length = 1;

    int32_t ftype;
};

// res + downsample block at some ratio
struct encodec_encoder_block {
    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;

    // downsampling layers
    struct ggml_tensor * ds_conv_w;
    struct ggml_tensor * ds_conv_b;
};

struct encodec_lstm {
    struct ggml_tensor * l0_ih_w;
    struct ggml_tensor * l0_hh_w;

    struct ggml_tensor * l0_ih_b;
    struct ggml_tensor * l0_hh_b;

    struct ggml_tensor * l1_ih_w;
    struct ggml_tensor * l1_hh_w;

    struct ggml_tensor * l1_ih_b;
    struct ggml_tensor * l1_hh_b;
};

struct encodec_encoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_encoder_block> blocks;
};

struct encodec_quant_block {
    struct ggml_tensor * embed;
};

struct encodec_quantizer {
    std::vector<encodec_quant_block> blocks;
};

struct encodec_decoder_block {
    //upsampling layers
    struct ggml_tensor * us_conv_w;
    struct ggml_tensor * us_conv_b;

    // conv1
    struct ggml_tensor * conv_1_w;
    struct ggml_tensor * conv_1_b;

    // conv2
    struct ggml_tensor * conv_2_w;
    struct ggml_tensor * conv_2_b;

    // shortcut
    struct ggml_tensor * conv_sc_w;
    struct ggml_tensor * conv_sc_b;
};

struct encodec_decoder {
    struct ggml_tensor * init_conv_w;
    struct ggml_tensor * init_conv_b;

    encodec_lstm lstm;

    struct ggml_tensor * final_conv_w;
    struct ggml_tensor * final_conv_b;

    std::vector<encodec_decoder_block> blocks;
};

struct encodec_model {
    encodec_hparams hparams;

    encodec_encoder   encoder;
    encodec_quantizer quantizer;
    encodec_decoder   decoder;

    // context
    struct ggml_context * ctx;
    int n_loaded;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;

    std::map<std::string, struct ggml_tensor *> tensors;
};

struct encodec_context {
    encodec_model model;

    // buffer for model evaluation
    ggml_backend_buffer_t buf_compute;

    // custom allocrator
    struct ggml_allocr * allocr = NULL;

    // intermediate steps
    struct ggml_tensor * encoded = NULL;  // Encoded audio
    struct ggml_tensor * codes   = NULL;  // Quantized representation of audio in codebook
    struct ggml_tensor * decoded = NULL;  // Reconstructed audio from codes

    std::vector<int32_t> out_codes;
    std::vector<float>   out_audio;

    // statistics
    int64_t t_load_us    = 0;
    int64_t t_compute_ms = 0;
};

/**
 * Loads an encodec model from the specified file path.
 *
 * @param model_path The file path to the encodec model.
 * @param n_gpu_layers The number of GPU layers to use.
 * @return A pointer to the encodec context struct.
 */
struct encodec_context * encodec_load_model(
                 const std::string & model_path,
                               int   n_gpu_layers,
           encodec_verbosity_level   verbosity);
/**
 * Sets the target bandwidth for the given encodec context.
 *
 * @param ectx The encodec context to set the target bandwidth for.
 * @param bandwidth The target bandwidth to set, in bits per second.
 */
void encodec_set_target_bandwidth(
            struct encodec_context * ectx,
                               int   bandwidth);

/**
 * Reconstructs audio from raw audio data using the specified encodec context.
 *
 * @param ectx The encodec context to use for reconstruction.
 * @param raw_audio The raw audio data to reconstruct.
 * @param n_threads The number of threads to use for reconstruction.
 * @return True if the reconstruction was successful, false otherwise.
 */
bool encodec_reconstruct_audio(
            struct encodec_context * ectx,
                std::vector<float> & raw_audio,
                               int   n_threads);

/**
 * Compresses audio data using the specified encodec context.
 *
 * @param ectx The encodec context to use for compression.
 * @param raw_audio The raw audio data to compress.
 * @param n_threads The number of threads to use for compression.
 * @return True if the compression was successful, false otherwise.
 */
bool encodec_compress_audio(
            struct encodec_context * ectx,
                std::vector<float> & raw_audio,
                               int   n_threads);

/**
 * Decompresses audio data using the specified encodec context.
 *
 * @param ectx The encodec context to use for decompression.
 * @param codes The compressed audio data to decompress.
 * @param n_threads The number of threads to use for decompression.
 * @return True if the audio data was successfully decompressed, false otherwise.
 */
bool encodec_decompress_audio(
            struct encodec_context * ectx,
              std::vector<int32_t> & codes,
                               int   n_threads);

/**
 * @brief Frees the memory allocated for an encodec context.
 *
 * @param ectx The encodec context to free.
 */
void encodec_free(
            struct encodec_context * ectx);

void encodec_set_sample_rate(struct encodec_context * ectx, int sample_rate);