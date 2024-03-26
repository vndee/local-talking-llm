#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "encodec.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


typedef enum {
    // Run the end-to-end encoder-decoder pipeline
    full   = 0,
    // Encode an audio (encoder + quantizer encode)
    encode = 1,
    // Decode an audio from a compressed representation (quantizer decode + decoder)
    decode = 2,
} encodec_run_mode;

static void print_encodec_tensor(struct ggml_tensor * a) {
    float sum = 0;
    float maxv = -INFINITY;
    float minv = INFINITY;
    if (a) {
        for (int i = 0; i < a->ne[3]; i++) {
            for (int j = 0; j < a->ne[2]; j++) {
                for (int k = 0; k < a->ne[1]; k++) {
                    for (int l = 0; l < a->ne[0]; l++) {
                        if (a->type == GGML_TYPE_F32) {
                            float * aval = (float *) (
                                (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                            sum += *aval;
                            maxv = MAX(*aval, maxv);
                            minv = MIN(*aval, minv);
                            // printf("%.4f ", *aval);
                        } else if (a->type == GGML_TYPE_F16) {
                            ggml_fp16_t * tmp = (ggml_fp16_t *) (
                                (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                            float aval = ggml_fp16_to_fp32(*tmp);
                            sum += aval;
                            maxv = MAX(aval, maxv);
                            minv = MIN(aval, minv);
                            // printf("%.4f ", aval);
                        } else if (a->type == GGML_TYPE_I32) {
                            int32_t * aval = (int32_t *) (
                                (char *) a->data + i*a->nb[3] + j*a->nb[2] + k*a->nb[1] + l*a->nb[0]);
                            sum += (float) *aval;
                            maxv = MAX((float) *aval, maxv);
                            minv = MIN((float) *aval, minv);
                            // printf("%d ", *aval);
                        } else {
                            throw std::runtime_error("Wrong tensor type.");
                        }
                    }
                    // printf("\n");
                }
                // printf("\n\n");
            }
        }
        printf("sum=%.2f; max=%.2f; min=%.2f\n", sum, maxv, minv);
        printf("shape=[%lld, %lld, %lld, %lld]\n", a->ne[0], a->ne[1], a->ne[2], a->ne[3]);
    }
}

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

static void encodec_sigmoid_impl(
                    struct ggml_tensor * dst,
              const struct ggml_tensor * src,
                                   int   ith,
                                   int   nth,
                                  void * userdata) {
    GGML_ASSERT(userdata == NULL);
    GGML_ASSERT(ggml_are_same_shape(dst, src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src));

    const float * src_data = ggml_get_data_f32(src);
    float * dst_data = ggml_get_data_f32(dst);

    const int ne = (int)ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = 1.0f / (1.0f + expf(-src_data[i]));
    }
}

static struct ggml_tensor * encodec_sigmoid(
      struct ggml_context * ctx,
       struct ggml_tensor * x) {
    return ggml_map_custom1(ctx, x, encodec_sigmoid_impl, GGML_N_TASKS_MAX, NULL);
}

static int get_extra_padding_for_conv_1d(
                struct ggml_tensor * inp,
                             float   kernel_size,
                             float   stride,
                             float   padding_total) {
    float length = inp->ne[0];
    float n_frames = (length - kernel_size + padding_total) / stride + 1.0f;
    int ideal_length = (ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    return ideal_length - length;
}

static struct ggml_tensor * pad_1d(
      struct ggml_context * ctx0,
       struct ggml_tensor * inp,
                      int   padding_left,
                      int   padding_right) {
    int length = inp->ne[0];
    int dim = inp->ne[1];

    const int max_pad = std::max(padding_left, padding_right);
    int extra_pad = 0;

    if (length <= max_pad) {
        extra_pad = max_pad - length + 1;

        // constant padding
        struct ggml_tensor * out = ggml_new_tensor_2d(ctx0, inp->type, length+extra_pad, dim);
        ggml_set_zero(out);
        out = ggml_set_2d(ctx0, out, inp, out->nb[1], 0);
    }

    struct ggml_tensor * padded = ggml_pad_reflec_1d(ctx0, inp, padding_left, padding_right);

    const int end = padded->ne[0] - extra_pad;
    struct ggml_tensor *dest = ggml_view_2d(ctx0, padded, end, dim, padded->nb[1], 0);

    return dest;
}

static int32_t get_num_codebooks(float bandwidth, int hop_length, float sample_rate) {
    // The number of codebooks is determined by the bandwidth selected.
    // Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8),
    // 12 kbps (n_q = 16) and 24kbps (n_q = 32).
    return (int32_t) ceilf(1000 * bandwidth / (ceilf(sample_rate / hop_length) * 10));
}

static int32_t get_bandwidth_per_quantizer(int bins, float frame_rate) {
    return log2f((float) bins) * frame_rate;
}

static int32_t get_num_quantizers_for_bandwidth(int bins, float frame_rate, float bandwidth) {
    float bw_per_q = get_bandwidth_per_quantizer(bins, frame_rate);
    int32_t n_q = MAX(1, floorf(bandwidth * 1000 / bw_per_q));
    return n_q;
}

static struct ggml_tensor * unpad_1d(
      struct ggml_context * ctx0,
       struct ggml_tensor * inp,
                      int   padding_left,
                      int   padding_right) {
    int length = inp->ne[0];
    int dim    = inp->ne[1];

    assert(padding_left  >= 0);
    assert(padding_right >= 0);
    assert(padding_left + padding_right <= length);

    int end = length - padding_right;

    int offset = padding_left * inp->nb[1];
    struct ggml_tensor * dst = ggml_view_2d(ctx0, inp, end, dim, inp->nb[1], offset);

    return dst;
}

static struct ggml_tensor * strided_conv_1d(
             ggml_context * ctx0,
              ggml_tensor * inp,
              ggml_tensor * conv_w,
              ggml_tensor * conv_b,
                      int   stride) {
    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;
    int extra_padding = get_extra_padding_for_conv_1d(inp, kernel_size, stride, padding_total);

    struct ggml_tensor * padded_inp = pad_1d(ctx0, inp, padding_total, extra_padding);
    struct ggml_tensor * dst = ggml_conv_1d(ctx0, conv_w, padded_inp, stride, 0, 1);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    return dst;
}

static struct ggml_tensor * strided_conv_transpose_1d(
      struct ggml_context * ctx0,
       struct ggml_tensor * inp,
       struct ggml_tensor * conv_w,
       struct ggml_tensor * conv_b,
                      int   stride) {

    struct ggml_tensor * dst = ggml_conv_transpose_1d(
        ctx0, conv_w, inp, stride, 0 /* p0 */, 1 /* d0 */);

    // add bias
    dst = ggml_transpose(ctx0, dst);
    dst = ggml_add(ctx0, ggml_repeat(ctx0, conv_b, dst), dst);
    dst = ggml_cont(ctx0, ggml_transpose(ctx0, dst));

    int kernel_size   = conv_w->ne[0];
    int padding_total = kernel_size - stride;

    int padding_right = ceilf(padding_total);
    int padding_left = padding_total - padding_right;

    struct ggml_tensor * unpadded = unpad_1d(ctx0, dst, padding_left, padding_right);
    unpadded = ggml_cont(ctx0, unpadded);

    return unpadded;
}

static struct ggml_tensor * forward_pass_lstm_unilayer(
      struct ggml_context * ctx0,
       struct ggml_allocr * allocr,
       struct ggml_tensor * inp,
       struct ggml_tensor * weight_ih,
       struct ggml_tensor * weight_hh,
       struct ggml_tensor * bias_ih,
       struct ggml_tensor * bias_hh) {

    const int input_dim  = inp->ne[1];
    const int hidden_dim = weight_ih->ne[1]/4;
    const int seq_length = inp->ne[0];

    struct ggml_tensor * hs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_allocr_alloc(allocr, hs);

    struct ggml_tensor * c_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    ggml_allocr_alloc(allocr, c_t);

    struct ggml_tensor * h_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hidden_dim);
    ggml_allocr_alloc(allocr, h_t);

    if (!ggml_allocr_is_measure(allocr)) {
        h_t = ggml_set_zero(h_t);
        c_t = ggml_set_zero(c_t);
    }

    struct ggml_tensor * current = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    for (int t = 0; t < seq_length; t++) {
        struct ggml_tensor * x_t = ggml_view_1d(ctx0, current, input_dim, t*current->nb[1]);

        struct ggml_tensor * inp_gates = ggml_mul_mat(ctx0, weight_ih, x_t);
        inp_gates = ggml_add(ctx0, inp_gates, bias_ih);

        struct ggml_tensor * hid_gates = ggml_mul_mat(ctx0, weight_hh, h_t);
        hid_gates = ggml_add(ctx0, hid_gates, bias_hh);

        struct ggml_tensor * out_gates = ggml_add(ctx0, inp_gates, hid_gates);

        struct ggml_tensor * i_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 0*sizeof(float)*hidden_dim));
        struct ggml_tensor * f_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 1*sizeof(float)*hidden_dim));
        struct ggml_tensor * g_t = ggml_tanh      (ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 2*sizeof(float)*hidden_dim));
        struct ggml_tensor * o_t = encodec_sigmoid(ctx0, ggml_view_1d(ctx0, out_gates, hidden_dim, 3*sizeof(float)*hidden_dim));

        c_t = ggml_add(ctx0, ggml_mul(ctx0, f_t, c_t), ggml_mul(ctx0, i_t, g_t));

        h_t = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_t));

        hs = ggml_set_1d(ctx0, hs, h_t, t*hs->nb[1]);
    }

    hs = ggml_cont(ctx0, ggml_transpose(ctx0, hs));

    return hs;
}

bool encodec_load_model_weights(
            const std::string & fname,
                encodec_model & model,
                          int   n_gpu_layers,
      encodec_verbosity_level   verbosity) {
    if (verbosity == encodec_verbosity_level::MEDIUM || verbosity == encodec_verbosity_level::HIGH) {
        fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());
    }

    auto infile = std::ifstream(fname, std::ios::binary);
    if (!infile) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic (i.e. ggml signature in hex format)
    {
        uint32_t magic;
        read_safe(infile, magic);
        if (magic != ENCODEC_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        read_safe(infile, hparams.in_channels);
        read_safe(infile, hparams.hidden_dim);
        read_safe(infile, hparams.n_filters);
        read_safe(infile, hparams.kernel_size);
        read_safe(infile, hparams.residual_kernel_size);
        // read_safe(infile, hparams.ratios);
        read_safe(infile, hparams.n_bins);
        read_safe(infile, hparams.bandwidth);
        read_safe(infile, hparams.sr);
        read_safe(infile, hparams.ftype);

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        if (verbosity == encodec_verbosity_level::MEDIUM || verbosity == encodec_verbosity_level::HIGH) {
            printf("%s: in_channels = %d\n", __func__, hparams.in_channels);
            printf("%s: hidden_dim  = %d\n", __func__, hparams.hidden_dim);
            printf("%s: n_filters   = %d\n", __func__, hparams.n_filters);
            printf("%s: kernel_size = %d\n", __func__, hparams.kernel_size);
            printf("%s: res_kernel  = %d\n", __func__, hparams.residual_kernel_size);
            // printf("%s: ratios      = %d\n", __func__, hparams.ratios);
            printf("%s: n_bins      = %d\n", __func__, hparams.n_bins);
            printf("%s: bandwidth   = %d\n", __func__, hparams.bandwidth);
            printf("%s: sample_rate = %d\n", __func__, hparams.sr);
            printf("%s: ftype       = %d\n", __func__, hparams.ftype);
            printf("%s: qntvr       = %d\n", __func__, qntvr);
        }

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return 1;
    }

    auto & ctx = model.ctx;

    size_t buffer_size = 0;
    size_t n_tensors = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int in_channels   = hparams.in_channels;
        const int hidden_dim    = hparams.hidden_dim;
        const int n_filters     = hparams.n_filters;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_bins        = hparams.n_bins;
        const int *ratios       = hparams.ratios;
        const int n_lstm_layers = hparams.n_lstm_layers;

        // encoder
        {
            int mult = 1;  // scaling factor for hidden size

            // initial conv1d layer
            buffer_size += in_channels * n_filters * kernel_size * ggml_type_size(wtype);  // weight
            buffer_size += n_filters * ggml_type_size(GGML_TYPE_F32);                      // bias

            // resnet blocks
            for (int i = 0; i < 4; i++) {
                // conv1
                buffer_size += res_kernel_sz * (mult*n_filters) * (mult*n_filters/2) * ggml_type_size(wtype); // weight
                buffer_size += (mult*n_filters/2) * ggml_type_size(GGML_TYPE_F32);                            // bias

                // conv2
                buffer_size += (mult*n_filters/2) * (mult*n_filters) * ggml_type_size(wtype); // weight
                buffer_size += (mult*n_filters) * ggml_type_size(GGML_TYPE_F32);              // bias

                // shortcut
                buffer_size += (mult*n_filters) * (mult*n_filters) * ggml_type_size(wtype); // weight
                buffer_size += (mult*n_filters) * ggml_type_size(GGML_TYPE_F32);            // bias

                // downsampling layers
                buffer_size += (2*ratios[3-i]) * (mult*n_filters) * (mult*n_filters*2) * ggml_type_size(wtype); // weight
                buffer_size += (2*mult*n_filters) * ggml_type_size(GGML_TYPE_F32);                              // bias

                mult *= 2;
            }

            // lstm
            buffer_size += 2 * n_lstm_layers * (mult*n_filters) * (4*mult*n_filters) * ggml_type_size(wtype); // weight_ih and weight_hh
            buffer_size += 2 * n_lstm_layers * (4*mult*n_filters) * ggml_type_size(GGML_TYPE_F32);            // bias_ih and bias_hh

            // final conv
            buffer_size += kernel_size * (mult*n_filters) * hidden_dim * ggml_type_size(wtype); // weight
            buffer_size += hidden_dim * ggml_type_size(GGML_TYPE_F32);                          // bias
        }

        // decoder mirrors the encoder (same number of parameters), just double context size
        buffer_size *= 2;

        // quantizer
        int n_q = 32; // 32 is an upper bound on the number of codebooks.
        buffer_size += n_q * hidden_dim * n_bins * ggml_type_size(GGML_TYPE_F32);  // embed

        buffer_size += 10ull*MB;  // object overhead

        n_tensors = ((4 * 2) * 4 + 2 + 4 * n_lstm_layers + 2) * 2;  // encoder and decoder
        n_tensors += n_q * 1;  // quantizer

        if (verbosity == encodec_verbosity_level::HIGH) {
            printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
            printf("%s: backend buffer size = %6.2f MB\n", __func__, buffer_size/(1024.0*1024.0));
        }
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /* .mem_size   = */   ggml_tensor_overhead() * n_tensors,
            /* .mem_buffer = */   NULL,
            /* .no_alloc   = */   true,
        };

        model.ctx = ggml_init(params);
        if(!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

#ifdef GGML_USE_CUBLAS
    if (n_gpu_layers > 0) {
        if (verbosity == encodec_verbosity_level::HIGH) {
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
        }
        model.backend = ggml_backend_cuda_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        if (verbosity == encodec_verbosity_level::HIGH) {
            fprintf(stderr, "%s: using Metal backend\n", __func__);
        }
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        if (verbosity == encodec_verbosity_level::HIGH) {
            fprintf(stderr, "%s: using CPU backend\n", __func__);
        }
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    // allocate weights buffer
    model.buffer_w = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int in_channels   = hparams.in_channels;
        const int hidden_dim    = hparams.hidden_dim;
        const int n_filters     = hparams.n_filters;
        const int kernel_size   = hparams.kernel_size;
        const int res_kernel_sz = hparams.residual_kernel_size;
        const int n_q           = hparams.n_q;
        const int *ratios       = hparams.ratios;
        const int n_bins        = hparams.n_bins;

        // encoder
        {
            model.encoder.blocks.resize(4);

            int mult = 1;  // scaling factor for hidden size

            model.encoder.init_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, in_channels, mult*n_filters);
            model.encoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

            model.tensors["encoder.model.0.conv.conv.weight"] = model.encoder.init_conv_w;
            model.tensors["encoder.model.0.conv.conv.bias"]   = model.encoder.init_conv_b;

            for (int i = 0; i < 4; i++) {
                // conv1
                model.encoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, wtype, res_kernel_sz, mult*n_filters, mult*n_filters/2);
                model.encoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.1.conv.conv.weight"] = model.encoder.blocks[i].conv_1_w;
                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.1.conv.conv.bias"]   = model.encoder.blocks[i].conv_1_b;

                // conv2
                model.encoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, wtype, 1, mult*n_filters/2, mult*n_filters);
                model.encoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.3.conv.conv.weight"] = model.encoder.blocks[i].conv_2_w;
                model.tensors["encoder.model." + std::to_string(3*i+1) + ".block.3.conv.conv.bias"]   = model.encoder.blocks[i].conv_2_b;

                // shortcut conv
                model.encoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, wtype, 1, mult*n_filters, mult*n_filters);
                model.encoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

                model.tensors["encoder.model." + std::to_string(3*i+1) + ".shortcut.conv.conv.weight"] = model.encoder.blocks[i].conv_sc_w;
                model.tensors["encoder.model." + std::to_string(3*i+1) + ".shortcut.conv.conv.bias"]   = model.encoder.blocks[i].conv_sc_b;

                // downsampling
                model.encoder.blocks[i].ds_conv_w = ggml_new_tensor_3d(ctx, wtype, 2*ratios[3-i], mult*n_filters, mult*n_filters*2);
                model.encoder.blocks[i].ds_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters*2);

                model.tensors["encoder.model." + std::to_string(3*(i+1)) + ".conv.conv.weight"] = model.encoder.blocks[i].ds_conv_w;
                model.tensors["encoder.model." + std::to_string(3*(i+1)) + ".conv.conv.bias"]   = model.encoder.blocks[i].ds_conv_b;

                mult *= 2;
            }

            // LSTM
            model.encoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);
            model.encoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.weight_ih_l0"] = model.encoder.lstm.l0_ih_w;
            model.tensors["encoder.model.13.lstm.weight_ih_l1"] = model.encoder.lstm.l1_ih_w;

            model.encoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);
            model.encoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.weight_hh_l0"] = model.encoder.lstm.l0_hh_w;
            model.tensors["encoder.model.13.lstm.weight_hh_l1"] = model.encoder.lstm.l1_hh_w;

            model.encoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.encoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.bias_ih_l0"] = model.encoder.lstm.l0_ih_b;
            model.tensors["encoder.model.13.lstm.bias_ih_l1"] = model.encoder.lstm.l1_ih_b;

            model.encoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.encoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["encoder.model.13.lstm.bias_hh_l0"] = model.encoder.lstm.l0_hh_b;
            model.tensors["encoder.model.13.lstm.bias_hh_l1"] = model.encoder.lstm.l1_hh_b;

            // final conv
            model.encoder.final_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, mult*n_filters, hidden_dim);
            model.encoder.final_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);

            model.tensors["encoder.model.15.conv.conv.weight"] = model.encoder.final_conv_w;
            model.tensors["encoder.model.15.conv.conv.bias"]   = model.encoder.final_conv_b;
        }

        // decoder
        {
            model.decoder.blocks.resize(4);

            int mult = 16;  // 2**len(ratios)

            model.decoder.init_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, hidden_dim, mult*n_filters);
            model.decoder.init_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters);

            model.tensors["decoder.model.0.conv.conv.weight"] = model.decoder.init_conv_w;
            model.tensors["decoder.model.0.conv.conv.bias"]   = model.decoder.init_conv_b;

            // LSTM
            model.decoder.lstm.l0_ih_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);
            model.decoder.lstm.l1_ih_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.weight_ih_l0"] = model.decoder.lstm.l0_ih_w;
            model.tensors["decoder.model.1.lstm.weight_ih_l1"] = model.decoder.lstm.l1_ih_w;

            model.decoder.lstm.l0_hh_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);
            model.decoder.lstm.l1_hh_w = ggml_new_tensor_2d(ctx, wtype, mult*n_filters, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.weight_hh_l0"] = model.decoder.lstm.l0_hh_w;
            model.tensors["decoder.model.1.lstm.weight_hh_l1"] = model.decoder.lstm.l1_hh_w;

            model.decoder.lstm.l0_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.decoder.lstm.l1_ih_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.bias_ih_l0"] = model.decoder.lstm.l0_ih_b;
            model.tensors["decoder.model.1.lstm.bias_ih_l1"] = model.decoder.lstm.l1_ih_b;

            model.decoder.lstm.l0_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);
            model.decoder.lstm.l1_hh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*mult*n_filters);

            model.tensors["decoder.model.1.lstm.bias_hh_l0"] = model.decoder.lstm.l0_hh_b;
            model.tensors["decoder.model.1.lstm.bias_hh_l1"] = model.decoder.lstm.l1_hh_b;

            for (int i = 0; i < 4; i++) {
                // upsampling
                model.decoder.blocks[i].us_conv_w = ggml_new_tensor_3d(ctx, wtype, ratios[i]*2, mult*n_filters/2, mult*n_filters);
                model.decoder.blocks[i].us_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)) + ".convtr.convtr.weight"] = model.decoder.blocks[i].us_conv_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)) + ".convtr.convtr.bias"]   = model.decoder.blocks[i].us_conv_b;

                // conv1
                model.decoder.blocks[i].conv_1_w = ggml_new_tensor_3d(ctx, wtype, res_kernel_sz, mult*n_filters/2, mult*n_filters/4);
                model.decoder.blocks[i].conv_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/4);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.1.conv.conv.weight"] = model.decoder.blocks[i].conv_1_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.1.conv.conv.bias"]     = model.decoder.blocks[i].conv_1_b;

                // conv2
                model.decoder.blocks[i].conv_2_w = ggml_new_tensor_3d(ctx, wtype, 1, mult*n_filters/4, mult*n_filters/2);
                model.decoder.blocks[i].conv_2_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.3.conv.conv.weight"] = model.decoder.blocks[i].conv_2_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".block.3.conv.conv.bias"]   = model.decoder.blocks[i].conv_2_b;

                // shortcut
                model.decoder.blocks[i].conv_sc_w = ggml_new_tensor_3d(ctx, wtype, 1, mult*n_filters/2, mult*n_filters/2);
                model.decoder.blocks[i].conv_sc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mult*n_filters/2);

                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".shortcut.conv.conv.weight"] = model.decoder.blocks[i].conv_sc_w;
                model.tensors["decoder.model." + std::to_string(3*(i+1)+1) + ".shortcut.conv.conv.bias"]   = model.decoder.blocks[i].conv_sc_b;

                mult /= 2;
            }

            model.decoder.final_conv_w = ggml_new_tensor_3d(ctx, wtype, kernel_size, n_filters, in_channels);
            model.decoder.final_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);

            model.tensors["decoder.model.15.conv.conv.weight"] = model.decoder.final_conv_w;
            model.tensors["decoder.model.15.conv.conv.bias"]   = model.decoder.final_conv_b;
        }

        // quantizer
        {
            model.quantizer.blocks.resize(n_q);

            for (int i = 0; i < n_q; i++) {
                model.quantizer.blocks[i].embed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_bins);

                model.tensors["quantizer.vq.layers." + std::to_string(i) + "._codebook.embed"] = model.quantizer.blocks[i].embed;
            }
        }

    }

    // load weights
    {
        ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer_w);

        size_t total_size = 0;
        model.n_loaded    = 0;

        std::vector<char> read_buf;

        while(true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(infile, n_dims);
            read_safe(infile, length);
            read_safe(infile, ftype);

            if (infile.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = {1, 1, 1};
            for (int i = 0; i < n_dims; i++) {
                read_safe(infile, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> buf(length);
            infile.read(&buf[0], buf.size());
            name.assign(&buf[0], buf.size());

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld, %lld], expected [%d, %d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], tensor->ne[2], ne[0], ne[1], ne[2]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ftype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            ggml_allocr_alloc(alloc, tensor);

            if (ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
            ) {
                // for the CPU and Metal backends, we can read directly into the device memory
                infile.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                infile.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            if (verbosity == encodec_verbosity_level::HIGH) {
                printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            }

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        ggml_allocr_free(alloc);
        if (verbosity == encodec_verbosity_level::MEDIUM || verbosity == encodec_verbosity_level::HIGH) {
            printf("%s: model size = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
        }
    }

    infile.close();

    return true;
}

struct ggml_tensor * encodec_forward_encoder(
               struct encodec_context * ectx,
                  struct ggml_context * ctx0,
                   struct ggml_tensor * inp) {
    if (!inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto & model   = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr  = ectx->allocr;

    const int * ratios      = hparams.ratios;
    const int kernel_size   = hparams.kernel_size;
    const int res_kernel_sz = hparams.residual_kernel_size;
    const int stride        = hparams.stride;

    struct ggml_tensor * inpL = strided_conv_1d(
        ctx0, inp, model.encoder.init_conv_w, model.encoder.init_conv_b, stride);

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_encoder_block block = model.encoder.blocks[layer_ix];

        struct ggml_tensor * current = inpL;

        // shortcut
        struct ggml_tensor * shortcut = strided_conv_1d(
            ctx0, inpL, block.conv_sc_w, block.conv_sc_b, stride);

        // conv1
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_1_w, block.conv_1_b, stride);

        // conv2
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_2_w, block.conv_2_b, stride);

        // residual connection
        inpL = ggml_add(ctx0, current, shortcut);

        // downsampling layers
        inpL = ggml_elu(ctx0, inpL);

        inpL = strided_conv_1d(
            ctx0, inpL, block.ds_conv_w, block.ds_conv_b, ratios[3-layer_ix]);
    }

    // lstm
    {
        struct ggml_tensor * cur = inpL;

        const encodec_lstm lstm = model.encoder.lstm;

        // first lstm layer
        struct ggml_tensor * hs1 = forward_pass_lstm_unilayer(
            ctx0, allocr, cur, lstm.l0_ih_w, lstm.l0_hh_w,
            lstm.l0_ih_b, lstm.l0_hh_b);

        // second lstm layer
        struct ggml_tensor * out = forward_pass_lstm_unilayer(
            ctx0, allocr, hs1, lstm.l1_ih_w, lstm.l1_hh_w,
            lstm.l1_ih_b, lstm.l1_hh_b);

        inpL = ggml_add(ctx0, inpL, out);
    }

    // final conv
    inpL = ggml_elu(ctx0, inpL);

    struct ggml_tensor * encoded_inp = strided_conv_1d(
        ctx0, inpL, model.encoder.final_conv_w, model.encoder.final_conv_b, stride);

    return encoded_inp;
}

struct ggml_tensor * encodec_forward_quantizer_encode(
               struct encodec_context * ectx,
                  struct ggml_context * ctx0,
                   struct ggml_tensor * encoded_inp) {
    if (!encoded_inp) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto & model   = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr  = ectx->allocr;

    const int n_bins     = hparams.n_bins;
    const int sr         = hparams.sr;
    const int bandwidth  = hparams.bandwidth;
    const int hop_length = hparams.hop_length;

    const int frame_rate = (int) ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    const int seq_length = encoded_inp->ne[0];

    struct ggml_tensor * codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, seq_length, n_q);
    ggml_allocr_alloc(allocr, codes);

    struct ggml_tensor * dist_scale = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_allocr_alloc(allocr, dist_scale);

    if (!ggml_allocr_is_measure(allocr)) {
        float s = -2.0f;
        ggml_backend_tensor_set(dist_scale, &s, 0, sizeof(s));
    }

    struct ggml_tensor * inpL = ggml_cont(ctx0, ggml_transpose(ctx0, encoded_inp));
    struct ggml_tensor * residual = inpL;
    struct ggml_tensor * indices;

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = model.quantizer.blocks[i];

        // compute distance
        // [seq_length, n_bins]
        struct ggml_tensor * dp = ggml_scale(
                ctx0, ggml_mul_mat(ctx0, block.embed, residual), dist_scale);

        // [n_bins]
        struct ggml_tensor * sqr_embed = ggml_sqr(ctx0, block.embed);
        struct ggml_tensor * sqr_embed_nrm = ggml_sum_rows(ctx0, sqr_embed);

        // [seq_length]
        struct ggml_tensor * sqr_inp = ggml_sqr(ctx0, residual);
        struct ggml_tensor * sqr_inp_nrm = ggml_sum_rows(ctx0, sqr_inp);

        // [seq_length, n_bins]
        struct ggml_tensor * dist = ggml_add(ctx0, ggml_repeat(ctx0, sqr_inp_nrm, dp), dp);
        dist = ggml_add(ctx0, ggml_repeat(ctx0, ggml_transpose(ctx0, sqr_embed_nrm), dist), dist);
        dist = ggml_neg(ctx0, dist);

        // take the argmax over the column dimension
        // [seq_length]
        indices = ggml_argmax(ctx0, dist);

        // look up in embedding table
        struct ggml_tensor * quantized = ggml_get_rows(ctx0, block.embed, indices);

        residual = ggml_sub(ctx0, residual, quantized);

        codes = ggml_set_1d(ctx0, codes, indices, i*codes->nb[1]);
    }

    return codes;
}

struct ggml_tensor * encodec_forward_quantizer_decode(
               struct encodec_context * ectx,
                  struct ggml_context * ctx0,
                   struct ggml_tensor * codes) {
    if (!codes) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto & model   = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr  = ectx->allocr;

    const int hidden_dim = hparams.hidden_dim;
    const int seq_length = codes->ne[0];

    const int n_bins     = hparams.n_bins;
    const int sr         = hparams.sr;
    const int bandwidth  = hparams.bandwidth;
    const int hop_length = hparams.hop_length;

    const int frame_rate = (int) ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    assert(n_q == codes->ne[1]);

    struct ggml_tensor * quantized_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, seq_length);
    ggml_allocr_alloc(allocr, quantized_out);

    if (!ggml_allocr_is_measure(allocr)) {
        quantized_out = ggml_set_zero(quantized_out);
    }

    for (int i = 0; i < n_q; i++) {
        encodec_quant_block block = model.quantizer.blocks[i];

        struct ggml_tensor * indices   = ggml_view_1d(ctx0, codes, seq_length, i*codes->nb[1]);
        struct ggml_tensor * quantized = ggml_get_rows(ctx0, block.embed, indices);

        quantized_out = ggml_add(ctx0, quantized_out, quantized);
    }

    quantized_out = ggml_cont(ctx0, ggml_transpose(ctx0, quantized_out));

    return quantized_out;
}

struct ggml_tensor * encodec_forward_decoder(
               struct encodec_context * ectx,
                  struct ggml_context * ctx0,
                   struct ggml_tensor * quantized_out) {
    if (!quantized_out) {
        fprintf(stderr, "%s: null input tensor\n", __func__);
        return NULL;
    }

    const auto & model = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr = ectx->allocr;

    const int * ratios      = hparams.ratios;
    const int kernel_size   = hparams.kernel_size;
    const int res_kernel_sz = hparams.residual_kernel_size;
    const int stride        = hparams.stride;

    struct ggml_tensor * inpL = strided_conv_1d(
        ctx0, quantized_out, model.decoder.init_conv_w,
        model.decoder.init_conv_b, stride);

    // lstm
    {
        struct ggml_tensor * cur = inpL;

        const encodec_lstm lstm = model.decoder.lstm;

        // first lstm layer
        struct ggml_tensor * hs1 = forward_pass_lstm_unilayer(
            ctx0, allocr, cur, lstm.l0_ih_w, lstm.l0_hh_w,
            lstm.l0_ih_b, lstm.l0_hh_b);

        // second lstm layer
        struct ggml_tensor * out = forward_pass_lstm_unilayer(
            ctx0, allocr, hs1, lstm.l1_ih_w, lstm.l1_hh_w,
            lstm.l1_ih_b, lstm.l1_hh_b);

        inpL = ggml_add(ctx0, inpL, out);
    }

    for (int layer_ix = 0; layer_ix < 4; layer_ix++) {
        encodec_decoder_block block = model.decoder.blocks[layer_ix];

        // upsampling layers
        inpL = ggml_elu(ctx0, inpL);

        inpL = strided_conv_transpose_1d(
            ctx0, inpL, block.us_conv_w, block.us_conv_b, ratios[layer_ix]);

        struct ggml_tensor * current = inpL;

        // shortcut
        struct ggml_tensor * shortcut = strided_conv_1d(
            ctx0, inpL, block.conv_sc_w, block.conv_sc_b, stride);

        // conv1
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_1_w, block.conv_1_b, stride);

        // conv2
        current = ggml_elu(ctx0, current);

        current = strided_conv_1d(
            ctx0, current, block.conv_2_w, block.conv_2_b, stride);

        // residual connection
        inpL = ggml_add(ctx0, current, shortcut);
    }

    // final conv
    inpL = ggml_elu(ctx0, inpL);

    struct ggml_tensor * decoded_inp = strided_conv_1d(
            ctx0, inpL, model.decoder.final_conv_w,
            model.decoder.final_conv_b, stride);

    return decoded_inp;
}

struct ggml_cgraph * encodec_build_graph(
        struct encodec_context * ectx,
            std::vector<float> & inp_audio,
        const encodec_run_mode   mode) {
    assert(mode == encodec_run_mode::full || mode == encodec_run_mode::encode);

    const auto & model = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr = ectx->allocr;

    const int n_q = hparams.n_q;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the
    // ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
    };

    struct ggml_context * ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    const int N = inp_audio.size();

    struct ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, N);
    ggml_allocr_alloc(allocr, inp);

    // avoid writing to tensors if we are only measuring the memory usage
    if (!ggml_allocr_is_measure(allocr)) {
        ggml_backend_tensor_set(inp, inp_audio.data(), 0, N*ggml_element_size(inp));
    }

    struct ggml_tensor * encoded   = encodec_forward_encoder(ectx, ctx0, inp);
    struct ggml_tensor * codes     = encodec_forward_quantizer_encode(ectx, ctx0, encoded);
    struct ggml_tensor * quantized = encodec_forward_quantizer_decode(ectx, ctx0, codes);
    struct ggml_tensor * decoded   = encodec_forward_decoder(ectx, ctx0, quantized);

    switch(mode) {
        case encodec_run_mode::full:
            {
                ggml_build_forward_expand(gf, decoded);
            } break;
        case encodec_run_mode::encode:
            {
                ggml_build_forward_expand(gf, codes);
            } break;
        case encodec_run_mode::decode:
            {
                return NULL;
            } break;
        default:
            {
                fprintf(stderr, "%s: unknown run mode\n", __func__);
                return NULL;
            } break;
    }

    ggml_free(ctx0);

    ectx->encoded = encoded;
    ectx->codes   = codes;
    ectx->decoded = decoded;

    return gf;
}

struct ggml_cgraph * encodec_build_graph(
        struct encodec_context * ectx,
          std::vector<int32_t> & codes,
        const encodec_run_mode   mode) {
    assert(mode == encodec_run_mode::decode);

    const auto & model = ectx->model;
    const auto & hparams = model.hparams;
    const auto & allocr = ectx->allocr;

    const int n_bins     = hparams.n_bins;
    const int sr         = hparams.sr;
    const int bandwidth  = hparams.bandwidth;
    const int hop_length = hparams.hop_length;

    const int frame_rate = (int) ceilf(sr / hop_length);
    const int n_q = get_num_quantizers_for_bandwidth(n_bins, frame_rate, bandwidth);

    if (codes.size() % n_q != 0) {
        fprintf(stderr, "%s: invalid number of codes (length=%zu)\n", __func__, codes.size());
        return NULL;
    }

    const int N = codes.size() / n_q;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the
    // ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(ggml_params);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, N, n_q);
    ggml_allocr_alloc(allocr, inp_codes);

    // avoid writing to tensors if we are only measuring the memory usage
    if (!ggml_allocr_is_measure(allocr)) {
        ggml_backend_tensor_set(inp_codes, codes.data(), 0, N*n_q*ggml_element_size(inp_codes));
    }

    struct ggml_tensor * quantized = encodec_forward_quantizer_decode(ectx, ctx0, inp_codes);
    struct ggml_tensor * decoded   = encodec_forward_decoder(ectx, ctx0, quantized);

    switch(mode) {
        case encodec_run_mode::decode:
            {
                ggml_build_forward_expand(gf, decoded);
            } break;
        default:
            {
                fprintf(stderr, "%s: unknown run mode\n", __func__);
                return NULL;
            } break;
    }

    ggml_free(ctx0);

    ectx->codes   = inp_codes;
    ectx->decoded = decoded;

    return gf;
}

bool encodec_eval_internal(
        struct encodec_context * ectx,
            std::vector<float> & raw_audio,
                     const int   n_threads,
        const encodec_run_mode   mode) {
    auto & model  = ectx->model;
    auto & allocr = ectx->allocr;

    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = encodec_build_graph(ectx, raw_audio, mode);

    if (!gf) {
        fprintf(stderr, "%s: failed to build Encodec graph\n", __func__);
        return false;
    }

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif
    ggml_backend_graph_compute(model.backend, gf);

    return true;
}

bool encodec_eval_internal(
        struct encodec_context * ectx,
          std::vector<int32_t> & codes,
                     const int   n_threads,
        const encodec_run_mode   mode) {
    auto & model  = ectx->model;
    auto & allocr = ectx->allocr;

    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = encodec_build_graph(ectx, codes, mode);

    if (!gf) {
        fprintf(stderr, "%s: failed to build Encodec graph\n", __func__);
        return false;
    }

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);

    // run the computation
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif
    ggml_backend_graph_compute(model.backend, gf);

    return true;
}


bool encodec_eval(
            struct encodec_context * ectx,
                std::vector<float> & raw_audio,
                         const int   n_threads,
            const encodec_run_mode   mode) {
    const int64_t t_start_ms = ggml_time_ms();

    // allocate the compute buffer
    {
        // alignment required by the backend
        size_t align = ggml_backend_get_alignment(ectx->model.backend);
        ectx->allocr = ggml_allocr_new_measure(align);

        // create the graph for memory usage estimation
        struct ggml_cgraph * gf = encodec_build_graph(ectx, raw_audio, mode);

        if (!gf) {
            fprintf(stderr, "%s: failed to build Encodec graph\n", __func__);
            return false;
        }

        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(ectx->allocr, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(ectx->allocr);
        ectx->buf_compute = ggml_backend_alloc_buffer(ectx->model.backend, mem_size);
        ectx->allocr = ggml_allocr_new_from_buffer(ectx->buf_compute);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size/1024.0/1024.0);
    }

    // encodec eval
    if (!encodec_eval_internal(ectx, raw_audio, n_threads, mode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    ectx->t_compute_ms = ggml_time_ms() - t_start_ms;

    return true;
}

bool encodec_eval(
            struct encodec_context * ectx,
              std::vector<int32_t> & codes,
                         const int   n_threads,
            const encodec_run_mode   mode) {
    const int64_t t_start_ms = ggml_time_ms();

    // allocate the compute buffer
    {
        // alignment required by the backend
        size_t align = ggml_backend_get_alignment(ectx->model.backend);
        ectx->allocr = ggml_allocr_new_measure(align);

        // create the graph for memory usage estimation
        struct ggml_cgraph * gf = encodec_build_graph(ectx, codes, mode);

        if (!gf) {
            fprintf(stderr, "%s: failed to build Encodec graph\n", __func__);
            return false;
        }

        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(ectx->allocr, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(ectx->allocr);
        ectx->buf_compute = ggml_backend_alloc_buffer(ectx->model.backend, mem_size);
        ectx->allocr = ggml_allocr_new_from_buffer(ectx->buf_compute);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n\n", __func__, mem_size/1024.0/1024.0);
    }

    // encodec eval
    if (!encodec_eval_internal(ectx, codes, n_threads, mode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    ectx->t_compute_ms = ggml_time_ms() - t_start_ms;

    return true;
}

bool encodec_reconstruct_audio(
            struct encodec_context * ectx,
                std::vector<float> & raw_audio,
                               int   n_threads) {
    if (!encodec_eval(ectx, raw_audio, n_threads, encodec_run_mode::full)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    if (!ectx->decoded) {
        fprintf(stderr, "%s: null decoded tensor\n", __func__);
        return false;
    }

    struct ggml_tensor * decoded = ectx->decoded;

    auto & out_audio = ectx->out_audio;

    int out_length = decoded->ne[0];
    out_audio.resize(out_length);

    ggml_backend_tensor_get(decoded, out_audio.data(), 0, out_length*ggml_element_size(decoded));

    return true;
}

bool encodec_compress_audio(
            struct encodec_context * ectx,
                std::vector<float> & raw_audio,
                               int   n_threads) {
    if(!encodec_eval(ectx, raw_audio, n_threads, encodec_run_mode::encode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    if (!ectx->codes) {
        fprintf(stderr, "%s: null codes tensor\n", __func__);
        return false;
    }

    struct ggml_tensor * codes = ectx->codes;

    auto & out_codes = ectx->out_codes;

    int out_length = codes->ne[0]*codes->ne[1];
    out_codes.resize(out_length);

    ggml_backend_tensor_get(codes, out_codes.data(), 0, out_length*ggml_element_size(codes));

    return true;
}

bool encodec_decompress_audio(
            struct encodec_context * ectx,
              std::vector<int32_t> & codes,
                               int   n_threads) {
    if (!encodec_eval(ectx, codes, n_threads, encodec_run_mode::decode)) {
        fprintf(stderr, "%s: failed to run encodec eval\n", __func__);
        return false;
    }

    if (!ectx->decoded) {
        fprintf(stderr, "%s: null decoded tensor\n", __func__);
        return false;
    }

    struct ggml_tensor * decoded = ectx->decoded;

    auto & out_audio = ectx->out_audio;

    int out_length = decoded->ne[0];
    out_audio.resize(out_length);

    ggml_backend_tensor_get(decoded, out_audio.data(), 0, out_length*ggml_element_size(decoded));

    return true;
}

struct encodec_context * encodec_load_model(
                const std::string & model_path,
                              int   n_gpu_layers,
          encodec_verbosity_level   verbosity) {
    int64_t t_start_load_us = ggml_time_us();

    struct encodec_context * ectx = new encodec_context();

    ectx->model = encodec_model();
    if (!encodec_load_model_weights(model_path, ectx->model, n_gpu_layers, verbosity)) {
        fprintf(stderr, "%s: failed to load model weights from '%s'\n", __func__, model_path.c_str());
        return {};
    }

    // pre-compute the number of codebooks required
    int bandwidth = ectx->model.hparams.bandwidth;
    int sr        = ectx->model.hparams.sr;

    int hop_length = 1;
    for (int i = 0; i < 4; i++) {
        hop_length *= ectx->model.hparams.ratios[i];
    }
    ectx->model.hparams.hop_length = hop_length;

    ectx->model.hparams.n_q = get_num_codebooks(bandwidth, hop_length, sr);
    fprintf(stderr, "%s: n_q = %d\n", __func__, ectx->model.hparams.n_q);

    ectx->t_load_us = ggml_time_us() - t_start_load_us;

    return ectx;
}

void encodec_free(struct encodec_context * ectx) {
    if (!ectx) {
        return;
    }

    if (ectx->model.ctx) {
        ggml_free(ectx->model.ctx);
    }

    if (ectx->buf_compute) {
        ggml_backend_buffer_free(ectx->buf_compute);
    }

    ggml_backend_buffer_free(ectx->model.buffer_w);
    ggml_backend_free(ectx->model.backend);

    delete ectx;
}

void encodec_set_target_bandwidth(struct encodec_context * ectx, int bandwidth) {
    ectx->model.hparams.bandwidth = bandwidth;
}

void encodec_set_sample_rate(struct encodec_context * ectx, int sample_rate) {
    ectx->model.hparams.sr = sample_rate;
}
