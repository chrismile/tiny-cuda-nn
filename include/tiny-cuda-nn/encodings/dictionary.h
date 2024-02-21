/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION, Christoph Neuhauser.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   composite.h
 *  @author Christoph Neuhauser, TUM
 *  @brief  Embedding similar to: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
 * Input: Vector of integers.
 * Output: Set of feature vectors.
 * JSON settings:
 * - num_embeddings (dictionary size)
 * - embedding_dim (number of features per dictionary entry)
 */

#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/common_device.h>

#include <numeric>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

namespace tcnn {

template <typename T>
__global__ void dictionary_encoding(
        const uint32_t num_elements,
        const uint32_t num_embeddings,
        const uint32_t num_features,
        const uint32_t num_to_encode,
        const uint32_t num_to_pad,
        const T* __restrict__ dictionary,
        MatrixView<const float> data_in,
        MatrixView<T> data_out)
{
    const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (encoded_index >= num_elements) return;

    const uint32_t fan_out_encoded = num_to_encode * num_features;
    const uint32_t fan_out = fan_out_encoded + num_to_pad;

    const uint32_t i = encoded_index / fan_out;
    const uint32_t j = encoded_index - i * fan_out;

    if (j >= fan_out_encoded) {
        data_out(j, i) = 1;
    } else {
        const uint32_t encoded_input_feature_i = j / num_features;
        const uint32_t feature_idx = j & num_features;
        const auto dictionary_entry_idx = uint(data_in(encoded_input_feature_i, i));
        data_out(j, i) = dictionary[dictionary_entry_idx * num_features + feature_idx];
    }
}

template <typename T, typename GRAD_T>
__global__ void dictionary_encoding_backward(
        const uint32_t num_elements,
        const uint32_t n_dims_to_encode,
        const uint32_t num_embeddings,
        const uint32_t num_features,
        const T* __restrict__ dictionary,
        GRAD_T* __restrict__ dictionary_gradient,
        MatrixView<const T> dL_dy,
        MatrixView<const float> data_in,
        MatrixView<float> dL_dx)
{
    const uint32_t encoded_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (encoded_index >= num_elements) return;

    const uint32_t i = encoded_index / n_dims_to_encode;
    const uint32_t j = encoded_index - i * n_dims_to_encode;
    const auto dictionary_entry_idx = uint(data_in(j, i));

    dL_dx(j, i) = 0.0f;

    if (dictionary_gradient) {
        for (uint32_t feature_idx = 0; feature_idx < num_features; feature_idx++) {
            atomicAdd(
                    dictionary_gradient + (dictionary_entry_idx * num_features + feature_idx),
                    (GRAD_T)dL_dy(j * num_features + feature_idx, i));
        }
    }
}

template <typename T>
class DictionaryEncoding : public Encoding<T> {
public:
    // See remarks in grid.h regarding precision.
    using grad_t = float;

    DictionaryEncoding(uint32_t n_embeddings, uint32_t n_features, uint32_t n_dims_to_encode)
            : m_n_embeddings{n_embeddings}, m_n_features{n_features}, m_n_dims_to_encode{n_dims_to_encode} {
        m_n_output_dims = m_n_dims_to_encode * m_n_features;
    }

    std::unique_ptr<Context> forward_impl(
            cudaStream_t stream,
            const GPUMatrixDynamic<float>& input,
            GPUMatrixDynamic<T>* output = nullptr,
            bool use_inference_params = false,
            bool prepare_input_gradients = false
    ) override {
        if (!output || padded_output_width() == 0) {
            return std::make_unique<Context>();
        }

        linear_kernel(dictionary_encoding<T>, 0, stream,
                      input.n() * padded_output_width(),
                      m_n_embeddings,
                      m_n_features,
                      m_n_dims_to_encode,
                      m_n_to_pad,
                      use_inference_params ? this->inference_params() : this->params(),
                      input.view(),
                      output->view()
        );

        return std::make_unique<Context>();
    }

    void backward_impl(
            cudaStream_t stream,
            const Context& ctx,
            const GPUMatrixDynamic<float>& input,
            const GPUMatrixDynamic<T>& output,
            const GPUMatrixDynamic<T>& dL_doutput,
            GPUMatrixDynamic<float>* dL_dinput = nullptr,
            bool use_inference_params = false,
            GradientMode param_gradients_mode = GradientMode::Overwrite
    ) override {
        if (!dL_dinput) {
            return;
        }

        grad_t* dictionary_gradient = nullptr;
        // We accumulate gradients with grad_t precision, which, for performance reasons, is not always T.
        // If not, accumulate in a temporary buffer and cast later.
        GPUMemoryArena::Allocation dictionary_gradient_tmp;
        if (param_gradients_mode != GradientMode::Ignore) {
            if (!std::is_same<grad_t, T>::value) {
                dictionary_gradient_tmp = allocate_workspace(stream, m_n_embeddings * m_n_features * sizeof(grad_t));
                dictionary_gradient = (grad_t*)dictionary_gradient_tmp.data();
            } else {
                dictionary_gradient = (grad_t*)this->gradients();
            }

            if (param_gradients_mode == GradientMode::Overwrite) {
                CUDA_CHECK_THROW(cudaMemsetAsync(dictionary_gradient, 0, n_params() * sizeof(grad_t), stream));
            }
        }

        linear_kernel(dictionary_encoding_backward<T, grad_t>, 0, stream,
                      input.n() * m_n_dims_to_encode,
                      m_n_dims_to_encode,
                      m_n_embeddings,
                      m_n_features,
                      use_inference_params ? this->inference_params() : this->params(),
                      dictionary_gradient,
                      dL_doutput.view(),
                      input.view(),
                      dL_dinput->view()
        );

        if (param_gradients_mode != GradientMode::Ignore) {
            if (!std::is_same<grad_t, T>::value) {
                parallel_for_gpu(stream, n_params(), [grad=this->gradients(), grad_tmp=dictionary_gradient] __device__ (size_t i) {
                    grad[i] = (T)grad_tmp[i];
                });
            }
        }
    }

    uint32_t input_width() const override {
        return m_n_dims_to_encode;
    }

    uint32_t padded_output_width() const override {
        return m_n_output_dims + m_n_to_pad;
    }

    uint32_t output_width() const override {
        return padded_output_width();
    }

    uint32_t required_input_alignment() const override {
        return 1;
    }

    void set_padded_output_width(uint32_t padded_output_width) override {
        CHECK_THROW(padded_output_width >= m_n_output_dims);
        m_n_to_pad = padded_output_width - m_n_output_dims;
    }

    uint32_t required_output_alignment() const override {
        return 1;
    }

    MatrixLayout preferred_output_layout() const override {
        return AoS;
    }

    void initialize_params(pcg32& rnd, float* params_full_precision, float scale = 1) override {
        // Initialize the dictionary from the GPU, because the number of parameters can be quite large.
        generate_random_uniform<float>(rnd, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
    }

    size_t n_params() const override {
        return m_n_embeddings * m_n_features;
    }

    json hyperparams() const override {
        return {
                {"otype", "Dictionary"},
                {"num_embeddings", m_n_embeddings},
                {"embedding_dim", m_n_features},
        };
    }

private:
    uint32_t m_n_embeddings;
    uint32_t m_n_features;
    uint32_t m_n_dims_to_encode;

    // derived sizes
    uint32_t m_n_output_dims;
    uint32_t m_n_to_pad = 0;
};

}
