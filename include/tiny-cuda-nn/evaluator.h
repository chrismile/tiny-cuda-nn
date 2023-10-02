/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

/** @file   evaluator.h
 *  @author Thomas Müller, NVIDIA; Christoph Neuhauser, TUM
 *  @brief  Class that performs evaluation of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <random>

namespace tcnn {

template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
class Evaluator : public ObjectWithMutableHyperparams {
public:
	Evaluator(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model)
	: m_model{model} {
        size_t n_params = m_model->n_params();
        log_debug("Evaluator: allocating {} params.", n_params);

        m_params_buffer.resize(sizeof(PARAMS_T) * n_params);
        m_params_buffer.memset(0);
        reset_param_pointers();
	}

	virtual ~Evaluator() {}

    void update_hyperparams(const json& params) override {
    }

    json hyperparams() const override {
		return {
			{"otype", "Evaluator"}
		};
	}

	PARAMS_T* params() const {
		return m_params;
	}

	void set_params_full_precision(const float* params, size_t n_params, bool device_ptr = false) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set fp params because buffer has the wrong size."};
		}

        if (device_ptr) {
            parallel_for_gpu(n_params, [params_fp=params, params_inference=m_params] __device__ (size_t i) {
                params_inference[i] = (PARAMS_T)params_fp[i];
            });
        } else {
            auto* params_cpu = new PARAMS_T[n_params];
            for (size_t i = 0; i < n_params; i++) {
                params_cpu[i] = (PARAMS_T)params[i];
            }
            CUDA_CHECK_THROW(cudaMemcpy(m_params, params_cpu, sizeof(PARAMS_T)*n_params, cudaMemcpyHostToDevice));
            delete[] params_cpu;
        }
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void set_params(const PARAMS_T* params, size_t n_params, bool device_ptr = false) {
		if (n_params != m_model->n_params()) {
			throw std::runtime_error{"Can't set params because buffer has the wrong size."};
		}

		CUDA_CHECK_THROW(cudaMemcpy(m_params, params, sizeof(PARAMS_T)*n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() {
		return m_model;
	}

	json serialize(bool serialize_optimizer = false) {
		size_t n_params = m_model->n_params();

		json data;
		data["n_params"] = n_params;
		data["params_type"] = type_to_string<PARAMS_T>();
		data["params_binary"] = gpu_memory_to_json_binary(m_params, sizeof(PARAMS_T)*n_params);

		return data;
	}

	void deserialize(const json& data) {
		std::string type = data.value("params_type", type_to_string<PARAMS_T>());
		if (type == "float") {
			GPUMemory<float> params = data["params_binary"];
			set_params_full_precision(params.data(), params.size(), true);
		} else if (type == "__half") {
			GPUMemory<__half> params_hp = data["params_binary"];
			size_t n_params = params_hp.size();

			GPUMemory<PARAMS_T> params(n_params);
			parallel_for_gpu(n_params, [params=params.data(), params_hp=params_hp.data()] __device__ (size_t i) {
				params[i] = (PARAMS_T)params_hp[i];
			});

			set_params(params.data(), params.size(), true);
		} else {
			throw std::runtime_error{"Evaluator: snapshot parameters must be of type float of __half"};
		}

		reset_param_pointers();
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}

	void reset_param_pointers() {
		size_t n_params = m_model->n_params();
        m_params = (PARAMS_T*)(m_params_buffer.data());
		m_model->set_params(m_params, m_params, nullptr);
	}

	size_t n_params() const {
		return m_model->n_params();
	}

private:
	std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
	CudaGraph m_graph;
	GPUMemory<char> m_params_buffer;
	PARAMS_T* m_params = nullptr;
};

}
