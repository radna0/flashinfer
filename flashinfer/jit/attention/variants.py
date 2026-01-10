"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

attention_sink_fa2_decl = r"""
struct AttentionSink : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ AttentionSink(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = (params.window_left >= 0) ? params.window_left : kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return (kv_idx + qo_len + window_left >= kv_len + qo_idx);
  })

  REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, {
    float log_sink = (kv_tile_idx == 0 && qo_head_idx < params.num_qo_heads) ? params.sink[qo_head_idx] * math::log2e : -math::inf;
    float m_new = (log_sink > m) ? log_sink : m;
    scale = math::ptx_exp2(m - m_new);
    float d_new = math::ptx_exp2(log_sink - m_new) + d * scale;
    // Update m and d
    m = m_new;
    d = d_new;
  })

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d) : 0.f;
    return output * scale * d_rcp;
  });
};
"""

attention_sink_fa3_decl = r"""

// Needed for FP8 kernel path:
// - get_{q,k,v}_scale helpers
// - scale handling conventions (scale_pv semantics)
// - compatibility with quantization mainloop expectations (PQuantize/ODequantize)
// Note: BF16 prefill path already indirectly includes this via prefill_sm90.cuh, but
// the FP8 prefill path includes quantization/prefill_sm90.cuh which does not.
// Include explicitly to keep the custom AttentionSink variant self-contained.
#include <flashinfer/attention/hopper/variants.cuh>

template <int NUM_ROWS_PER_THREAD>
struct OnlineSoftmaxWithSink {
  constexpr static float fill_value = -math::inf;
  using TensorT = decltype(make_tensor<float>(Shape<Int<NUM_ROWS_PER_THREAD>>{}));
  TensorT row_max, row_sum, scores_scale;
  float sm_scale_log2;
  float log_sink;

  CUTLASS_DEVICE OnlineSoftmaxWithSink(float sm_scale_log2, float log_sink) : sm_scale_log2(sm_scale_log2), log_sink(log_sink) {
    clear(scores_scale);
  };

  __forceinline__ __device__ TensorT get_lse() const { return row_sum; }

  template <bool init, typename Tensor0>
  __forceinline__ __device__ void update(Tensor0& acc_s) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));

    static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
    if constexpr (init) {
      reduce_max</*init=*/true>(scores, row_max);
      scale_apply_exp2(scores, row_max, sm_scale_log2);
      reduce_sum</*init=*/true, /*warp_reduce=*/false>(scores, row_sum);
    } else {
      // update row_max
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      reduce_max</*init=*/false>(scores, row_max);
      // update scores_scale and scale row_sum
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = row_max(mi);
        scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * sm_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
      // perform exp2 on scores
      scale_apply_exp2(scores, row_max, sm_scale_log2);
      // update row_sum
      reduce_sum</*init=*/false, /*warp_reduce=*/false>(scores, row_sum);
    }
  };

  template <typename Tensor0>
  __forceinline__ __device__ void finalize(Tensor0& acc_s, float pv_scale = 1.f) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    // Note (Yilong): use pv_scale to dequantize the output
    Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == NUM_ROWS_PER_THREAD);
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);

#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
      float m = row_max(mi) * sm_scale_log2;
      float d = row_sum(mi);

      float m_new = (log_sink > m) ? log_sink : m;
      float scale = math::ptx_exp2(m - m_new);
      float d_new = math::ptx_exp2(log_sink - m_new) + d * scale;

      // Update m and d
      row_max(mi) = m_new;
      row_sum(mi) = d_new;

      scores_scale(mi) = pv_scale * scale / d_new;
      row_sum(mi) = row_max(mi) + math::ptx_log2(d_new);
    }
  };

  template <typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1& acc_o) {
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == NUM_ROWS_PER_THREAD);
#pragma unroll
    for (int mi = 0; mi < size(row_max); ++mi) {
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale(mi);
      }
    }
  };
};

struct AttentionSink : AttentionVariantBase {
  float sm_scale_log2;
  float log_sink;
  float p_scale;
  float scale_pv;
  int qo_len, kv_len;

  // Init
  template <typename MainloopParams, typename BlockCoord>
  __device__ __host__ AttentionSink(const MainloopParams& params, const BlockCoord& block_coord) {
    auto [_, qo_head_idx, kv_head_idx, ___, ____, qo_len_, kv_len_, batch_idx] =
        block_coord;
    float q_scale = get_q_scale(params.additional_params, qo_head_idx);
    float k_scale = get_k_scale(params.additional_params, kv_head_idx);
    sm_scale_log2 = q_scale * k_scale * params.additional_params.sm_scale * math::log2e;
    log_sink = params.additional_params.sink[qo_head_idx] * math::log2e;
    float v_scale = get_v_scale(params.additional_params, kv_head_idx);

    // For FP8 kernels, attention probabilities are quantized to FP8 before the PV GEMM.
    // Match StandardFP8Attention's convention: PQuantize multiplies by p_scale (max FP8),
    // and the online softmax finalize applies scale_pv = v_scale / p_scale.
    //
    // For non-FP8 kernels, p_scale must be 1 and scale_pv should be the (optional) v_scale.
    using DTypeKV = std::remove_const_t<std::remove_pointer_t<decltype(params.K_ptr)>>;
    if constexpr (std::is_same_v<DTypeKV, cutlass::float_e4m3_t> ||
                  std::is_same_v<DTypeKV, cutlass::float_e5m2_t>) {
      p_scale = std::numeric_limits<DTypeKV>::max();
      scale_pv = v_scale / p_scale;
    } else {
      p_scale = 1.0f;
      scale_pv = v_scale;
    }

    qo_len = qo_len_;
    kv_len = kv_len_;
  }

  template <int NUM_ROWS_PER_THREAD>
  __device__ auto GetAttentionUpdater() {
    return OnlineSoftmaxWithSink<NUM_ROWS_PER_THREAD>(sm_scale_log2, log_sink);
  }

  template <typename Tensor0>
  __device__ __forceinline__ void PQuantize(Tensor0& tSrS) {
#pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
      tSrS(i) *= p_scale;
    }
  }

  template <typename MainloopParams, typename Tensor0>
  __device__ __forceinline__ void ODequantize(const MainloopParams& params, Tensor0& tOrO,
                                              uint32_t qo_head_idx, uint32_t kv_head_idx) {
    // PV dequantization is fused into OnlineSoftmaxWithSink::finalize via scale_pv.
  }
};
"""

attention_sink_decl = {
    "fa2": attention_sink_fa2_decl,
    "fa3": attention_sink_fa3_decl,
}
