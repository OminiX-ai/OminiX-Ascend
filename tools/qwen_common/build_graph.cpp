#include "build_graph.h"
#include "ggml.h"
#include <string>

void cb(ggml_context *ctx0, ggml_tensor *cur0, const char *name, int il) {
  // TODO
  //   if (ctx_manager_->debug_graph_) {
  //     ggml_context *ctx0 = ctx_compute_.get();
  //     ggml_tensor *cur = ggml_cpy(ctx0, cur0, ggml_dup_tensor(ctx0, cur0));
  //     std::string cur_name =
  //         il >= 0 ? std::string(name) + "_" + std::to_string(il) : name;
  //     ggml_set_name(cur, cur_name.c_str());
  //     ggml_set_output(cur);
  //     ggml_build_forward_expand(gf, cur);
  //     ctx_manager_->debug_print_tensors_.push_back(cur);
  //   }
}

ggml_tensor *build_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *mw,
                        ggml_tensor *mb, norm_type type, float norm_eps,
                        int n_groups, int il) {
  // For GroupNorm: ggml_group_norm expects channels in ne[2], but conv1d
  // outputs (OL, OC, N) with channels in ne[1]. Reshape before/after.
  bool gn_needs_reshape = (type == NORM_TYPE_GROUP && cur->ne[2] == 1 && cur->ne[1] > 1);
  int64_t saved_ne0 = cur->ne[0];
  int64_t saved_ne1 = cur->ne[1];
  if (gn_needs_reshape) {
    cur = ggml_reshape_3d(ctx0, cur, saved_ne0, 1, saved_ne1);
  }

  switch (type) {
  case NORM_TYPE_RMS:
    cur = ggml_rms_norm(ctx0, cur, norm_eps);
    break;
  case NORM_TYPE_GROUP:
    cur = ggml_group_norm(ctx0, cur, n_groups, norm_eps);
    break;
  case NORM_TYPE_NORMAL:
  default:
    cur = ggml_norm(ctx0, cur, norm_eps);
    break;
  }

  if (mw || mb) {
    cb(ctx0, cur, "norm", il);
  }

  // Apply affine weight (gamma)
  if (mw) {
    if (gn_needs_reshape) {
      // Weight (C,) reshaped to (1, 1, C) to broadcast against (OL, 1, C)
      ggml_tensor *mw_reshaped = ggml_reshape_3d(ctx0, mw, 1, 1, mw->ne[0]);
      cur = ggml_mul(ctx0, cur, mw_reshaped);
    } else {
      cur = ggml_mul(ctx0, cur, mw);
    }
    if (mb) {
      cb(ctx0, cur, "norm_w", il);
    }
  }

  // Apply affine bias (beta)
  if (mb) {
    if (gn_needs_reshape) {
      ggml_tensor *mb_reshaped = ggml_reshape_3d(ctx0, mb, 1, 1, mb->ne[0]);
      cur = ggml_add(ctx0, cur, mb_reshaped);
    } else {
      cur = ggml_add(ctx0, cur, mb);
    }
  }

  // Restore original layout
  if (gn_needs_reshape) {
    cur = ggml_reshape_3d(ctx0, cur, saved_ne0, saved_ne1, 1);
  }

  return cur;
}

ggml_tensor *build_linear(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int il) {
  cur = ggml_mul_mat(ctx0, w, cur);
  cb(ctx0, cur, "linear", il);
  if (b) {
    cur = ggml_add(ctx0, cur, b);
    cb(ctx0, cur, "linear_b", il);
  }
  return cur;
}

ggml_tensor *build_ffn(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *up,
                       ggml_tensor *up_b, ggml_tensor *gate,
                       ggml_tensor *gate_b, ggml_tensor *down,
                       ggml_tensor *down_b, ffn_op_type type_op, int il) {
  ggml_tensor *tmp = up ? ggml_mul_mat(ctx0, up, cur) : cur;
  cb(ctx0, tmp, "ffn_up", il);

  if (up_b) {
    tmp = ggml_add(ctx0, tmp, up_b);
    cb(ctx0, tmp, "ffn_up_b", il);
  }

  if (gate) {
    cur = ggml_mul_mat(ctx0, gate, cur);
    cb(ctx0, cur, "ffn_gate", il);

    if (gate_b) {
      cur = ggml_add(ctx0, cur, gate_b);
      cb(ctx0, cur, "ffn_gate_b", il);
    }
  } else {
    cur = tmp;
  }

  // we only support parallel ffn for now
  switch (type_op) {
  case FFN_SILU:
    if (gate) {
      cur = ggml_swiglu_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_swiglu", il);
    } else {
      cur = ggml_silu(ctx0, cur);
      cb(ctx0, cur, "ffn_silu", il);
    }
    break;
  case FFN_GELU:
    if (gate) {
      cur = ggml_geglu_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_geglu", il);
    } else {
      cur = ggml_gelu(ctx0, cur);
      cb(ctx0, cur, "ffn_gelu", il);
    }
    break;
  case FFN_GELU_ERF:
    if (gate) {
      cur = ggml_geglu_erf_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_geglu_erf", il);
    } else {
      cur = ggml_gelu_erf(ctx0, cur);
      cb(ctx0, cur, "ffn_gelu_erf", il);
    }
    break;
  case FFN_GELU_QUICK:
    if (gate) {
      cur = ggml_geglu_quick_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_geglu_quick", il);
    } else {
      cur = ggml_gelu_quick(ctx0, cur);
      cb(ctx0, cur, "ffn_gelu_quick", il);
    }
    break;
  }

  if (down) {
    cur = ggml_mul_mat(ctx0, down, cur);
  }

  if (down_b) {
    cb(ctx0, cur, "ffn_down", il);
  }

  if (down_b) {
    cur = ggml_add(ctx0, cur, down_b);
  }

  return cur;
}

ggml_tensor *build_attn(ggml_context *ctx0, ggml_tensor *wo, ggml_tensor *wo_b,
                        ggml_tensor *q_cur, ggml_tensor *k_cur,
                        ggml_tensor *v_cur, ggml_tensor *kq_mask,
                        float kq_scale, int il) {
  // these nodes are added to the graph together so that they are not
  // reordered by doing so, the number of splits in the graph is reduced
  //   ggml_build_forward_expand(gf, q_cur);
  //   ggml_build_forward_expand(gf, k_cur);
  //   ggml_build_forward_expand(gf, v_cur);

  ggml_tensor *q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
  // cb(q, "q", il);

  ggml_tensor *k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
  // cb(k, "k", il);

  ggml_tensor *v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
  v = ggml_cont(ctx0, v);
  // cb(k, "v", il);

  ggml_tensor *cur;

  // TODO @ngxson : support flash attention
  {
    const auto n_tokens = q->ne[1];
    const auto n_head = q->ne[2];
    // const auto n_kv     = k->ne[1]; // for flash attention

    ggml_tensor *kq = ggml_mul_mat(ctx0, k, q);
    // F32 may not needed for vision encoders?
    // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

    ggml_tensor *kqv = ggml_mul_mat(ctx0, v, kq);
    cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * n_head, n_tokens);
  }

  cb(ctx0, cur, "kqv_out", il);

  if (wo) {
    cur = ggml_mul_mat(ctx0, wo, cur);
  }

  if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
  }

  return cur;
}

ggml_tensor *build_mish(ggml_context *ctx0, ggml_tensor *inp,
                        ggml_tensor *one) {
  // softplus(x)=max(x,0)+log(1+exp(−∣x∣))
  // mish(x) = x * tanh(softplus(x))

  ggml_tensor *softplus_x_stable = ggml_add(
      ctx0, ggml_relu(ctx0, inp),
      ggml_log(ctx0,
               ggml_add(ctx0,
                        ggml_exp(ctx0, ggml_neg(ctx0, ggml_abs(ctx0, inp))),
                        one)));
  ggml_tensor *cur = ggml_mul(ctx0, ggml_tanh(ctx0, softplus_x_stable), inp);
  return cur;
}

ggml_tensor *build_conv1d(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int s, int p, int d) {
  // support default padding
  if (p < 0) {
    p = (w->ne[0] - 1) / 2;
  }
  // ggml_conv_1d uses im2col with a hardcoded GGML_TYPE_F16 kernel type.
  // If the GGUF stored the weight in F32 (newer decoder exports do),
  // calling ggml_conv_1d trips an F16 assertion on the CPU fallback path
  // (CANN falls back for some conv ops). Cast the weight to F16 first.
  // `build_conv1d_f32` would be more precise but in practice produces
  // silent output on this decoder graph — needs deeper investigation.
  if (w->type != GGML_TYPE_F16) {
    w = ggml_cast(ctx0, w, GGML_TYPE_F16);
  }
  cur = ggml_conv_1d(ctx0, w, cur, s, p, d);
  if (b) {
    if (cur->ne[0] != b->ne[0]) {
      b = ggml_reshape_2d(ctx0, b, 1, b->ne[0]);
    }
    cur = ggml_add(ctx0, cur, b);
  }
  return cur;
}

// F32 variant of conv1d: uses GGML_TYPE_F32 im2col to avoid f16 precision loss.
// ggml_conv_1d hardcodes GGML_TYPE_F16 for im2col output; this version overrides that.
ggml_tensor *build_conv1d_f32(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                               ggml_tensor *b, int s, int p, int d) {
  if (p < 0) {
    p = (w->ne[0] - 1) / 2;
  }
  // Replicate ggml_conv_1d logic but with GGML_TYPE_F32 im2col
  ggml_tensor *im2col = ggml_im2col(ctx0, w, cur, s, 0, p, 0, d, 0, false, GGML_TYPE_F32);
  ggml_tensor *result = ggml_mul_mat(ctx0,
      ggml_reshape_2d(ctx0, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
      ggml_reshape_2d(ctx0, w, w->ne[0] * w->ne[1], w->ne[2]));
  result = ggml_reshape_3d(ctx0, result, im2col->ne[1], w->ne[2], im2col->ne[2]);
  if (b) {
    if (result->ne[0] != b->ne[0]) {
      b = ggml_reshape_2d(ctx0, b, 1, b->ne[0]);
    }
    result = ggml_add(ctx0, result, b);
  }
  return result;
}

ggml_tensor *build_conv1d_grouped(ggml_context *ctx0, ggml_tensor *cur,
                                  ggml_tensor *w, ggml_tensor *b, int groups,
                                  int s, int p, int d) {
  // cur shape: (seq_len, in_channels) or (seq_len, in_channels, batch)
  // w shape: (kernel_size, in_channels/groups, out_channels)
  // For Conv1d(768, 768, kernel=128, groups=16):
  //   w shape should be (128, 48, 768) where 48 = 768/16

  if (groups == 1) {
    return build_conv1d(ctx0, cur, w, b, s, p, d);
  }

  int64_t seq_len = cur->ne[0];
  int64_t in_channels = cur->ne[1];
  int64_t batch = cur->ne[2] > 0 ? cur->ne[2] : 1;

  int64_t kernel_size = w->ne[0];
  int64_t out_channels = w->ne[2];

  int64_t in_channels_per_group = in_channels / groups;
  int64_t out_channels_per_group = out_channels / groups;

  // support default padding
  if (p < 0) {
    p = (kernel_size - 1) / 2;
  }

  ggml_tensor *result = nullptr;

  for (int g = 0; g < groups; g++) {
    // Extract input for this group: (seq_len, in_channels_per_group)
    ggml_tensor *inp_g =
        ggml_view_3d(ctx0, cur, seq_len, in_channels_per_group, batch,
                     cur->nb[1], cur->nb[2],
                     g * in_channels_per_group * cur->nb[1]);
    inp_g = ggml_cont(ctx0, inp_g);

    // Extract weight for this group: (kernel_size, in_channels_per_group,
    // out_channels_per_group)
    ggml_tensor *w_g = ggml_view_3d(
        ctx0, w, kernel_size, in_channels_per_group, out_channels_per_group,
        w->nb[1], w->nb[2], g * out_channels_per_group * w->nb[2]);
    w_g = ggml_cont(ctx0, w_g);

    // Apply conv1d for this group
    ggml_tensor *out_g = ggml_conv_1d(ctx0, w_g, inp_g, s, p, d);

    // Concatenate results along channel dimension
    if (result == nullptr) {
      result = out_g;
    } else {
      result = ggml_concat(ctx0, result, out_g, 1);
    }
  }

  // Add bias if present
  if (b) {
    if (result->ne[0] != b->ne[0]) {
      b = ggml_reshape_2d(ctx0, b, 1, b->ne[0]);
    }
    result = ggml_add(ctx0, result, b);
  }

  return result;
}

// TODO: to implement
ggml_tensor *build_flip(ggml_context *ctx0, ggml_tensor *cur, int ic) {
  return cur;
}
// TODO: to implement
ggml_tensor *build_conv_transpose1d(ggml_context *ctx0, ggml_tensor *cur,
                                    ggml_tensor *w, ggml_tensor *b, int s,
                                    int p, int d) {
  cur = ggml_conv_transpose_1d(ctx0, w, cur, s, p, d);
  ggml_tensor *transposed_b = ggml_permute(ctx0, b, 1, 0, 2, 3);
  transposed_b = ggml_cont(ctx0, transposed_b);
  if (b) {
    cur = ggml_add(ctx0, cur, transposed_b);
  }
  return cur;
}