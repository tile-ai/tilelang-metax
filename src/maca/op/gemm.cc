/*!
 * \file tl/backend/maca/op/gemm.cc
 * \brief MACA implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "maca/target_utils.h"
#include "span_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

namespace maca {

namespace {

constexpr const char *kMacaMMA = "maca.mma";

bool CheckWgmma(const GemmNode &op) {
  if (op.b_.scope() != "shared.dyn" && op.b_.scope() != "shared") {
    return false;
  }

  if (op.c_->dtype == DataType::Float(16)) {
    if (op.a_->dtype == DataType::Float(16) &&
        op.b_->dtype == DataType::Float(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  if (op.c_->dtype == DataType::Float(32)) {
    if (op.a_->dtype == DataType::Float(16) &&
        op.b_->dtype == DataType::Float(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype == DataType::BFloat(16) &&
        op.b_->dtype == DataType::BFloat(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype == DataType::Float(32) &&
        op.b_->dtype == DataType::Float(32))
      return (!op.transA_) && op.transB_ && op.k_ % 8 == 0;
    if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  if (op.c_->dtype == DataType::Int(32)) {
    if (op.a_->dtype == DataType::Int(8) && op.b_->dtype == DataType::Int(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::Int(8) && op.b_->dtype == DataType::UInt(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::UInt(8) && op.b_->dtype == DataType::Int(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::UInt(8) && op.b_->dtype == DataType::UInt(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  return false;
}

bool AllowMacaMMA(const GemmNode &op) {
  const DataType &a = op.a_->dtype;
  const DataType &b = op.b_->dtype;
  const DataType &c = op.c_->dtype;

  if (a != b) {
    return false;
  }

  if (a == DataType::Float(16)) {
    return c == DataType::Float(16) || c == DataType::Float(32);
  }

  if (a.is_float8_e4m3() || a.is_float8_e4m3fn() || a.is_float8_e5m2()) {
    return c == DataType::Float(32);
  }

  if ((a.is_int() || a.is_uint()) && a.bits() == 8) {
    return c == DataType::Int(32);
  }

  if (a.is_tfloat32() && b.is_tfloat32()) {
    return c == DataType::Float(32);
  }

  if (a == DataType::BFloat(16)) {
    return c == DataType::Float(32);
  }

  if (a == DataType::Float(32)) {
    return c == DataType::Float(32);
  }

  if (a == DataType::Float(64)) {
    return c == DataType::Float(64);
  }

  return false;
}

// TODO: may be used for fp8
void FatalMacaFp8MmaUnavailable(const GemmNode &op, Target target) {
  LOG(FATAL)
      << "T.gemm() with fp8 operands requires a MACA target with fp8 "
         "mma.sync atoms , but got target="
      << target << " with A(dtype=" << op.a_->dtype
      << "), B(dtype=" << op.b_->dtype
      << "). MACA arch() have no .e4m3/.e5m2 mma instruction; the kernel "
         "would compile and then trap at launch."
      << SpanHintSuffix({op.a_->span, op.b_->span, op.c_->span});
}

void FatalMacaMmaUnavailable(const GemmNode &op, Target target) {
  LOG(FATAL)
      << "T.gemm requires native mma.sync lowering on MACA, "
         "but the operand dtype configuration is not supported. Got target="
      << target << ", A(scope=" << op.a_.scope() << ", dtype=" << op.a_->dtype
      << "), B(scope=" << op.b_.scope() << ", dtype=" << op.b_->dtype
      << "), C(scope=" << op.c_.scope() << ", dtype=" << op.c_->dtype
      << "), M=" << op.m_ << ", N=" << op.n_ << ", K=" << op.k_;
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  auto is_valid = [&](int m, int n) {
    return m * n == num_warps && M % (m * kMPerWarp) == 0 &&
           N % (n * k_n_per_warp) == 0;
  };

  bool found = false;
  if (policy.IsFullRow()) {
    for (int m = num_warps; m >= 1; m--) {
      if (num_warps % m != 0 || !is_valid(m, num_warps / m))
        continue;
      m_warp = m;
      n_warp = num_warps / m;
      found = true;
      break;
    }
  } else if (policy.IsFullCol()) {
    for (int n = num_warps; n >= 1; n--) {
      if (num_warps % n != 0 || !is_valid(num_warps / n, n))
        continue;
      n_warp = n;
      m_warp = num_warps / n;
      found = true;
      break;
    }
  } else if (policy.IsSquare()) {
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= num_warps; m++) {
      if (num_warps % m != 0)
        continue;
      int n = num_warps / m;
      if (!is_valid(m, n))
        continue;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        m_warp = m;
        n_warp = n;
        found = true;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  if (!found) {
    LOG(FATAL) << "No valid warp partition for T.gemm: M=" << M << ", N=" << N
               << " cannot be evenly covered by " << num_warps
               << " warps (policy="
               << (policy.IsFullRow()   ? "FullRow"
                   : policy.IsFullCol() ? "FullCol"
                                        : "Square")
               << "). Each warp must own a multiple of " << kMPerWarp
               << " rows and " << k_n_per_warp
               << " columns; adjust `threads` or the block tile shape.";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (!AllowMacaMMA(op)) {
      FatalMacaMmaUnavailable(op, target);
    }
    return kMacaMMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetMacaGetWarpSize(target);
    int k_n_per_warp = 16;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kMacaMMA;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kMacaMMA) {
      return "mma";
    }
    return "unknown";
  }
};

} // namespace maca

namespace {

bool MatchMacaGemmTarget(Target target) {
  return TargetIsMaca(target) || TargetIsCuTeDSL(target);
}

bool RegisterMacaGemm() {
  RegisterGemmImpl(GemmImpl{
      "maca.Gemm",
      MatchMacaGemmTarget,
      maca::Gemm::SelectInst,
      maca::Gemm::ComputeWarpPartition,
      maca::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool maca_gemm_registered = RegisterMacaGemm();

} // namespace

} // namespace tl
} // namespace tvm
