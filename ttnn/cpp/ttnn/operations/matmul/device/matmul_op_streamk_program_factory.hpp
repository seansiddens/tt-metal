// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/matmul_op.hpp"

namespace ttnn::operations::matmul {

/**
 * StreamK GEMM Implementation
 */
tt::tt_metal::operation::ProgramWithCallbacks matmul_streamk(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    Tensor& output_tensor,
    const MatmulStreamKProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::matmul
