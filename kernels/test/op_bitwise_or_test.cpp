/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

class OpBitwiseOrTensorOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_or_tensor_out(
      const Tensor& self,
      const Tensor& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_or_outf(context_, self, other, out);
  }
};

class OpBitwiseOrScalarOutTest : public OperatorTest {
 protected:
  Tensor& op_bitwise_or_scalar_out(
      const Tensor& self,
      const Scalar& other,
      Tensor& out) {
    return torch::executor::aten::bitwise_or_outf(context_, self, other, out);
  }
};

TEST_F(OpBitwiseOrTensorOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({2, 2}, {2, 3, 2, 5});
  Tensor b = tf.make({2, 2}, {1, 6, 2, 3});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_or_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {3, 7, 2, 7}));
}

TEST_F(OpBitwiseOrTensorOutTest, SmokeTestBool) {
  TensorFactory<ScalarType::Bool> tf;

  Tensor a = tf.make({2, 2}, {true, false, true, false});
  Tensor b = tf.make({2, 2}, {true, true, false, false});

  Tensor out = tf.zeros({2, 2});

  op_bitwise_or_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {true, true, true, false}));
}

TEST_F(OpBitwiseOrTensorOutTest, SmokeTestMixed) {
  TensorFactory<ScalarType::Int> tfInt;
  TensorFactory<ScalarType::Bool> tfBool;

  Tensor a = tfInt.make({2, 2}, {2, 3, 2, 5});
  Tensor b = tfBool.make({2, 2}, {true, true, false, false});

  Tensor out = tfInt.zeros({2, 2});

  op_bitwise_or_tensor_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tfInt.make({2, 2}, {3, 3, 2, 5}));
}

TEST_F(OpBitwiseOrScalarOutTest, SmokeTestInt) {
  TensorFactory<ScalarType::Int> tf;

  Tensor a = tf.make({2, 2}, {2, 3, 2, 5});
  Scalar b = 6;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_or_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {6, 7, 6, 7}));
}

TEST_F(OpBitwiseOrScalarOutTest, SmokeTestBool) {
  TensorFactory<ScalarType::Bool> tf;

  Tensor a = tf.make({2, 2}, {true, false, true, false});
  Scalar b = true;

  Tensor out = tf.zeros({2, 2});

  op_bitwise_or_scalar_out(a, b, out);
  EXPECT_TENSOR_EQ(out, tf.make({2, 2}, {true, true, true, true}));
}
