#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

BUILD_TOOL=$1
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
else
  echo "Setup Linux for ${BUILD_TOOL} ..."
fi

# As Linux job is running inside a Docker container, all of its dependencies
# have already been installed, so we use PyTorch build from source here instead
# of nightly. This allows CI to test against latest commits from PyTorch
install_executorch "use-pt-pinned-commit"
build_executorch_runner "${BUILD_TOOL}"
do_not_use_nightly_on_ci
