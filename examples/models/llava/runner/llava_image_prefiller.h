/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given a image tensor, prefill the KV cache of LLaVA.

#pragma once

#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/tensor/tensor.h>

namespace example {

class ET_EXPERIMENTAL LlavaImagePrefiller
    : public ::executorch::extension::llm::ImagePrefiller {
 public:
  LlavaImagePrefiller(::executorch::extension::Module* module)
      : ImagePrefiller(module){};
  /**
   * Prefill an LLM Module with the given image input.
   * @param image The image input to LLaVa.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * @return logits of the image prefill.
   */
  inline ::executorch::runtime::Result<exec_aten::Tensor> prefill(
      ::executorch::extension::llm::Image& image,
      int64_t& start_pos) override {
    // First verify methods are loaded
    ET_CHECK_MSG(
        module_->is_method_loaded(kImageEncoderMethod),
        "Image encoder method not loaded");
    ET_CHECK_MSG(
        module_->is_method_loaded(kTextModelMethod),
        "Text model method not loaded");

    // Log available methods for debugging
    auto methods_res = module_->method_names();
    if (methods_res.ok()) {
        for (const auto& method : methods_res.get()) {
            ET_LOG(Info, "Available method: %s", method.c_str());
        }
    }

    // Create image tensor with proper ownership
    std::vector<uint8_t> image_data_copy(image.data);
    auto image_tensor = executorch::extension::from_blob(
        image_data_copy.data(),
        {3, image.height, image.width},
        ::executorch::aten::ScalarType::Byte);

    ET_LOG(Info, "Executing image encoder");
    auto image_encoder_outputs =
        ET_UNWRAP(module_->execute(kImageEncoderMethod, image_tensor));
    ET_LOG(Info, "Image encoder execution complete");

    ET_CHECK_MSG(
        !image_encoder_outputs.empty() && image_encoder_outputs[0].isTensor(),
        "Invalid output from image encoder");
        
    // Log image encoder output shape
    auto& image_embeds = image_encoder_outputs[0].toTensor();
    ET_LOG(Info, "Image encoder output shape: [%ld, %ld, %ld]", 
           image_embeds.size(0), image_embeds.size(1), image_embeds.size(2));

    // Get embedding length and validate against cache size
    int64_t embedding_length = image_embeds.size(1);
    constexpr int64_t MAX_CACHE_SIZE = 768;  // From the error message
    
    ET_CHECK_MSG(
        embedding_length < MAX_CACHE_SIZE,
        "Image embedding length (%ld) exceeds max cache size (%ld)",
        embedding_length, MAX_CACHE_SIZE);
    
    ET_LOG(Info, "Image embedding length: %ld", embedding_length);
    
    // Always start from position 0 for text model to process the entire sequence
    int64_t text_start_pos = 0;
    ET_LOG(Info, "Text model will start at position: %ld", text_start_pos);

    // Create start_pos tensor with position 0
    std::vector<int64_t> start_pos_data = {text_start_pos};
    auto start_pos_tensor = executorch::extension::from_blob(
        start_pos_data.data(),
        {1},
        ::executorch::aten::ScalarType::Long);

    ET_LOG(Info, "Executing text model with start_pos: %ld", text_start_pos);
    auto outputs_res = ET_UNWRAP(module_->execute(
        kTextModelMethod, {start_pos_tensor, image_encoder_outputs[0]}));
    ET_LOG(Info, "Text model execution complete");

    ET_CHECK_MSG(
        outputs_res[0].isTensor(),
        "Non Tensor Output returned from executing image prefill");

    // Update start_pos to the length of processed embeddings
    start_pos = embedding_length;
    ET_LOG(Info, "Updated start_pos to: %ld", start_pos);

    return outputs_res[0].toTensor();
  }

  /**
   * Load the Module for image prefill purpose.
   * @return The error code.
   */
  inline ::executorch::runtime::Error load() override {
    if (is_method_loaded()) {
      return ::executorch::runtime::Error::Ok;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kImageEncoderMethod));
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(kTextModelMethod));
    return ::executorch::runtime::Error::Ok;
  }

  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  inline bool is_method_loaded() override {
    ::executorch::runtime::Result<std::unordered_set<std::string>> methods_res =
        module_->method_names();
    if (methods_res.error() != ::executorch::runtime::Error::Ok) {
      ET_CHECK_MSG(false, "Failed to get method names");
    }
    std::unordered_set<std::string> methods = methods_res.get();
    bool methods_exist = methods.find(kImageEncoderMethod) != methods.end() &&
        methods.find(kTextModelMethod) != methods.end();
    if (!methods_exist) {
      for (const auto& method : methods) {
        ET_LOG(Error, "Method: %s", method.c_str());
      }
      ET_CHECK_MSG(
          methods_exist,
          "Missing required methods (%s, %s) in the model",
          kImageEncoderMethod.c_str(),
          kTextModelMethod.c_str());
    }
    bool methods_loaded = module_->is_method_loaded(kImageEncoderMethod) &&
        module_->is_method_loaded(kTextModelMethod);
    return methods_loaded;
  }

  inline static const std::string kImageEncoderMethod = "image_encoder";
  inline static const std::string kTextModelMethod = "text_model";
};

} // namespace example
