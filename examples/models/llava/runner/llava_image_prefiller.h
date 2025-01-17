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
private:
    static constexpr size_t MEMORY_PAGE_SIZE = 4096;
    static constexpr size_t BLOCK_SIZE = 64 * 1024 * 1024; // 64MB
    static constexpr size_t ALIGNMENT = 64; // Ensure 64-byte alignment
    
    // Memory blocks with alignment
    alignas(ALIGNMENT) std::vector<uint8_t> image_block_;
    alignas(ALIGNMENT) std::vector<uint8_t> encoder_block_;
    alignas(ALIGNMENT) std::vector<int64_t> start_pos_block_;
    
    // Strong references to tensors and outputs
    std::vector<executorch::runtime::EValue> encoder_outputs_;
    std::vector<executorch::runtime::EValue> text_model_outputs_;
    executorch::extension::TensorPtr image_tensor_ptr_;
    executorch::extension::TensorPtr start_pos_tensor_ptr_;
    executorch::extension::TensorPtr encoder_output_tensor_ptr_; // New: Keep encoder output tensor

public:
    LlavaImagePrefiller(::executorch::extension::Module* module)
        : ImagePrefiller(module) {
        // Pre-allocate with alignment
        image_block_.resize(((512 * 1024 + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT);
        encoder_block_.resize(((BLOCK_SIZE + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT);
        start_pos_block_.resize(((16 * sizeof(int64_t) + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT / sizeof(int64_t));
        
        // Zero initialize
        std::memset(image_block_.data(), 0, image_block_.size());
        std::memset(encoder_block_.data(), 0, encoder_block_.size());
        std::memset(start_pos_block_.data(), 0, start_pos_block_.size() * sizeof(int64_t));
    }

    /**
     * Prefill an LLM Module with the given image input.
     * @param image The image input to LLaVa.
     * @param start_pos The starting position in KV cache of the input in the LLM
     * @return logits of the image prefill.
     */
    inline ::executorch::runtime::Result<exec_aten::Tensor> prefill(
        ::executorch::extension::llm::Image& image,
        int64_t& start_pos) override {
        
        // Clear previous state
        encoder_outputs_.clear();
        text_model_outputs_.clear();
        image_tensor_ptr_.reset();
        start_pos_tensor_ptr_.reset();
        encoder_output_tensor_ptr_.reset();
        
        // Memory barrier
        std::atomic_thread_fence(std::memory_order_release);
        
        // Create image tensor
        size_t image_size = image.width * image.height * image.channels;
        ET_LOG(Info, "Creating image tensor with dimensions: [%d, %d, %d]", 
               image.channels, image.height, image.width);
        
        std::memcpy(image_block_.data(), image.data.data(), image_size);
        image_tensor_ptr_ = executorch::extension::from_blob(
            image_block_.data(),
            {image.channels, image.height, image.width},
            ::executorch::aten::ScalarType::Byte);
        
        ET_LOG(Info, "Image tensor address: %p", image_tensor_ptr_.get());
        ET_LOG(Info, "Image tensor size: %zu", image_size);
        
        // Run image encoder
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto encoder_result = ET_UNWRAP(module_->execute(kImageEncoderMethod, *image_tensor_ptr_));
        encoder_outputs_ = std::move(encoder_result);
        
        // Create a copy of encoder output in our aligned buffer
        const auto& encoder_tensor = encoder_outputs_[0].toTensor();
        size_t encoder_size = encoder_tensor.numel() * sizeof(float);
        std::memcpy(encoder_block_.data(), encoder_tensor.data_ptr(), encoder_size);
        
        // Create new tensor from our aligned buffer
        const auto& sizes = encoder_tensor.sizes();
        std::vector<executorch::aten::SizesType> sizes_vec(sizes.begin(), sizes.end());
        encoder_output_tensor_ptr_ = executorch::extension::from_blob(
            encoder_block_.data(),
            sizes_vec,
            encoder_tensor.scalar_type());
        
        ET_LOG(Info, "Image encoder outputs address: %p", encoder_block_.data());
        ET_LOG(Info, "Encoder output tensor size: %zu", encoder_tensor.numel());
        
        // Create start position tensor
        start_pos_block_[0] = start_pos;
        start_pos_tensor_ptr_ = executorch::extension::from_blob(
            start_pos_block_.data(),
            {1},
            ::executorch::aten::ScalarType::Long);
            
        ET_LOG(Info, "Start pos tensor address: %p", start_pos_tensor_ptr_.get());
        
        // Run text model with our aligned tensors
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto outputs_res = ET_UNWRAP(module_->execute(
            kTextModelMethod,
            {*start_pos_tensor_ptr_, *encoder_output_tensor_ptr_}));
        
        text_model_outputs_ = std::move(outputs_res);
        
        ET_LOG(Info, "Text model outputs address: %p", text_model_outputs_.data());
        
        ET_CHECK_MSG(
            text_model_outputs_[0].isTensor(),
            "Non Tensor Output returned from executing image prefill");

        // Update start position
        start_pos += encoder_tensor.size(1);
        
        ET_LOG(Info, "Updated start pos value: %ld", start_pos);
        return text_model_outputs_[0].toTensor();
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
