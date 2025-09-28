#include "model_base.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace edge_ai {

BaseModel::BaseModel(ModelType type) : model_type_(type) {}

float BaseModel::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float BaseModel::relu(float x) {
    return std::max(0.0f, x);
}

float BaseModel::relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float BaseModel::softmax(const std::vector<float>& x, int index) {
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (float val : x) {
        sum += std::exp(val - max_val);
    }

    return std::exp(x[index] - max_val) / sum;
}

std::vector<int> BaseModel::argmax(const std::vector<std::vector<float>>& predictions) {
    std::vector<int> results;
    results.reserve(predictions.size());

    for (const auto& row : predictions) {
        int max_idx = 0;
        float max_val = row[0];
        for (size_t i = 1; i < row.size(); ++i) {
            if (row[i] > max_val) {
                max_val = row[i];
                max_idx = static_cast<int>(i);
            }
        }
        results.push_back(max_idx);
    }

    return results;
}

float BaseModel::calculate_accuracy(const std::vector<int>& true_labels,
                                  const std::vector<int>& predicted_labels) {
    if (true_labels.size() != predicted_labels.size()) {
        return 0.0f;
    }

    int correct = 0;
    for (size_t i = 0; i < true_labels.size(); ++i) {
        if (true_labels[i] == predicted_labels[i]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / true_labels.size();
}

void BaseModel::quantize(const QuantizationConfig& config) {
    quantized_ = true;
    quant_config_ = config;
}

void BaseModel::dequantize() {
    quantized_ = false;
}

void BaseModel::quantize_model() {
    if (quantized_) return;

    original_weights_ = get_weights();
    quantize(quant_config_);
}

void BaseModel::dequantize_model() {
    if (!quantized_) return;

    set_weights(original_weights_);
    quantized_ = false;
}

}