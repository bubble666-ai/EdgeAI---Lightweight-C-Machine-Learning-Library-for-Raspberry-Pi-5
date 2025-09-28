#include "quantization.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace edge_ai {

std::vector<int8_t> Quantizer::quantize_float32(const std::vector<float>& data,
                                               float scale, int zero_point) {
    std::vector<int8_t> quantized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        quantized[i] = float_to_int8(data[i], scale, zero_point);
    }
    return quantized;
}

std::vector<float> Quantizer::dequantize_int8(const std::vector<int8_t>& data,
                                             float scale, int zero_point) {
    std::vector<float> dequantized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        dequantized[i] = int8_to_float(data[i], scale, zero_point);
    }
    return dequantized;
}

std::pair<float, int> Quantizer::calculate_quantization_params(const std::vector<float>& data) {
    auto [scale, zero_point] = calculate_scale_and_zero_point(data);
    return {scale, zero_point};
}

std::pair<float, int> Quantizer::calculate_scale_and_zero_point(const std::vector<float>& data,
                                                               int qmin, int qmax) {
    if (data.empty()) {
        return {1.0f, 0};
    }

    float min_val = *std::min_element(data.begin(), data.end());
    float max_val = *std::max_element(data.begin(), data.end());

    float qfmin = static_cast<float>(qmin);
    float qfmax = static_cast<float>(qmax);

    float scale = (max_val - min_val) / (qfmax - qfmin);
    float zero_point = qfmin - min_val / scale;

    if (zero_point < qmin) {
        zero_point = qmin;
    } else if (zero_point > qmax) {
        zero_point = qmax;
    }

    return {scale, static_cast<int>(std::round(zero_point))};
}

void Quantizer::quantize_in_place(std::vector<float>& data, float scale, int zero_point) {
    for (auto& val : data) {
        val = float_to_int8(val, scale, zero_point);
    }
}

void Quantizer::dequantize_in_place(std::vector<int8_t>& data, float scale, int zero_point) {
    for (auto& val : data) {
        val = int8_to_float(val, scale, zero_point);
    }
}

int8_t Quantizer::float_to_int8(float value, float scale, int zero_point) {
    if (scale == 0.0f) {
        return static_cast<int8_t>(zero_point);
    }

    float qvalue = value * scale + zero_point;
    qvalue = std::max(-128.0f, std::min(127.0f, qvalue));
    return static_cast<int8_t>(std::round(qvalue));
}

float Quantizer::int8_to_float(int8_t value, float scale, int zero_point) {
    if (scale == 0.0f) {
        return 0.0f;
    }
    return (static_cast<float>(value) - zero_point) / scale;
}

void Quantizer::prune_weights(std::vector<float>& weights, float threshold) {
    for (auto& weight : weights) {
        if (std::abs(weight) < threshold) {
            weight = 0.0f;
        }
    }
}

void Quantizer::quantize_model_weights(std::vector<float>& weights, std::vector<int8_t>& quantized_weights,
                                      float& scale, int& zero_point) {
    auto params = calculate_quantization_params(weights);
    scale = params.first;
    zero_point = params.second;
    quantized_weights = quantize_float32(weights, scale, zero_point);
}

void Quantizer::dequantize_model_weights(const std::vector<int8_t>& quantized_weights,
                                        std::vector<float>& weights, float scale, int zero_point) {
    weights = dequantize_int8(quantized_weights, scale, zero_point);
}

}