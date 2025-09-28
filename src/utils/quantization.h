#pragma once

#include <vector>
#include <cstdint>

namespace edge_ai {

class Quantizer {
public:
    static std::vector<int8_t> quantize_float32(const std::vector<float>& data,
                                               float scale, int zero_point);
    static std::vector<float> dequantize_int8(const std::vector<int8_t>& data,
                                             float scale, int zero_point);
    static std::pair<float, int> calculate_quantization_params(const std::vector<float>& data);
    static std::vector<float> calculate_scale_and_zero_point(const std::vector<float>& data,
                                                           int qmin = -128, int qmax = 127);
    static void quantize_in_place(std::vector<float>& data, float scale, int zero_point);
    static void dequantize_in_place(std::vector<int8_t>& data, float scale, int zero_point);
    static int8_t float_to_int8(float value, float scale, int zero_point);
    static float int8_to_float(int8_t value, float scale, int zero_point);
    static void prune_weights(std::vector<float>& weights, float threshold);
    static void quantize_model_weights(std::vector<float>& weights, int8_t& quantized_weights,
                                      float& scale, int& zero_point);
    static void dequantize_model_weights(const std::vector<int8_t>& quantized_weights,
                                        std::vector<float>& weights, float scale, int zero_point);
};

}