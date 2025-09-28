#pragma once

#include "../include/model_base.h"
#include <vector>
#include <memory>

namespace edge_ai {

struct Conv2DConfig {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
};

struct DenseConfig {
    int input_size;
    int output_size;
};

struct FlattenConfig {
    int input_size;
};

struct CNNConfig {
    std::vector<Conv2DConfig> conv_layers;
    std::vector<FlattenConfig> flatten_layers;
    std::vector<DenseConfig> dense_layers;
};

struct CNNLayer {
    enum class Type {
        CONV2D,
        DENSE,
        FLATTEN,
        RELU,
        MAX_POOL2D,
        SOFTMAX
    };

    Type type;
    union {
        Conv2DConfig conv_config;
        DenseConfig dense_config;
        FlattenConfig flatten_config;
    } config;
};

class CNN : public BaseModel {
public:
    CNN(const CNNConfig& config);
    ~CNN() = default;

    bool train(const Dataset& data, const TrainingConfig& config = {}) override;
    std::vector<int> predict(const std::vector<std::vector<float>>& data) override;
    bool save_model(const std::string& filepath) override;
    bool load_model(const std::string& filepath) override;

    std::vector<float> get_weights() const override;
    void set_weights(const std::vector<float>& weights) override;

private:
    CNNConfig config_;
    std::vector<std::vector<float>> weights_;
    std::vector<std::vector<float>> biases_;
    std::vector<std::vector<std::vector<float>>> training_data_;
    std::vector<int> training_labels_;

    std::vector<std::vector<std::vector<float>>> weights_3d_;
    std::vector<std::vector<std::vector<float>>> biases_3d_;

    int input_height_;
    int input_width_;
    int input_channels_;

    void initialize_parameters();
    std::vector<std::vector<std::vector<float>>>
        conv2d_forward(const std::vector<std::vector<std::vector<float>>& input,
                      const std::vector<std::vector<std::vector<float>>& kernel,
                      const std::vector<std::vector<float>>& bias,
                      int stride, int padding);
    std::vector<std::vector<float>>
        max_pool2d_forward(const std::vector<std::vector<std::vector<float>>& input,
                         int pool_size, int stride);
    std::vector<std::vector<float>>
        dense_forward(const std::vector<std::vector<float>>& input,
                     const std::vector<std::vector<float>>& weights,
                     const std::vector<float>& bias);
    std::vector<std::vector<std::vector<float>>>
        conv2d_backward(const std::vector<std::vector<std::vector<float>>& input,
                       const std::vector<std::vector<std::vector<float>>& kernel,
                       int stride, int padding);
    std::vector<std::vector<float>>
        max_pool2d_backward(const std::vector<std::vector<std::vector<float>>& input,
                          const std::vector<std::vector<float>>& mask,
                          int pool_size, int stride);
    std::vector<std::vector<float>>
        dense_backward(const std::vector<std::vector<float>>& input,
                      const std::vector<std::vector<float>>& weights);
    void update_parameters_conv2d(const std::vector<std::vector<std::vector<float>>& grad_weights,
                                const std::vector<std::vector<float>>& grad_bias,
                                const TrainingConfig& config);
    void update_parameters_dense(const std::vector<std::vector<float>>& grad_weights,
                               const std::vector<float>& grad_bias,
                               const TrainingConfig& config);
    float calculate_loss(const std::vector<std::vector<float>>& predictions,
                        const std::vector<int>& labels);
    std::vector<int> argmax_2d(const std::vector<std::vector<float>>& predictions);
};

}