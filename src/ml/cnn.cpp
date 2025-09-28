#include "cnn.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>

namespace edge_ai {

CNN::CNN(const CNNConfig& config)
    : BaseModel(ModelType::CNN)
    , config_(config) {
    initialize_parameters();
}

void CNN::initialize_parameters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for (const auto& conv_config : config_.conv_layers) {
        std::vector<std::vector<std::vector<float>>> kernel(conv_config.out_channels,
                                                          std::vector<std::vector<float>>(
                                                              conv_config.kernel_size,
                                                              std::vector<float>(conv_config.kernel_size)));

        for (int oc = 0; oc < conv_config.out_channels; ++oc) {
            for (int kh = 0; kh < conv_config.kernel_size; ++kh) {
                for (int kw = 0; kw < conv_config.kernel_size; ++kw) {
                    kernel[oc][kh][kw] = dist(gen);
                }
            }
        }

        std::vector<std::vector<float>> bias(conv_config.out_channels,
                                          std::vector<float>(conv_config.out_channels, 0.0f));

        weights_3d_.push_back(kernel);
        biases_3d_.push_back(bias);
    }

    for (const auto& dense_config : config_.dense_layers) {
        std::vector<std::vector<float>> weights(dense_config.output_size,
                                               std::vector<float>(dense_config.input_size));

        for (int i = 0; i < dense_config.output_size; ++i) {
            for (int j = 0; j < dense_config.input_size; ++j) {
                weights[i][j] = dist(gen);
            }
        }

        std::vector<float> bias(dense_config.output_size, 0.0f);

        weights_.push_back(weights);
        biases_.push_back(bias);
    }
}

bool CNN::train(const Dataset& data, const TrainingConfig& config) {
    if (data.num_samples == 0 || data.num_features == 0) {
        return false;
    }

    input_height_ = 32;
    input_width_ = 32;
    input_channels_ = 3;

    training_data_.resize(data.num_samples);
    training_labels_.resize(data.num_samples);

    for (int i = 0; i < data.num_samples; ++i) {
        training_data_[i] = std::vector<std::vector<float>>(
            input_channels_,
            std::vector<float>(input_height_ * input_width_));
        training_labels_[i] = data.y[i];

        for (int c = 0; c < input_channels_; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int idx = c * input_height_ * input_width_ + h * input_width_ + w;
                    if (idx < data.X[i].size()) {
                        training_data_[i][c][h * input_width_ + w] = data.X[i][idx];
                    }
                }
            }
        }
    }

    for (int epoch = 0; epoch < config.max_epochs; ++epoch) {
        float loss = 0.0f;

        for (int i = 0; i < training_data_.size(); ++i) {
            auto conv1_output = conv2d_forward(training_data_[i], weights_3d_[0], biases_3d_[0],
                                             config_.conv_layers[0].stride, config_.conv_layers[0].padding);

            auto relu1_output = conv1_output;
            for (auto& channel : relu1_output) {
                for (auto& val : channel) {
                    val = relu(val);
                }
            }

            auto pool1_output = max_pool2d_forward(relu1_output, 2, 2);

            auto flatten1_output = std::vector<std::vector<float>>(1);
            int flatten_size = 0;
            for (const auto& channel : pool1_output) {
                flatten_size += channel.size();
            }
            flatten1_output[0].resize(flatten_size);

            int pos = 0;
            for (const auto& channel : pool1_output) {
                for (float val : channel) {
                    flatten1_output[0][pos++] = val;
                }
            }

            auto dense1_output = dense_forward(flatten1_output, weights_[0], biases_[0]);

            auto relu2_output = dense1_output;
            for (auto& val : relu2_output[0]) {
                val = relu(val);
            }

            auto dense2_output = dense_forward(relu2_output, weights_[1], biases_[1]);

            auto softmax_output = dense2_output;
            float max_val = *std::max_element(softmax_output[0].begin(), softmax_output[0].end());
            float sum = 0.0f;

            for (auto& val : softmax_output[0]) {
                val = std::exp(val - max_val);
                sum += val;
            }

            for (auto& val : softmax_output[0]) {
                val /= sum;
            }

            loss += -std::log(softmax_output[0][training_labels_[i]] + 1e-15f);

            float target = training_labels_[i];
            std::vector<float> grad_output(softmax_output[0].size(), 0.0f);
            grad_output[target] = 1.0f;

            for (size_t j = 0; j < softmax_output[0].size(); ++j) {
                grad_output[j] -= softmax_output[0][j];
            }

            auto grad_dense2 = dense_backward(relu2_output[0], weights_[1]);
            auto grad_weights2 = std::vector<std::vector<float>>(weights_[1].size(),
                                                                std::vector<float>(weights_[1][0].size(), 0.0f));
            auto grad_bias2 = std::vector<float>(weights_[1].size(), 0.0f);

            for (size_t i = 0; i < grad_weights2.size(); ++i) {
                for (size_t j = 0; j < grad_weights2[0].size(); ++j) {
                    grad_weights2[i][j] = grad_output[i] * relu2_output[0][j];
                }
                grad_bias2[i] = grad_output[i];
            }

            auto grad_relu2 = dense1_output;
            for (size_t i = 0; i < grad_relu2[0].size(); ++i) {
                grad_relu2[0][i] = grad_dense2[i] * (grad_relu2[0][i] > 0 ? 1.0f : 0.0f);
            }

            auto grad_dense1 = dense_backward(flatten1_output[0], weights_[0]);
            auto grad_weights1 = std::vector<std::vector<float>>(weights_[0].size(),
                                                                std::vector<float>(weights_[0][0].size(), 0.0f));
            auto grad_bias1 = std::vector<float>(weights_[0].size(), 0.0f);

            for (size_t i = 0; i < grad_weights1.size(); ++i) {
                for (size_t j = 0; j < grad_weights1[0].size(); ++j) {
                    grad_weights1[i][j] = grad_relu2[0][i] * flatten1_output[0][j];
                }
                grad_bias1[i] = grad_relu2[0][i];
            }

            update_parameters_dense(grad_weights1, grad_bias1, config);
            update_parameters_dense(grad_weights2, grad_bias2, config);
        }

        loss /= training_data_.size();

        if (config.verbose && epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }

    return true;
}

std::vector<std::vector<std::vector<float>>>
CNN::conv2d_forward(const std::vector<std::vector<std::vector<float>>& input,
                    const std::vector<std::vector<std::vector<float>>& kernel,
                    const std::vector<std::vector<float>>& bias,
                    int stride, int padding) {
    int batch_size = input.size();
    int in_channels = input[0].size();
    int in_height = input[0][0].size();
    int in_width = input[0][0].size();
    int out_channels = kernel.size();
    int kernel_size = kernel[0].size();

    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;

    std::vector<std::vector<std::vector<float>>> output(batch_size,
                                                       std::vector<std::vector<float>>(
                                                           out_channels,
                                                           std::vector<float>(out_height * out_width)));

    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    sum += input[b][ic][ih * in_width + iw] * kernel[oc][ic][kh * kernel_size + kw];
                                }
                            }
                        }
                    }
                    output[b][oc][oh * out_width + ow] = sum + bias[oc][0];
                }
            }
        }
    }

    return output;
}

std::vector<std::vector<float>>
CNN::max_pool2d_forward(const std::vector<std::vector<std::vector<float>>& input,
                        int pool_size, int stride) {
    int batch_size = input.size();
    int channels = input[0].size();
    int in_height = input[0][0].size();
    int in_width = static_cast<int>(std::sqrt(in_height));

    int out_height = (in_height - pool_size) / stride + 1;
    int out_width = (in_width - pool_size) / stride + 1;

    std::vector<std::vector<float>> output(batch_size,
                                          std::vector<float>(channels * out_height * out_width));

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::max();
                    for (int ph = 0; ph < pool_size; ++ph) {
                        for (int pw = 0; pw < pool_size; ++pw) {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            if (ih < in_height && iw < in_width) {
                                int idx = ih * in_width + iw;
                                max_val = std::max(max_val, input[b][c][idx]);
                            }
                        }
                    }
                    int out_idx = c * out_height * out_width + oh * out_width + ow;
                    output[b][out_idx] = max_val;
                }
            }
        }
    }

    return output;
}

std::vector<std::vector<float>>
CNN::dense_forward(const std::vector<std::vector<float>>& input,
                   const std::vector<std::vector<float>>& weights,
                   const std::vector<float>& bias) {
    std::vector<std::vector<float>> output(1, std::vector<float>(weights.size(), 0.0f));

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            output[0][i] += input[0][j] * weights[i][j];
        }
        output[0][i] += bias[i];
    }

    return output;
}

std::vector<std::vector<float>>
CNN::conv2d_backward(const std::vector<std::vector<std::vector<float>>& input,
                     const std::vector<std::vector<std::vector<float>>& kernel,
                     int stride, int padding) {
    return std::vector<std::vector<std::vector<float>>>(input.size());
}

std::vector<std::vector<float>>
CNN::max_pool2d_backward(const std::vector<std::vector<std::vector<float>>& input,
                        const std::vector<std::vector<float>>& mask,
                        int pool_size, int stride) {
    return std::vector<std::vector<float>>(input.size());
}

std::vector<std::vector<float>>
CNN::dense_backward(const std::vector<std::vector<float>>& input,
                   const std::vector<std::vector<float>>& weights) {
    std::vector<std::vector<float>> grad_input(1, std::vector<float>(weights[0].size(), 0.0f));

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            grad_input[0][j] += input[0][i] * weights[i][j];
        }
    }

    return grad_input;
}

void CNN::update_parameters_conv2d(const std::vector<std::vector<std::vector<float>>& grad_weights,
                                   const std::vector<std::vector<float>>& grad_bias,
                                   const TrainingConfig& config) {
    for (size_t i = 0; i < grad_weights.size(); ++i) {
        for (size_t j = 0; j < grad_weights[i].size(); ++j) {
            for (size_t k = 0; k < grad_weights[i][j].size(); ++k) {
                weights_3d_[i][j][k] -= config.learning_rate * grad_weights[i][j][k];
            }
        }
    }

    for (size_t i = 0; i < grad_bias.size(); ++i) {
        for (size_t j = 0; j < grad_bias[i].size(); ++j) {
            biases_3d_[i][i][j] -= config.learning_rate * grad_bias[i][j];
        }
    }
}

void CNN::update_parameters_dense(const std::vector<std::vector<float>>& grad_weights,
                                const std::vector<float>& grad_bias,
                                const TrainingConfig& config) {
    for (size_t i = 0; i < grad_weights.size(); ++i) {
        for (size_t j = 0; j < grad_weights[0].size(); ++j) {
            weights_[0][i][j] -= config.learning_rate * grad_weights[i][j];
        }
    }

    for (size_t i = 0; i < grad_bias.size(); ++i) {
        biases_[0][i] -= config.learning_rate * grad_bias[i];
    }
}

float CNN::calculate_loss(const std::vector<std::vector<float>>& predictions,
                        const std::vector<int>& labels) {
    float loss = 0.0f;
    for (size_t i = 0; i < labels.size(); ++i) {
        loss -= std::log(predictions[0][labels[i]] + 1e-15f);
    }
    return loss / labels.size();
}

std::vector<int> CNN::predict(const std::vector<std::vector<float>>& data) {
    std::vector<std::vector<std::vector<float>>> input(data.size());
    std::vector<int> predictions(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        input[i] = std::vector<std::vector<float>>(
            input_channels_,
            std::vector<float>(input_height_ * input_width_));

        for (int c = 0; c < input_channels_; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    int idx = c * input_height_ * input_width_ + h * input_width_ + w;
                    if (idx < data[i].size()) {
                        input[i][c][h * input_width_ + w] = data[i][idx];
                    }
                }
            }
        }

        auto conv1_output = conv2d_forward(input[i], weights_3d_[0], biases_3d_[0],
                                         config_.conv_layers[0].stride, config_.conv_layers[0].padding);

        auto relu1_output = conv1_output;
        for (auto& channel : relu1_output) {
            for (auto& val : channel) {
                val = relu(val);
            }
        }

        auto pool1_output = max_pool2d_forward(relu1_output, 2, 2);

        auto flatten1_output = std::vector<std::vector<float>>(1);
        int flatten_size = 0;
        for (const auto& channel : pool1_output) {
            flatten_size += channel.size();
        }
        flatten1_output[0].resize(flatten_size);

        int pos = 0;
        for (const auto& channel : pool1_output) {
            for (float val : channel) {
                flatten1_output[0][pos++] = val;
            }
        }

        auto dense1_output = dense_forward(flatten1_output, weights_[0], biases_[0]);

        auto relu2_output = dense1_output;
        for (auto& val : relu2_output[0]) {
            val = relu(val);
        }

        auto dense2_output = dense_forward(relu2_output, weights_[1], biases_[1]);

        auto softmax_output = dense2_output;
        float max_val = *std::max_element(softmax_output[0].begin(), softmax_output[0].end());
        float sum = 0.0f;

        for (auto& val : softmax_output[0]) {
            val = std::exp(val - max_val);
            sum += val;
        }

        for (auto& val : softmax_output[0]) {
            val /= sum;
        }

        int max_idx = 0;
        float max_prob = 0.0f;
        for (size_t j = 0; j < softmax_output[0].size(); ++j) {
            if (softmax_output[0][j] > max_prob) {
                max_prob = softmax_output[0][j];
                max_idx = static_cast<int>(j);
            }
        }

        predictions[i] = max_idx;
    }

    return predictions;
}

bool CNN::save_model(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    size_t num_conv_layers = config_.conv_layers.size();
    file.write(reinterpret_cast<const char*>(&num_conv_layers), sizeof(num_conv_layers));

    for (size_t i = 0; i < num_conv_layers; ++i) {
        const auto& conv = config_.conv_layers[i];
        file.write(reinterpret_cast<const char*>(&conv.in_channels), sizeof(conv.in_channels));
        file.write(reinterpret_cast<const char*>(&conv.out_channels), sizeof(conv.out_channels));
        file.write(reinterpret_cast<const char*>(&conv.kernel_size), sizeof(conv.kernel_size));
        file.write(reinterpret_cast<const char*>(&conv.stride), sizeof(conv.stride));
        file.write(reinterpret_cast<const char*>(&conv.padding), sizeof(conv.padding));

        size_t kernel_size = weights_3d_[i].size();
        for (const auto& kernel : weights_3d_[i]) {
            size_t kernel_dim = kernel.size();
            file.write(reinterpret_cast<const char*>(&kernel_dim), sizeof(kernel_dim));
            for (const auto& row : kernel) {
                size_t row_size = row.size();
                file.write(reinterpret_cast<const char*>(&row_size), sizeof(row_size));
                file.write(reinterpret_cast<const char*>(row.data()), row_size * sizeof(float));
            }
        }
    }

    size_t num_dense_layers = config_.dense_layers.size();
    file.write(reinterpret_cast<const char*>(&num_dense_layers), sizeof(num_dense_layers));

    for (size_t i = 0; i < num_dense_layers; ++i) {
        const auto& dense = config_.dense_layers[i];
        file.write(reinterpret_cast<const char*>(&dense.input_size), sizeof(dense.input_size));
        file.write(reinterpret_cast<const char*>(&dense.output_size), sizeof(dense.output_size));

        size_t weight_size = weights_[i].size();
        file.write(reinterpret_cast<const char*>(&weight_size), sizeof(weight_size));
        for (const auto& row : weights_[i]) {
            size_t row_size = row.size();
            file.write(reinterpret_cast<const char*>(&row_size), sizeof(row_size));
            file.write(reinterpret_cast<const char*>(row.data()), row_size * sizeof(float));
        }
    }

    file.write(reinterpret_cast<const char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.write(reinterpret_cast<const char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.write(reinterpret_cast<const char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    return file.good();
}

bool CNN::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    size_t num_conv_layers;
    file.read(reinterpret_cast<char*>(&num_conv_layers), sizeof(num_conv_layers));

    config_.conv_layers.resize(num_conv_layers);
    weights_3d_.resize(num_conv_layers);
    biases_3d_.resize(num_conv_layers);

    for (size_t i = 0; i < num_conv_layers; ++i) {
        auto& conv = config_.conv_layers[i];
        file.read(reinterpret_cast<char*>(&conv.in_channels), sizeof(conv.in_channels));
        file.read(reinterpret_cast<char*>(&conv.out_channels), sizeof(conv.out_channels));
        file.read(reinterpret_cast<char*>(&conv.kernel_size), sizeof(conv.kernel_size));
        file.read(reinterpret_cast<char*>(&conv.stride), sizeof(conv.stride));
        file.read(reinterpret_cast<char*>(&conv.padding), sizeof(conv.padding));

        weights_3d_[i].resize(conv.out_channels);
        for (int oc = 0; oc < conv.out_channels; ++oc) {
            size_t kernel_dim;
            file.read(reinterpret_cast<char*>(&kernel_dim), sizeof(kernel_dim));
            weights_3d_[i][oc].resize(kernel_dim);
            for (auto& row : weights_3d_[i][oc]) {
                size_t row_size;
                file.read(reinterpret_cast<char*>(&row_size), sizeof(row_size));
                row.resize(row_size);
                file.read(reinterpret_cast<char*>(row.data()), row_size * sizeof(float));
            }
        }
    }

    size_t num_dense_layers;
    file.read(reinterpret_cast<char*>(&num_dense_layers), sizeof(num_dense_layers));

    config_.dense_layers.resize(num_dense_layers);
    weights_.resize(num_dense_layers);
    biases_.resize(num_dense_layers);

    for (size_t i = 0; i < num_dense_layers; ++i) {
        auto& dense = config_.dense_layers[i];
        file.read(reinterpret_cast<char*>(&dense.input_size), sizeof(dense.input_size));
        file.read(reinterpret_cast<char*>(&dense.output_size), sizeof(dense.output_size));

        weights_[i].resize(dense.output_size);
        for (auto& row : weights_[i]) {
            size_t row_size;
            file.read(reinterpret_cast<char*>(&row_size), sizeof(row_size));
            row.resize(row_size);
            file.read(reinterpret_cast<char*>(row.data()), row_size * sizeof(float));
        }
    }

    file.read(reinterpret_cast<char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.read(reinterpret_cast<char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.read(reinterpret_cast<char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    return file.good();
}

std::vector<float> CNN::get_weights() const {
    std::vector<float> flat_weights;

    for (const auto& kernel : weights_3d_) {
        for (const auto& channel : kernel) {
            for (float val : channel) {
                flat_weights.push_back(val);
            }
        }
    }

    for (const auto& weight_matrix : weights_) {
        for (const auto& row : weight_matrix) {
            for (float val : row) {
                flat_weights.push_back(val);
            }
        }
    }

    return flat_weights;
}

void CNN::set_weights(const std::vector<float>& weights) {
    size_t pos = 0;

    for (auto& kernel : weights_3d_) {
        for (auto& channel : kernel) {
            for (auto& val : channel) {
                val = weights[pos++];
            }
        }
    }

    for (auto& weight_matrix : weights_) {
        for (auto& row : weight_matrix) {
            for (auto& val : row) {
                val = weights[pos++];
            }
        }
    }
}

std::vector<int> CNN::argmax_2d(const std::vector<std::vector<float>>& predictions) {
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

}