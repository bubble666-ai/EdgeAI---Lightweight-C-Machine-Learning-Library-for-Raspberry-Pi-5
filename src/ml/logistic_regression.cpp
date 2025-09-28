#include "logistic_regression.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace edge_ai {

LogisticRegression::LogisticRegression(int input_dim, int num_classes)
    : BaseModel(ModelType::LOGISTIC_REGRESSION)
    , input_dim_(input_dim)
    , num_classes_(num_classes) {
    initialize_parameters();
}

void LogisticRegression::initialize_parameters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);

    weights_.resize(num_classes_, std::vector<float>(input_dim_));
    biases_.resize(num_classes_);

    for (int i = 0; i < num_classes_; ++i) {
        for (int j = 0; j < input_dim_; ++j) {
            weights_[i][j] = dist(gen);
        }
        biases_[i] = 0.0f;
    }
}

bool LogisticRegression::train(const Dataset& data, const TrainingConfig& config) {
    if (data.num_samples == 0 || data.num_features != input_dim_) {
        return false;
    }

    float prev_loss = std::numeric_limits<float>::max();
    int no_improvement_count = 0;

    for (int epoch = 0; epoch < config.max_epochs; ++epoch) {
        std::vector<std::vector<float>> predictions = forward_pass(data.X);

        float loss = calculate_loss(predictions, data.y);

        if (config.verbose && epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }

        if (std::abs(prev_loss - loss) < config.tolerance) {
            no_improvement_count++;
            if (no_improvement_count >= 10) break;
        } else {
            no_improvement_count = 0;
        }
        prev_loss = loss;

        gradients_w_.resize(num_classes_, std::vector<float>(input_dim_, 0.0f));
        std::vector<float> gradients_b(num_classes_, 0.0f);

        for (int i = 0; i < data.num_samples; ++i) {
            for (int j = 0; j < num_classes_; ++j) {
                float error = predictions[i][j] - (data.y[i] == j ? 1.0f : 0.0f);
                for (int k = 0; k < input_dim_; ++k) {
                    gradients_w_[j][k] += error * data.X[i][k];
                }
                gradients_b[j] += error;
            }
        }

        for (int j = 0; j < num_classes_; ++j) {
            for (int k = 0; k < input_dim_; ++k) {
                gradients_w_[j][k] /= data.num_samples;
            }
            gradients_b[j] /= data.num_samples;
        }

        update_parameters(gradients_w_, gradients_b, config);
    }

    return true;
}

std::vector<std::vector<float>> LogisticRegression::forward_pass(const std::vector<std::vector<float>>& X) {
    std::vector<std::vector<float>> predictions(X.size(), std::vector<float>(num_classes_));

    for (size_t i = 0; i < X.size(); ++i) {
        for (int j = 0; j < num_classes_; ++j) {
            float logit = biases_[j];
            for (int k = 0; k < input_dim_; ++k) {
                logit += weights_[j][k] * X[i][k];
            }
            predictions[i][j] = logit;
        }

        for (int j = 0; j < num_classes_; ++j) {
            predictions[i][j] = softmax(predictions[i], j);
        }
    }

    return predictions;
}

float LogisticRegression::calculate_loss(const std::vector<std::vector<float>>& predictions,
                                      const std::vector<int>& labels) {
    float loss = 0.0f;
    for (size_t i = 0; i < labels.size(); ++i) {
        loss -= std::log(predictions[i][labels[i]] + 1e-15f);
    }
    return loss / labels.size();
}

void LogisticRegression::update_parameters(const std::vector<std::vector<float>>& gradients_w,
                                         const std::vector<float>& gradients_b,
                                         const TrainingConfig& config) {
    for (int i = 0; i < num_classes_; ++i) {
        for (int j = 0; j < input_dim_; ++j) {
            weights_[i][j] -= config.learning_rate * gradients_w[i][j];
        }
        biases_[i] -= config.learning_rate * gradients_b[i];
    }
}

std::vector<int> LogisticRegression::predict(const std::vector<std::vector<float>>& data) {
    std::vector<std::vector<float>> predictions = forward_pass(data);
    return argmax(predictions);
}

bool LogisticRegression::save_model(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<const char*>(&input_dim_), sizeof(input_dim_));
    file.write(reinterpret_cast<const char*>(&num_classes_), sizeof(num_classes_));
    file.write(reinterpret_cast<const char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.write(reinterpret_cast<const char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.write(reinterpret_cast<const char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    for (const auto& row : weights_) {
        size_t size = row.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(row.data()), size * sizeof(float));
    }

    size_t bias_size = biases_.size();
    file.write(reinterpret_cast<const char*>(&bias_size), sizeof(bias_size));
    file.write(reinterpret_cast<const char*>(biases_.data()), bias_size * sizeof(float));

    return file.good();
}

bool LogisticRegression::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    file.read(reinterpret_cast<char*>(&input_dim_), sizeof(input_dim_));
    file.read(reinterpret_cast<char*>(&num_classes_), sizeof(num_classes_));
    file.read(reinterpret_cast<char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.read(reinterpret_cast<char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.read(reinterpret_cast<char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    weights_.resize(num_classes_);
    for (int i = 0; i < num_classes_; ++i) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        weights_[i].resize(size);
        file.read(reinterpret_cast<char*>(weights_[i].data()), size * sizeof(float));
    }

    size_t bias_size;
    file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
    biases_.resize(bias_size);
    file.read(reinterpret_cast<char*>(biases_.data()), bias_size * sizeof(float));

    return file.good();
}

std::vector<float> LogisticRegression::get_weights() const {
    std::vector<float> flat_weights;
    for (const auto& row : weights_) {
        flat_weights.insert(flat_weights.end(), row.begin(), row.end());
    }
    return flat_weights;
}

void LogisticRegression::set_weights(const std::vector<float>& weights) {
    size_t pos = 0;
    for (int i = 0; i < num_classes_; ++i) {
        for (int j = 0; j < input_dim_; ++j) {
            weights_[i][j] = weights[pos++];
        }
    }
}

}