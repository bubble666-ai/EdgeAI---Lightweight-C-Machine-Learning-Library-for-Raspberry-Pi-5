#pragma once

#include "../include/model_base.h"
#include <vector>
#include <random>

namespace edge_ai {

class LogisticRegression : public BaseModel {
public:
    LogisticRegression(int input_dim, int num_classes = 2);
    ~LogisticRegression() = default;

    bool train(const Dataset& data, const TrainingConfig& config = {}) override;
    std::vector<int> predict(const std::vector<std::vector<float>>& data) override;
    bool save_model(const std::string& filepath) override;
    bool load_model(const std::string& filepath) override;

    std::vector<float> get_weights() const override;
    void set_weights(const std::vector<float>& weights) override;

    inline int get_input_dim() const { return input_dim_; }
    inline int get_num_classes() const { return num_classes_; }

private:
    int input_dim_;
    int num_classes_;
    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
    std::vector<std::vector<float>> gradients_w_;
    std::vector<float> gradients_b_;

    void initialize_parameters();
    std::vector<std::vector<float>> forward_pass(const std::vector<std::vector<float>>& X);
    void update_parameters(const std::vector<std::vector<float>>& gradients_w,
                          const std::vector<float>& gradients_b,
                          const TrainingConfig& config);
    float calculate_loss(const std::vector<std::vector<float>>& predictions,
                        const std::vector<int>& labels);
};

}