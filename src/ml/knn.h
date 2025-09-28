#pragma once

#include "../include/model_base.h"
#include <vector>
#include <cmath>

namespace edge_ai {

struct DistanceMetric {
    enum class Type {
        EUCLIDEAN,
        MANHATTAN,
        COSINE
    };
};

class KNN : public BaseModel {
public:
    KNN(int k = 5, DistanceMetric::Type metric = DistanceMetric::Type::EUCLIDEAN);
    ~KNN() = default;

    bool train(const Dataset& data, const TrainingConfig& config = {}) override;
    std::vector<int> predict(const std::vector<std::vector<float>>& data) override;
    bool save_model(const std::string& filepath) override;
    bool load_model(const std::string& filepath) override;

    std::vector<float> get_weights() const override;
    void set_weights(const std::vector<float>& weights) override;

    void set_k(int k) { k_ = k; }
    int get_k() const { return k_; }
    void set_metric(DistanceMetric::Type metric) { metric_ = metric; }
    DistanceMetric::Type get_metric() const { return metric_; }

private:
    int k_;
    DistanceMetric::Type metric_;
    std::vector<std::vector<float>> training_data_;
    std::vector<int> training_labels_;

    float calculate_distance(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<std::pair<int, float>> get_k_nearest_neighbors(const std::vector<float>& sample);
    int vote_majority(const std::vector<std::pair<int, float>>& neighbors);
};

}