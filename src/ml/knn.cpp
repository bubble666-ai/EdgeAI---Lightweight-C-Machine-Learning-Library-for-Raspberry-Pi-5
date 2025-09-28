#include "knn.h"
#include <fstream>
#include <algorithm>
#include <limits>

namespace edge_ai {

KNN::KNN(int k, DistanceMetric::Type metric)
    : BaseModel(ModelType::KNN)
    , k_(k)
    , metric_(metric) {
}

bool KNN::train(const Dataset& data, const TrainingConfig& config) {
    if (data.num_samples == 0 || data.num_features == 0) {
        return false;
    }

    training_data_ = data.X;
    training_labels_ = data.y;
    return true;
}

float KNN::calculate_distance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<float>::max();
    }

    switch (metric_) {
        case DistanceMetric::Type::EUCLIDEAN: {
            float sum = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }
        case DistanceMetric::Type::MANHATTAN: {
            float sum = 0.0f;
            for (size_t i = 0; i < a.size(); ++i) {
                sum += std::abs(a[i] - b[i]);
            }
            return sum;
        }
        case DistanceMetric::Type::COSINE: {
            float dot_product = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;

            for (size_t i = 0; i < a.size(); ++i) {
                dot_product += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            norm_a = std::sqrt(norm_a);
            norm_b = std::sqrt(norm_b);

            if (norm_a == 0.0f || norm_b == 0.0f) {
                return 0.0f;
            }

            float similarity = dot_product / (norm_a * norm_b);
            return 1.0f - similarity;
        }
        default:
            return std::numeric_limits<float>::max();
    }
}

std::vector<std::pair<int, float>> KNN::get_k_nearest_neighbors(const std::vector<float>& sample) {
    std::vector<std::pair<int, float>> distances;
    distances.reserve(training_data_.size());

    for (size_t i = 0; i < training_data_.size(); ++i) {
        float dist = calculate_distance(sample, training_data_[i]);
        distances.emplace_back(training_labels_[i], dist);
    }

    std::sort(distances.begin(), distances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    if (distances.size() > static_cast<size_t>(k_)) {
        distances.resize(k_);
    }

    return distances;
}

int KNN::vote_majority(const std::vector<std::pair<int, float>>& neighbors) {
    std::vector<int> votes(*std::max_element(neighbors.begin(), neighbors.end(),
                                            [](const auto& a, const auto& b) { return a.first < b.first; }).first + 1, 0);

    for (const auto& [label, dist] : neighbors) {
        votes[label]++;
    }

    int max_votes = 0;
    int best_label = 0;
    for (size_t i = 0; i < votes.size(); ++i) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            best_label = static_cast<int>(i);
        }
    }

    return best_label;
}

std::vector<int> KNN::predict(const std::vector<std::vector<float>>& data) {
    std::vector<int> predictions(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        auto neighbors = get_k_nearest_neighbors(data[i]);
        predictions[i] = vote_majority(neighbors);
    }
    return predictions;
}

bool KNN::save_model(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<const char*>(&k_), sizeof(k_));
    file.write(reinterpret_cast<const char*>(&metric_), sizeof(metric_));
    file.write(reinterpret_cast<const char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.write(reinterpret_cast<const char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.write(reinterpret_cast<const char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    size_t train_size = training_data_.size();
    file.write(reinterpret_cast<const char*>(&train_size), sizeof(train_size));

    for (const auto& sample : training_data_) {
        size_t sample_size = sample.size();
        file.write(reinterpret_cast<const char*>(&sample_size), sizeof(sample_size));
        file.write(reinterpret_cast<const char*>(sample.data()), sample_size * sizeof(float));
    }

    size_t label_size = training_labels_.size();
    file.write(reinterpret_cast<const char*>(&label_size), sizeof(label_size));
    file.write(reinterpret_cast<const char*>(training_labels_.data()), label_size * sizeof(int));

    return file.good();
}

bool KNN::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    file.read(reinterpret_cast<char*>(&k_), sizeof(k_));
    file.read(reinterpret_cast<char*>(&metric_), sizeof(metric_));
    file.read(reinterpret_cast<char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.read(reinterpret_cast<char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.read(reinterpret_cast<char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    size_t train_size;
    file.read(reinterpret_cast<char*>(&train_size), sizeof(train_size));
    training_data_.resize(train_size);

    for (auto& sample : training_data_) {
        size_t sample_size;
        file.read(reinterpret_cast<char*>(&sample_size), sizeof(sample_size));
        sample.resize(sample_size);
        file.read(reinterpret_cast<char*>(sample.data()), sample_size * sizeof(float));
    }

    size_t label_size;
    file.read(reinterpret_cast<char*>(&label_size), sizeof(label_size));
    training_labels_.resize(label_size);
    file.read(reinterpret_cast<char*>(training_labels_.data()), label_size * sizeof(int));

    return file.good();
}

std::vector<float> KNN::get_weights() const {
    std::vector<float> weights;
    weights.push_back(static_cast<float>(k_));
    weights.push_back(static_cast<float>(metric_));
    return weights;
}

void KNN::set_weights(const std::vector<float>& weights) {
    if (weights.size() >= 1) {
        k_ = static_cast<int>(weights[0]);
    }
    if (weights.size() >= 2) {
        metric_ = static_cast<DistanceMetric::Type>(static_cast<int>(weights[1]));
    }
}

}