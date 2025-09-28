#pragma once

#include "../include/model_base.h"
#include "../ml/logistic_regression.h"
#include "../ml/decision_tree.h"
#include "../ml/knn.h"
#include "../ml/cnn.h"
#include <memory>
#include <string>

namespace edge_ai {

class ModelFactory {
public:
    static std::unique_ptr<BaseModel> create_model(ModelType type, const std::vector<int>& params = {});

    static std::unique_ptr<BaseModel> create_logistic_regression(int input_dim, int num_classes = 2) {
        return std::make_unique<LogisticRegression>(input_dim, num_classes);
    }

    static std::unique_ptr<BaseModel> create_decision_tree(int max_depth = 10, int min_samples_split = 2) {
        return std::make_unique<DecisionTree>(max_depth, min_samples_split);
    }

    static std::unique_ptr<BaseModel> create_knn(int k = 5, int input_dim = 4) {
        return std::make_unique<KNN>(k, DistanceMetric::Type::EUCLIDEAN);
    }

    static std::unique_ptr<BaseModel> create_cnn() {
        CNNConfig config;
        config.conv_layers.push_back({3, 16, 3, 1, 0});
        config.conv_layers.push_back({16, 32, 3, 1, 0});
        config.dense_layers.push_back({32 * 30 * 30, 128});
        config.dense_layers.push_back({128, 10});
        return std::make_unique<CNN>(config);
    }

    static ModelType string_to_model_type(const std::string& type_str);
    static std::string model_type_to_string(ModelType type);
};

}