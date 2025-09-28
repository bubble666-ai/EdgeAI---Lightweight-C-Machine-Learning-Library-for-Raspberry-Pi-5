#include "model_factory.h"

namespace edge_ai {

std::unique_ptr<BaseModel> ModelFactory::create_model(ModelType type, const std::vector<int>& params) {
    switch (type) {
        case ModelType::LOGISTIC_REGRESSION: {
            int input_dim = params.size() > 0 ? params[0] : 4;
            int num_classes = params.size() > 1 ? params[1] : 2;
            return create_logistic_regression(input_dim, num_classes);
        }
        case ModelType::DECISION_TREE: {
            int max_depth = params.size() > 0 ? params[0] : 10;
            int min_samples_split = params.size() > 1 ? params[1] : 2;
            return create_decision_tree(max_depth, min_samples_split);
        }
        case ModelType::KNN: {
            int k = params.size() > 0 ? params[0] : 5;
            return create_knn(k);
        }
        case ModelType::CNN: {
            return create_cnn();
        }
        default:
            throw std::invalid_argument("Unknown model type");
    }
}

ModelType ModelFactory::string_to_model_type(const std::string& type_str) {
    if (type_str == "logistic_regression" || type_str == "logreg") {
        return ModelType::LOGISTIC_REGRESSION;
    } else if (type_str == "decision_tree" || type_str == "dt") {
        return ModelType::DECISION_TREE;
    } else if (type_str == "knn" || type_str == "k_nearest_neighbors") {
        return ModelType::KNN;
    } else if (type_str == "cnn" || type_str == "convolutional_neural_network") {
        return ModelType::CNN;
    } else {
        throw std::invalid_argument("Unknown model type string: " + type_str);
    }
}

std::string ModelFactory::model_type_to_string(ModelType type) {
    switch (type) {
        case ModelType::LOGISTIC_REGRESSION:
            return "logistic_regression";
        case ModelType::DECISION_TREE:
            return "decision_tree";
        case ModelType::KNN:
            return "knn";
        case ModelType::CNN:
            return "cnn";
        default:
            return "unknown";
    }
}

}