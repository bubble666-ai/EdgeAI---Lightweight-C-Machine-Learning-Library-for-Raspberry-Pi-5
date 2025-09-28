#include "decision_tree.h"
#include <algorithm>
#include <fstream>
#include <queue>

namespace edge_ai {

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : BaseModel(ModelType::DECISION_TREE)
    , max_depth_(max_depth)
    , min_samples_split_(min_samples_split) {
}

bool DecisionTree::train(const Dataset& data, const TrainingConfig& config) {
    if (data.num_samples == 0 || data.num_features == 0) {
        return false;
    }

    root_ = build_tree(data.X, data.y, 0);
    return true;
}

std::shared_ptr<TreeNode> DecisionTree::build_tree(const std::vector<std::vector<float>>& X,
                                                const std::vector<int>& y,
                                                int depth) {
    std::shared_ptr<TreeNode> node = std::make_shared<TreeNode>();
    node->is_leaf = true;
    node->class_label = 0;
    node->feature_index = -1;
    node->threshold = 0.0f;
    node->impurity = 1.0f;
    node->samples_count = y.size();

    if (should_stop(X, y, depth)) {
        int max_class = 0;
        int max_count = 0;
        std::vector<int> class_counts(*std::max_element(y.begin(), y.end()) + 1, 0);

        for (int label : y) {
            class_counts[label]++;
            if (class_counts[label] > max_count) {
                max_count = class_counts[label];
                max_class = label;
            }
        }

        node->class_label = max_class;
        node->impurity = calculate_gini(y);
        return node;
    }

    std::vector<int> indices(y.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto [best_threshold, best_feature] = find_best_split(X, y, indices);

    if (best_feature == -1) {
        int max_class = 0;
        int max_count = 0;
        std::vector<int> class_counts(*std::max_element(y.begin(), y.end()) + 1, 0);

        for (int label : y) {
            class_counts[label]++;
            if (class_counts[label] > max_count) {
                max_count = class_counts[label];
                max_class = label;
            }
        }

        node->class_label = max_class;
        node->impurity = calculate_gini(y);
        return node;
    }

    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->impurity = calculate_gini(y);

    std::vector<std::vector<float>> X_left, X_right;
    std::vector<int> y_left, y_right;
    split_data(X, y, best_feature, best_threshold, X_left, X_right, y_left, y_right);

    if (y_left.size() >= min_samples_split_ && y_right.size() >= min_samples_split_) {
        node->left = build_tree(X_left, y_left, depth + 1);
        node->right = build_tree(X_right, y_right, depth + 1);
    } else {
        int max_class_left = 0, max_class_right = 0;
        int max_count_left = 0, max_count_right = 0;
        std::vector<int> class_counts_left(*std::max_element(y_left.begin(), y_left.end()) + 1, 0);
        std::vector<int> class_counts_right(*std::max_element(y_right.begin(), y_right.end()) + 1, 0);

        for (int label : y_left) {
            class_counts_left[label]++;
            if (class_counts_left[label] > max_count_left) {
                max_count_left = class_counts_left[label];
                max_class_left = label;
            }
        }

        for (int label : y_right) {
            class_counts_right[label]++;
            if (class_counts_right[label] > max_count_right) {
                max_count_right = class_counts_right[label];
                max_class_right = label;
            }
        }

        auto leaf_left = std::make_shared<TreeNode>();
        leaf_left->is_leaf = true;
        leaf_left->class_label = max_class_left;
        leaf_left->impurity = calculate_gini(y_left);
        leaf_left->samples_count = y_left.size();
        node->left = leaf_left;

        auto leaf_right = std::make_shared<TreeNode>();
        leaf_right->is_leaf = true;
        leaf_right->class_label = max_class_right;
        leaf_right->impurity = calculate_gini(y_right);
        leaf_right->samples_count = y_right.size();
        node->right = leaf_right;
    }

    return node;
}

bool DecisionTree::should_stop(const std::vector<std::vector<float>>& X,
                             const std::vector<int>& y,
                             int depth) {
    if (depth >= max_depth_ || y.size() < min_samples_split_) {
        return true;
    }

    bool all_same = true;
    for (size_t i = 1; i < y.size(); ++i) {
        if (y[i] != y[0]) {
            all_same = false;
            break;
        }
    }
    return all_same;
}

std::pair<float, int> DecisionTree::find_best_split(const std::vector<std::vector<float>>& X,
                                                  const std::vector<int>& y,
                                                  const std::vector<int>& indices) {
    float best_gain = -1.0f;
    int best_feature = -1;
    float best_threshold = 0.0f;

    float parent_gini = calculate_gini(y);

    for (int feature = 0; feature < X[0].size(); ++feature) {
        std::vector<float> feature_values;
        for (int idx : indices) {
            feature_values.push_back(X[idx][feature]);
        }
        std::sort(feature_values.begin(), feature_values.end());

        for (size_t i = 0; i < feature_values.size() - 1; ++i) {
            float threshold = (feature_values[i] + feature_values[i + 1]) / 2.0f;

            std::vector<int> left_indices, right_indices;
            for (int idx : indices) {
                if (X[idx][feature] <= threshold) {
                    left_indices.push_back(idx);
                } else {
                    right_indices.push_back(idx);
                }
            }

            if (left_indices.size() == 0 || right_indices.size() == 0) {
                continue;
            }

            std::vector<int> y_left, y_right;
            for (int idx : left_indices) y_left.push_back(y[idx]);
            for (int idx : right_indices) y_right.push_back(y[idx]);

            float left_gini = calculate_gini(y_left);
            float right_gini = calculate_gini(y_right);
            float n_left = static_cast<float>(left_indices.size());
            float n_right = static_cast<float>(right_indices.size());
            float n_total = n_left + n_right;

            float gain = parent_gini - (n_left / n_total * left_gini + n_right / n_total * right_gini);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feature;
                best_threshold = threshold;
            }
        }
    }

    return {best_threshold, best_feature};
}

float DecisionTree::calculate_gini(const std::vector<int>& y_subset) {
    if (y_subset.empty()) return 1.0f;

    std::vector<int> class_counts(*std::max_element(y_subset.begin(), y_subset.end()) + 1, 0);
    for (int label : y_subset) {
        class_counts[label]++;
    }

    float gini = 1.0f;
    float n = static_cast<float>(y_subset.size());
    for (int count : class_counts) {
        if (count > 0) {
            float p = static_cast<float>(count) / n;
            gini -= p * p;
        }
    }

    return gini;
}

void DecisionTree::split_data(const std::vector<std::vector<float>>& X,
                           const std::vector<int>& y,
                           int feature_index,
                           float threshold,
                           std::vector<std::vector<float>>& X_left,
                           std::vector<std::vector<float>>& X_right,
                           std::vector<int>& y_left,
                           std::vector<int>& y_right) {
    X_left.clear();
    X_right.clear();
    y_left.clear();
    y_right.clear();

    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][feature_index] <= threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }
}

int DecisionTree::predict_single(const std::vector<float>& sample, std::shared_ptr<TreeNode> node) {
    if (node->is_leaf) {
        return node->class_label;
    }

    if (sample[node->feature_index] <= node->threshold) {
        return predict_single(sample, node->left);
    } else {
        return predict_single(sample, node->right);
    }
}

std::vector<int> DecisionTree::predict(const std::vector<std::vector<float>>& data) {
    std::vector<int> predictions(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        predictions[i] = predict_single(data[i], root_);
    }
    return predictions;
}

bool DecisionTree::save_model(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<const char*>(&max_depth_), sizeof(max_depth_));
    file.write(reinterpret_cast<const char*>(&min_samples_split_), sizeof(min_samples_split_));
    file.write(reinterpret_cast<const char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.write(reinterpret_cast<const char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.write(reinterpret_cast<const char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    save_tree_helper(root_, file);
    return file.good();
}

void DecisionTree::save_tree_helper(std::shared_ptr<TreeNode> node, std::ofstream& file) {
    if (!node) {
        bool is_null = true;
        file.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
        return;
    }

    bool is_null = false;
    file.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
    file.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(node->is_leaf));
    file.write(reinterpret_cast<const char*>(&node->class_label), sizeof(node->class_label));
    file.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(node->feature_index));
    file.write(reinterpret_cast<const char*>(&node->threshold), sizeof(node->threshold));
    file.write(reinterpret_cast<const char*>(&node->impurity), sizeof(node->impurity));
    file.write(reinterpret_cast<const char*>(&node->samples_count), sizeof(node->samples_count));

    save_tree_helper(node->left, file);
    save_tree_helper(node->right, file);
}

bool DecisionTree::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;

    file.read(reinterpret_cast<char*>(&max_depth_), sizeof(max_depth_));
    file.read(reinterpret_cast<char*>(&min_samples_split_), sizeof(min_samples_split_));
    file.read(reinterpret_cast<char*>(&quantized_), sizeof(quantized_));

    if (quantized_) {
        file.read(reinterpret_cast<char*>(&quant_config_.scale), sizeof(quant_config_.scale));
        file.read(reinterpret_cast<char*>(&quant_config_.zero_point), sizeof(quant_config_.zero_point));
    }

    return load_tree_helper(root_, file);
}

bool DecisionTree::load_tree_helper(std::shared_ptr<TreeNode>& node, std::ifstream& file) {
    bool is_null;
    file.read(reinterpret_cast<char*>(&is_null), sizeof(is_null));
    if (is_null) {
        node = nullptr;
        return true;
    }

    node = std::make_shared<TreeNode>();
    file.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(node->is_leaf));
    file.read(reinterpret_cast<char*>(&node->class_label), sizeof(node->class_label));
    file.read(reinterpret_cast<char*>(&node->feature_index), sizeof(node->feature_index));
    file.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold));
    file.read(reinterpret_cast<char*>(&node->impurity), sizeof(node->impurity));
    file.read(reinterpret_cast<char*>(&node->samples_count), sizeof(node->samples_count));

    return load_tree_helper(node->left, file) && load_tree_helper(node->right, file);
}

std::vector<float> DecisionTree::get_weights() const {
    std::vector<float> weights;
    if (root_) {
        std::queue<std::shared_ptr<TreeNode>> q;
        q.push(root_);
        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            if (node->is_leaf) {
                weights.push_back(static_cast<float>(node->class_label));
            } else {
                weights.push_back(static_cast<float>(node->feature_index));
                weights.push_back(node->threshold);
                q.push(node->left);
                q.push(node->right);
            }
        }
    }
    return weights;
}

void DecisionTree::set_weights(const std::vector<float>& weights) {
    weights_.resize(weights.size());
    std::copy(weights.begin(), weights.end(), weights_.begin());
}

}