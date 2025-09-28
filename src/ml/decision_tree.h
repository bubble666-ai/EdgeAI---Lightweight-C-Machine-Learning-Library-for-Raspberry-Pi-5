#pragma once

#include "../include/model_base.h"
#include <vector>
#include <random>

namespace edge_ai {

struct TreeNode {
    bool is_leaf;
    int class_label;
    int feature_index;
    float threshold;
    std::shared_ptr<TreeNode> left;
    std::shared_ptr<TreeNode> right;
    float impurity;
    int samples_count;
};

class DecisionTree : public BaseModel {
public:
    DecisionTree(int max_depth = 10, int min_samples_split = 2);
    ~DecisionTree() = default;

    bool train(const Dataset& data, const TrainingConfig& config = {}) override;
    std::vector<int> predict(const std::vector<std::vector<float>>& data) override;
    bool save_model(const std::string& filepath) override;
    bool load_model(const std::string& filepath) override;

    std::vector<float> get_weights() const override;
    void set_weights(const std::vector<float>& weights) override;

    int get_max_depth() const { return max_depth_; }
    int get_min_samples_split() const { return min_samples_split_; }

private:
    int max_depth_;
    int min_samples_split_;
    std::shared_ptr<TreeNode> root_;

    std::shared_ptr<TreeNode> build_tree(const std::vector<std::vector<float>>& X,
                                        const std::vector<int>& y,
                                        int depth);
    bool should_stop(const std::vector<std::vector<float>>& X,
                   const std::vector<int>& y,
                   int depth);
    std::pair<float, int> find_best_split(const std::vector<std::vector<float>>& X,
                                         const std::vector<int>& y,
                                         const std::vector<int>& indices);
    float calculate_gini(const std::vector<int>& y_subset);
    void split_data(const std::vector<std::vector<float>>& X,
                   const std::vector<int>& y,
                   int feature_index,
                   float threshold,
                   std::vector<std::vector<float>>& X_left,
                   std::vector<std::vector<float>>& X_right,
                   std::vector<int>& y_left,
                   std::vector<int>& y_right);
    int predict_single(const std::vector<float>& sample, std::shared_ptr<TreeNode> node);
    void save_tree_helper(std::shared_ptr<TreeNode> node, std::ofstream& file);
    bool load_tree_helper(std::shared_ptr<TreeNode>& node, std::ifstream& file);
};

}