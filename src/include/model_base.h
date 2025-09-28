#pragma once

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>

namespace edge_ai {

enum class ModelType {
    LOGISTIC_REGRESSION,
    DECISION_TREE,
    KNN,
    CNN
};

struct Dataset {
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    int num_samples;
    int num_features;
};

struct TrainingConfig {
    float learning_rate = 0.01f;
    int max_epochs = 1000;
    float tolerance = 1e-6f;
    bool verbose = false;
};

struct QuantizationConfig {
    bool enabled = false;
    float scale = 1.0f;
    int zero_point = 0;
};

class BaseModel {
public:
    BaseModel(ModelType type);
    virtual ~BaseModel() = default;

    virtual bool train(const Dataset& data, const TrainingConfig& config = {}) = 0;
    virtual std::vector<int> predict(const std::vector<std::vector<float>>& data) = 0;
    virtual bool save_model(const std::string& filepath) = 0;
    virtual bool load_model(const std::string& filepath) = 0;

    ModelType get_model_type() const { return model_type_; }
    bool is_quantized() const { return quantized_; }
    void quantize(const QuantizationConfig& config);
    void dequantize();
    void quantize_model();
    void dequantize_model();

    virtual std::vector<float> get_weights() const = 0;
    virtual void set_weights(const std::vector<float>& weights) = 0;

    static float calculate_accuracy(const std::vector<int>& true_labels,
                                   const std::vector<int>& predicted_labels);

protected:
    ModelType model_type_;
    bool quantized_ = false;
    QuantizationConfig quant_config_;
    std::vector<float> original_weights_;

    static float sigmoid(float x);
    static float relu(float x);
    static float relu_derivative(float x);
    static float softmax(const std::vector<float>& x, int index);

    std::vector<int> argmax(const std::vector<std::vector<float>>& predictions);
};

}