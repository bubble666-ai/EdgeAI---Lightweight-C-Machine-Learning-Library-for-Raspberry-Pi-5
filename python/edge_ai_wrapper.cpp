#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>

#include "../src/include/model_base.h"
#include "../src/ml/logistic_regression.h"
#include "../src/ml/decision_tree.h"
#include "../src/ml/knn.h"
#include "../src/ml/cnn.h"
#include "../src/utils/model_factory.h"
#include "../src/utils/quantization.h"

namespace py = pybind11;

class PyBaseModel : public edge_ai::BaseModel {
public:
    using BaseModel::BaseModel;

    bool train(const edge_ai::Dataset& data, const edge_ai::TrainingConfig& config = {}) override {
        PYBIND11_OVERLOAD_PURE(bool, BaseModel, train, data, config);
    }

    std::vector<int> predict(const std::vector<std::vector<float>>& data) override {
        PYBIND11_OVERLOAD_PURE(std::vector<int>, BaseModel, predict, data);
    }

    bool save_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD_PURE(bool, BaseModel, save_model, filepath);
    }

    bool load_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD_PURE(bool, BaseModel, load_model, filepath);
    }

    std::vector<float> get_weights() const override {
        PYBIND11_OVERLOAD_PURE(std::vector<float>, BaseModel, get_weights);
    }

    void set_weights(const std::vector<float>& weights) override {
        PYBIND11_OVERLOAD_PURE(void, BaseModel, set_weights, weights);
    }
};

class PyLogisticRegression : public edge_ai::LogisticRegression {
public:
    using LogisticRegression::LogisticRegression;

    bool train(const edge_ai::Dataset& data, const edge_ai::TrainingConfig& config = {}) override {
        PYBIND11_OVERLOAD(bool, LogisticRegression, train, data, config);
    }

    std::vector<int> predict(const std::vector<std::vector<float>>& data) override {
        PYBIND11_OVERLOAD(std::vector<int>, LogisticRegression, predict, data);
    }

    bool save_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, LogisticRegression, save_model, filepath);
    }

    bool load_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, LogisticRegression, load_model, filepath);
    }

    std::vector<float> get_weights() const override {
        PYBIND11_OVERLOAD(std::vector<float>, LogisticRegression, get_weights);
    }

    void set_weights(const std::vector<float>& weights) override {
        PYBIND11_OVERLOAD(void, LogisticRegression, set_weights, weights);
    }
};

class PyDecisionTree : public edge_ai::DecisionTree {
public:
    using DecisionTree::DecisionTree;

    bool train(const edge_ai::Dataset& data, const edge_ai::TrainingConfig& config = {}) override {
        PYBIND11_OVERLOAD(bool, DecisionTree, train, data, config);
    }

    std::vector<int> predict(const std::vector<std::vector<float>>& data) override {
        PYBIND11_OVERLOAD(std::vector<int>, DecisionTree, predict, data);
    }

    bool save_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, DecisionTree, save_model, filepath);
    }

    bool load_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, DecisionTree, load_model, filepath);
    }

    std::vector<float> get_weights() const override {
        PYBIND11_OVERLOAD(std::vector<float>, DecisionTree, get_weights);
    }

    void set_weights(const std::vector<float>& weights) override {
        PYBIND11_OVERLOAD(void, DecisionTree, set_weights, weights);
    }
};

class PyKNN : public edge_ai::KNN {
public:
    using KNN::KNN;

    bool train(const edge_ai::Dataset& data, const edge_ai::TrainingConfig& config = {}) override {
        PYBIND11_OVERLOAD(bool, KNN, train, data, config);
    }

    std::vector<int> predict(const std::vector<std::vector<float>>& data) override {
        PYBIND11_OVERLOAD(std::vector<int>, KNN, predict, data);
    }

    bool save_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, KNN, save_model, filepath);
    }

    bool load_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, KNN, load_model, filepath);
    }

    std::vector<float> get_weights() const override {
        PYBIND11_OVERLOAD(std::vector<float>, KNN, get_weights);
    }

    void set_weights(const std::vector<float>& weights) override {
        PYBIND11_OVERLOAD(void, KNN, set_weights, weights);
    }
};

class PyCNN : public edge_ai::CNN {
public:
    using CNN::CNN;

    bool train(const edge_ai::Dataset& data, const edge_ai::TrainingConfig& config = {}) override {
        PYBIND11_OVERLOAD(bool, CNN, train, data, config);
    }

    std::vector<int> predict(const std::vector<std::vector<float>>& data) override {
        PYBIND11_OVERLOAD(std::vector<int>, CNN, predict, data);
    }

    bool save_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, CNN, save_model, filepath);
    }

    bool load_model(const std::string& filepath) override {
        PYBIND11_OVERLOAD(bool, CNN, load_model, filepath);
    }

    std::vector<float> get_weights() const override {
        PYBIND11_OVERLOAD(std::vector<float>, CNN, get_weights);
    }

    void set_weights(const std::vector<float>& weights) override {
        PYBIND11_OVERLOAD(void, CNN, set_weights, weights);
    }
};

PYBIND11_MODULE(edge_ai_python, m) {
    m.doc() = "EdgeAI C++ Machine Learning Library Python Wrapper";

    py::class_<edge_ai::Dataset>(m, "Dataset")
        .def(py::init<>())
        .def_readwrite("X", &edge_ai::Dataset::X)
        .def_readwrite("y", &edge_ai::Dataset::y)
        .def_readwrite("num_samples", &edge_ai::Dataset::num_samples)
        .def_readwrite("num_features", &edge_ai::Dataset::num_features);

    py::class_<edge_ai::TrainingConfig>(m, "TrainingConfig")
        .def(py::init<>())
        .def_readwrite("learning_rate", &edge_ai::TrainingConfig::learning_rate)
        .def_readwrite("max_epochs", &edge_ai::TrainingConfig::max_epochs)
        .def_readwrite("tolerance", &edge_ai::TrainingConfig::tolerance)
        .def_readwrite("verbose", &edge_ai::TrainingConfig::verbose);

    py::class_<edge_ai::QuantizationConfig>(m, "QuantizationConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &edge_ai::QuantizationConfig::enabled)
        .def_readwrite("scale", &edge_ai::QuantizationConfig::scale)
        .def_readwrite("zero_point", &edge_ai::QuantizationConfig::zero_point);

    py::enum_<edge_ai::ModelType>(m, "ModelType")
        .value("LOGISTIC_REGRESSION", edge_ai::ModelType::LOGISTIC_REGRESSION)
        .value("DECISION_TREE", edge_ai::ModelType::DECISION_TREE)
        .value("KNN", edge_ai::ModelType::KNN)
        .value("CNN", edge_ai::ModelType::CNN);

    py::class_<PyBaseModel, edge_ai::BaseModel>(m, "BaseModel")
        .def("get_model_type", &edge_ai::BaseModel::get_model_type)
        .def("is_quantized", &edge_ai::BaseModel::is_quantized)
        .def("quantize", &edge_ai::BaseModel::quantize)
        .def("dequantize", &edge_ai::BaseModel::dequantize)
        .def("quantize_model", &edge_ai::BaseModel::quantize_model)
        .def("dequantize_model", &edge_ai::BaseModel::dequantize_model)
        .def("calculate_accuracy", &edge_ai::BaseModel::calculate_accuracy);

    py::class_<PyLogisticRegression, PyBaseModel>(m, "LogisticRegression")
        .def(py::init<int, int>(), py::arg("input_dim"), py::arg("num_classes") = 2)
        .def("get_input_dim", &edge_ai::LogisticRegression::get_input_dim)
        .def("get_num_classes", &edge_ai::LogisticRegression::get_num_classes);

    py::class_<PyDecisionTree, PyBaseModel>(m, "DecisionTree")
        .def(py::init<int, int>(), py::arg("max_depth") = 10, py::arg("min_samples_split") = 2)
        .def("get_max_depth", &edge_ai::DecisionTree::get_max_depth)
        .def("get_min_samples_split", &edge_ai::DecisionTree::get_min_samples_split);

    py::class_<PyKNN, PyBaseModel>(m, "KNN")
        .def(py::init<int>(), py::arg("k") = 5)
        .def("set_k", &edge_ai::KNN::set_k)
        .def("get_k", &edge_ai::KNN::get_k)
        .def("set_metric", &edge_ai::KNN::set_metric)
        .def("get_metric", &edge_ai::KNN::get_metric);

    py::class_<PyCNN, PyBaseModel>(m, "CNN")
        .def(py::init<const edge_ai::CNNConfig&>(), py::arg("config"));

    py::module_ quantizer = m.def_submodule("quantizer", "Quantization utilities");

    quantizer.def("quantize_float32", &edge_ai::Quantizer::quantize_float32,
                  py::arg("data"), py::arg("scale"), py::arg("zero_point"));

    quantizer.def("dequantize_int8", &edge_ai::Quantizer::dequantize_int8,
                  py::arg("data"), py::arg("scale"), py::arg("zero_point"));

    quantizer.def("calculate_quantization_params", &edge_ai::Quantizer::calculate_quantization_params,
                  py::arg("data"));

    quantizer.def("calculate_scale_and_zero_point", &edge_ai::Quantizer::calculate_scale_and_zero_point,
                  py::arg("data"), py::arg("qmin") = -128, py::arg("qmax") = 127);

    quantizer.def("prune_weights", &edge_ai::Quantizer::prune_weights,
                  py::arg("weights"), py::arg("threshold"));

    quantizer.def("quantize_model_weights", &edge_ai::Quantizer::quantize_model_weights,
                  py::arg("weights"), py::arg("quantized_weights"), py::arg("scale"), py::arg("zero_point"));

    quantizer.def("dequantize_model_weights", &edge_ai::Quantizer::dequantize_model_weights,
                  py::arg("quantized_weights"), py::arg("weights"), py::arg("scale"), py::arg("zero_point"));

    py::module_ factory = m.def_submodule("factory", "Model factory utilities");

    factory.def("create_logistic_regression", &edge_ai::ModelFactory::create_logistic_regression,
                py::arg("input_dim"), py::arg("num_classes") = 2,
                py::return_value_policy::move);

    factory.def("create_decision_tree", &edge_ai::ModelFactory::create_decision_tree,
                py::arg("max_depth") = 10, py::arg("min_samples_split") = 2,
                py::return_value_policy::move);

    factory.def("create_knn", &edge_ai::ModelFactory::create_knn,
                py::arg("k") = 5, py::return_value_policy::move);

    factory.def("create_cnn", &edge_ai::ModelFactory::create_cnn, py::return_value_policy::move);

    factory.def("create_model", &edge_ai::ModelFactory::create_model,
                py::arg("type"), py::arg("params") = std::vector<int>(),
                py::return_value_policy::move);

    factory.def("string_to_model_type", &edge_ai::ModelFactory::string_to_model_type,
                py::arg("type_str"));

    factory.def("model_type_to_string", &edge_ai::ModelFactory::model_type_to_string,
                py::arg("type"));

    m.def("calculate_accuracy", &edge_ai::BaseModel::calculate_accuracy,
          py::arg("true_labels"), py::arg("predicted_labels"));
}