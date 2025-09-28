#include "../src/include/model_base.h"
#include "../src/ml/logistic_regression.h"
#include "../src/ml/decision_tree.h"
#include "../src/ml/knn.h"
#include "../src/ml/cnn.h"
#include "../src/utils/model_factory.h"
#include "../src/utils/quantization.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

class TestAlgorithms {
public:
    TestAlgorithms() {}

    void run_all_tests() {
        std::cout << "=== Running ML Algorithm Tests ===" << std::endl;

        test_logistic_regression();
        test_decision_tree();
        test_knn();
        test_cnn();
        test_quantization();
        test_model_factory();
        test_model_persistence();

        std::cout << "=== All tests passed! ===" << std::endl;
    }

private:
    void test_logistic_regression() {
        std::cout << "Testing Logistic Regression..." << std::endl;

        edge_ai::Dataset data;
        data.X = {{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f}};
        data.y = {0, 0, 1, 1};
        data.num_samples = 4;
        data.num_features = 2;

        auto model = std::make_unique<edge_ai::LogisticRegression>(2, 2);

        edge_ai::TrainingConfig config;
        config.learning_rate = 0.01f;
        config.max_epochs = 1000;
        config.verbose = false;

        bool trained = model->train(data, config);
        assert(trained);

        auto test_data = std::vector<std::vector<float>>({{1.5f, 2.5f}, {3.5f, 4.5f}});
        auto predictions = model->predict(test_data);

        assert(predictions.size() == 2);

        auto weights = model->get_weights();
        assert(!weights.empty());

        std::cout << "Logistic Regression test passed!" << std::endl;
    }

    void test_decision_tree() {
        std::cout << "Testing Decision Tree..." << std::endl;

        edge_ai::Dataset data;
        data.X = {{1.0f}, {2.0f}, {3.0f}, {4.0f}};
        data.y = {0, 0, 1, 1};
        data.num_samples = 4;
        data.num_features = 1;

        auto model = std::make_unique<edge_ai::DecisionTree>(3, 1);

        bool trained = model->train(data);
        assert(trained);

        auto test_data = std::vector<std::vector<float>>({{1.5f}, {3.5f}});
        auto predictions = model->predict(test_data);

        assert(predictions.size() == 2);

        auto weights = model->get_weights();
        assert(!weights.empty());

        std::cout << "Decision Tree test passed!" << std::endl;
    }

    void test_knn() {
        std::cout << "Testing k-Nearest Neighbors..." << std::endl;

        edge_ai::Dataset data;
        data.X = {
            {1.0f, 2.0f},
            {2.0f, 3.0f},
            {3.0f, 4.0f},
            {4.0f, 5.0f}
        };
        data.y = {0, 0, 1, 1};
        data.num_samples = 4;
        data.num_features = 2;

        auto model = std::make_unique<edge_ai::KNN>(3);

        bool trained = model->train(data);
        assert(trained);

        auto test_data = std::vector<std::vector<float>>({{1.5f, 2.5f}, {3.5f, 4.5f}});
        auto predictions = model->predict(test_data);

        assert(predictions.size() == 2);

        auto weights = model->get_weights();
        assert(weights.size() >= 2);

        model->set_k(5);
        assert(model->get_k() == 5);

        std::cout << "k-Nearest Neighbors test passed!" << std::endl;
    }

    void test_cnn() {
        std::cout << "Testing CNN..." << std::endl;

        edge_ai::CNNConfig config;
        config.conv_layers.push_back({3, 8, 3, 1, 0});
        config.dense_layers.push_back({8 * 28 * 28, 10});

        auto model = std::make_unique<edge_ai::CNN>(config);

        edge_ai::Dataset data;
        data.X = std::vector<std::vector<float>>(10, std::vector<float>(28 * 28 * 3, 0.1f));
        data.y = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
        data.num_samples = 10;
        data.num_features = 28 * 28 * 3;

        bool trained = model->train(data);
        assert(trained);

        auto test_data = std::vector<std::vector<float>>({
            std::vector<float>(28 * 28 * 3, 0.2f)
        });
        auto predictions = model->predict(test_data);

        assert(predictions.size() == 1);

        auto weights = model->get_weights();
        assert(!weights.empty());

        std::cout << "CNN test passed!" << std::endl;
    }

    void test_quantization() {
        std::cout << "Testing Quantization..." << std::endl;

        std::vector<float> test_data = {0.1f, 0.5f, -0.2f, 0.8f, -0.4f, 0.6f};

        auto [scale, zero_point] = edge_ai::Quantizer::calculate_quantization_params(test_data);

        auto quantized = edge_ai::Quantizer::quantize_float32(test_data, scale, zero_point);
        auto dequantized = edge_ai::Quantizer::dequantize_int8(quantized, scale, zero_point);

        assert(quantized.size() == test_data.size());
        assert(dequantized.size() == test_data.size());

        for (size_t i = 0; i < test_data.size(); ++i) {
            assert(std::abs(dequantized[i] - test_data[i]) < 0.01f);
        }

        std::vector<float> weights = {0.01f, -0.02f, 0.03f, 0.0f, 0.05f, -0.04f};
        float threshold = 0.02f;
        edge_ai::Quantizer::prune_weights(weights, threshold);

        for (float weight : weights) {
            assert(std::abs(weight) >= threshold || weight == 0.0f);
        }

        std::cout << "Quantization test passed!" << std::endl;
    }

    void test_model_factory() {
        std::cout << "Testing Model Factory..." << std::endl;

        auto lr_model = ModelFactory::create_logistic_regression(4, 3);
        assert(lr_model->get_model_type() == edge_ai::ModelType::LOGISTIC_REGRESSION);

        auto dt_model = ModelFactory::create_decision_tree(5, 2);
        assert(dt_model->get_model_type() == edge_ai::ModelType::DECISION_TREE);

        auto knn_model = ModelFactory::create_knn(7);
        assert(knn_model->get_model_type() == edge_ai::ModelType::KNN);

        auto cnn_model = ModelFactory::create_cnn();
        assert(cnn_model->get_model_type() == edge_ai::ModelType::CNN);

        auto type = ModelFactory::string_to_model_type("logreg");
        assert(type == edge_ai::ModelType::LOGISTIC_REGRESSION);

        auto type_str = ModelFactory::model_type_to_string(edge_ai::ModelType::DECISION_TREE);
        assert(type_str == "decision_tree");

        std::cout << "Model Factory test passed!" << std::endl;
    }

    void test_model_persistence() {
        std::cout << "Testing Model Persistence..." << std::endl;

        auto model = ModelFactory::create_logistic_regression(2, 2);

        edge_ai::Dataset data;
        data.X = {{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f}};
        data.y = {0, 0, 1, 1};
        data.num_samples = 4;
        data.num_features = 2;

        model->train(data);

        bool saved = model->save_model("test_model.bin");
        assert(saved);

        auto model2 = ModelFactory::create_logistic_regression(2, 2);
        bool loaded = model2->load_model("test_model.bin");
        assert(loaded);

        auto test_data = std::vector<std::vector<float>>({{1.5f, 2.5f}});
        auto predictions1 = model->predict(test_data);
        auto predictions2 = model2->predict(test_data);

        assert(predictions1 == predictions2);

        std::cout << "Model Persistence test passed!" << std::endl;
    }
};

int main() {
    try {
        TestAlgorithms test;
        test.run_all_tests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}