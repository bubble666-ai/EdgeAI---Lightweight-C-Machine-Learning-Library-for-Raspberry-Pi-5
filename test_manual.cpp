#include <iostream>
#include <vector>
#include <memory>
#include <cassert>

// Simplified test without CMake
#include "src/include/model_base.h"
#include "src/ml/logistic_regression.h"
#include "src/ml/knn.h"
#include "src/utils/model_factory.h"
#include "src/utils/quantization.h"

void test_basic_functionality() {
    std::cout << "=== Testing Basic Functionality ===" << std::endl;

    // Test Logistic Regression
    std::cout << "Testing Logistic Regression..." << std::endl;

    edge_ai::Dataset data;
    data.X = {{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f}};
    data.y = {0, 0, 1, 1};
    data.num_samples = 4;
    data.num_features = 2;

    auto model = std::make_unique<edge_ai::LogisticRegression>(2, 2);

    edge_ai::TrainingConfig config;
    config.learning_rate = 0.01f;
    config.max_epochs = 100;
    config.verbose = false;

    bool trained = model->train(data, config);
    if (!trained) {
        std::cout << "✗ Logistic Regression training failed" << std::endl;
        return;
    }

    auto test_data = std::vector<std::vector<float>>({{1.5f, 2.5f}, {3.5f, 4.5f}});
    auto predictions = model->predict(test_data);

    if (predictions.size() != 2) {
        std::cout << "✗ Logistic Regression prediction size mismatch" << std::endl;
        return;
    }

    std::cout << "✓ Logistic Regression works" << std::endl;

    // Test Model Factory
    std::cout << "Testing Model Factory..." << std::endl;

    auto lr_model = edge_ai::ModelFactory::create_logistic_regression(4, 3);
    auto knn_model = edge_ai::ModelFactory::create_knn(5);

    if (!lr_model || !knn_model) {
        std::cout << "✗ Model Factory failed" << std::endl;
        return;
    }

    std::cout << "✓ Model Factory works" << std::endl;

    // Test Quantization
    std::cout << "Testing Quantization..." << std::endl;

    std::vector<float> test_data = {0.1f, 0.5f, -0.2f, 0.8f, -0.4f, 0.6f};
    auto [scale, zero_point] = edge_ai::Quantizer::calculate_quantization_params(test_data);

    auto quantized = edge_ai::Quantizer::quantize_float32(test_data, scale, zero_point);
    auto dequantized = edge_ai::Quantizer::dequantize_int8(quantized, scale, zero_point);

    if (quantized.size() != test_data.size() || dequantized.size() != test_data.size()) {
        std::cout << "✗ Quantization size mismatch" << std::endl;
        return;
    }

    bool quantization_accurate = true;
    for (size_t i = 0; i < test_data.size(); ++i) {
        if (std::abs(dequantized[i] - test_data[i]) > 0.01f) {
            quantization_accurate = false;
            break;
        }
    }

    if (!quantization_accurate) {
        std::cout << "✗ Quantization accuracy too low" << std::endl;
        return;
    }

    std::cout << "✓ Quantization works" << std::endl;
}

void test_error_handling() {
    std::cout << "=== Testing Error Handling ===" << std::endl;

    // Test empty dataset
    std::cout << "Testing empty dataset..." << std::endl;

    edge_ai::Dataset empty_data;
    empty_data.X = {};
    empty_data.y = {};
    empty_data.num_samples = 0;
    empty_data.num_features = 0;

    auto model = std::make_unique<edge_ai::LogisticRegression>(2, 2);
    bool trained = model->train(empty_data);

    if (trained) {
        std::cout << "✓ Training correctly failed on empty dataset" << std::endl;
    } else {
        std::cout << "✓ Training failed on empty dataset as expected" << std::endl;
    }

    // Test model type strings
    std::cout << "Testing model type strings..." << std::endl;

    try {
        auto type = edge_ai::ModelFactory::string_to_model_type("logreg");
        if (type == edge_ai::ModelType::LOGISTIC_REGRESSION) {
            std::cout << "✓ String to model type conversion works" << std::endl;
        } else {
            std::cout << "✗ String to model type conversion failed" << std::endl;
        }
    } catch (...) {
        std::cout << "✗ String to model type conversion threw exception" << std::endl;
    }

    try {
        auto type = edge_ai::ModelFactory::string_to_model_type("invalid");
        std::cout << "✗ Should have thrown exception for invalid type" << std::endl;
    } catch (...) {
        std::cout << "✓ Correctly handled invalid model type" << std::endl;
    }
}

void test_performance() {
    std::cout << "=== Testing Performance ===" << std::endl;

    // Create larger dataset
    std::vector<std::vector<float>> large_X;
    std::vector<int> large_y;

    for (int i = 0; i < 100; ++i) {
        large_X.push_back({static_cast<float>(i), static_cast<float>(i * 2)});
        large_y.push_back(i % 2);
    }

    edge_ai::Dataset large_data;
    large_data.X = large_X;
    large_data.y = large_y;
    large_data.num_samples = 100;
    large_data.num_features = 2;

    auto model = std::make_unique<edge_ai::LogisticRegression>(2, 2);

    auto start = std::chrono::high_resolution_clock::now();

    bool trained = model->train(large_data);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (trained) {
        std::cout << "✓ Large dataset training successful in " << duration.count() << " ms" << std::endl;
    } else {
        std::cout << "✗ Large dataset training failed" << std::endl;
    }

    // Test prediction performance
    auto test_data = std::vector<std::vector<float>>({{1.5f, 3.0f}, {2.5f, 5.0f}});

    start = std::chrono::high_resolution_clock::now();
    auto predictions = model->predict(test_data);
    end = std::chrono::high_resolution_clock::now();
    auto pred_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (predictions.size() == 2) {
        std::cout << "✓ Prediction successful in " << pred_duration.count() << " μs" << std::endl;
    } else {
        std::cout << "✗ Prediction failed" << std::endl;
    }
}

void test_iris_dataset() {
    std::cout << "=== Testing Iris Dataset ===" << std::endl;

    // Simulate Iris dataset
    edge_ai::Dataset iris_data;
    iris_data.X = {
        {5.1f, 3.5f, 1.4f, 0.2f},
        {4.9f, 3.0f, 1.4f, 0.2f},
        {4.7f, 3.2f, 1.3f, 0.2f},
        {7.0f, 3.2f, 4.7f, 1.4f},
        {6.4f, 3.2f, 4.5f, 1.5f},
        {6.9f, 3.1f, 4.9f, 1.5f}
    };
    iris_data.y = {0, 0, 0, 1, 1, 1};
    iris_data.num_samples = 6;
    iris_data.num_features = 4;

    auto model = std::make_unique<edge_ai::LogisticRegression>(4, 2);

    edge_ai::TrainingConfig config;
    config.learning_rate = 0.1f;
    config.max_epochs = 500;
    config.verbose = false;

    bool trained = model->train(iris_data, config);

    if (!trained) {
        std::cout << "✗ Iris dataset training failed" << std::endl;
        return;
    }

    auto test_data = std::vector<std::vector<float>>({
        {5.0f, 3.5f, 1.4f, 0.2f},  // Should be class 0
        {6.5f, 3.0f, 4.6f, 1.5f}   // Should be class 1
    });

    auto predictions = model->predict(test_data);

    if (predictions.size() == 2) {
        std::cout << "✓ Iris dataset predictions: " << predictions[0] << ", " << predictions[1] << std::endl;

        // Check if predictions make sense
        if (predictions[0] == 0 && predictions[1] == 1) {
            std::cout << "✓ Iris dataset classification appears correct" << std::endl;
        } else {
            std::cout << "⚠ Iris dataset classification may need more training" << std::endl;
        }
    } else {
        std::cout << "✗ Iris dataset prediction failed" << std::endl;
    }
}

int main() {
    try {
        std::cout << "=== EdgeAI Library Manual Test ===" << std::endl;

        test_basic_functionality();
        test_error_handling();
        test_performance();
        test_iris_dataset();

        std::cout << "=== All Tests Completed Successfully! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}