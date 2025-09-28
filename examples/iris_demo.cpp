#include "../src/include/model_base.h"
#include "../src/ml/logistic_regression.h"
#include "../src/ml/decision_tree.h"
#include "../src/ml/knn.h"
#include "../src/ml/cnn.h"
#include "../src/utils/model_factory.h"
#include "../src/utils/quantization.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>

class IrisDatasetDemo {
public:
    IrisDatasetDemo() {}

    edge_ai::Dataset load_iris_dataset() {
        edge_ai::Dataset data;
        data.X = {
            {5.1f, 3.5f, 1.4f, 0.2f},
            {4.9f, 3.0f, 1.4f, 0.2f},
            {4.7f, 3.2f, 1.3f, 0.2f},
            {4.6f, 3.1f, 1.5f, 0.2f},
            {5.0f, 3.6f, 1.4f, 0.2f},
            {5.4f, 3.9f, 1.7f, 0.4f},
            {4.6f, 3.4f, 1.4f, 0.3f},
            {5.0f, 3.4f, 1.5f, 0.2f},
            {4.4f, 2.9f, 1.4f, 0.2f},
            {4.9f, 3.1f, 1.5f, 0.1f},
            {7.0f, 3.2f, 4.7f, 1.4f},
            {6.4f, 3.2f, 4.5f, 1.5f},
            {6.9f, 3.1f, 4.9f, 1.5f},
            {5.5f, 2.3f, 4.0f, 1.3f},
            {6.5f, 2.8f, 4.6f, 1.5f},
            {5.7f, 2.8f, 4.5f, 1.3f},
            {6.3f, 3.3f, 4.7f, 1.6f},
            {4.9f, 2.4f, 3.3f, 1.0f},
            {6.6f, 2.9f, 4.6f, 1.3f},
            {5.2f, 2.7f, 3.9f, 1.4f},
            {6.3f, 3.3f, 6.0f, 2.5f},
            {5.8f, 2.7f, 5.1f, 1.9f},
            {7.1f, 3.0f, 5.9f, 2.1f},
            {6.3f, 2.9f, 5.6f, 1.8f},
            {6.5f, 3.0f, 5.8f, 2.2f},
            {7.6f, 3.0f, 6.6f, 2.1f},
            {4.9f, 2.5f, 4.5f, 1.7f},
            {7.3f, 2.9f, 6.3f, 1.8f},
            {6.7f, 2.5f, 5.8f, 1.8f},
            {7.2f, 3.6f, 6.1f, 2.5f}
        };

        data.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

        data.num_samples = 30;
        data.num_features = 4;

        return data;
    }

    void run() {
        auto iris_data = load_iris_dataset();
        auto test_data = std::vector<std::vector<float>>({
            {5.1f, 3.5f, 1.4f, 0.2f},
            {6.3f, 3.3f, 4.7f, 1.6f},
            {6.3f, 3.3f, 6.0f, 2.5f}
        });

        std::vector<std::string> model_names = {
            "Logistic Regression",
            "Decision Tree",
            "k-Nearest Neighbors",
            "CNN"
        };

        std::vector<std::unique_ptr<edge_ai::BaseModel>> models;

        models.push_back(ModelFactory::create_logistic_regression(4, 3));
        models.push_back(ModelFactory::create_decision_tree(5, 2));
        models.push_back(ModelFactory::create_knn(5, 4));

        edge_ai::Dataset cnn_data;
        cnn_data.X = iris_data.X;
        cnn_data.y = iris_data.y;
        cnn_data.num_samples = iris_data.num_samples;
        cnn_data.num_features = iris_data.num_features;

        auto cnn_model = ModelFactory::create_cnn();
        cnn_model->train(cnn_data);
        models.push_back(std::move(cnn_model));

        std::cout << "=== Iris Dataset Classification Demo ===" << std::endl;
        std::cout << "Training and testing various ML algorithms..." << std::endl << std::endl;

        for (size_t i = 0; i < models.size(); ++i) {
            std::cout << "=== " << model_names[i] << " ===" << std::endl;

            auto start = std::chrono::high_resolution_clock::now();

            edge_ai::TrainingConfig config;
            config.learning_rate = 0.01f;
            config.max_epochs = 100;
            config.verbose = false;

            models[i]->train(iris_data, config);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            auto predictions = models[i]->predict(test_data);
            std::vector<int> true_labels = {0, 1, 2};

            float accuracy = edge_ai::BaseModel::calculate_accuracy(true_labels, predictions);

            std::cout << "Training time: " << duration.count() << " ms" << std::endl;
            std::cout << "Test predictions: ";
            for (int pred : predictions) {
                std::cout << pred << " ";
            }
            std::cout << std::endl;

            std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
            std::cout << std::endl;

            models[i]->save_model("model_" + std::to_string(i) + ".bin");
            std::cout << "Model saved as model_" + std::to_string(i) + ".bin" << std::endl;

            auto weights = models[i]->get_weights();
            std::cout << "Model size (weights): " << weights.size() << std::endl;

            auto original_weights = weights;
            float original_memory = weights.size() * sizeof(float);

            edge_ai::QuantizationConfig quant_config;
            quant_config.enabled = true;
            quant_config.scale = 0.01f;
            quant_config.zero_point = 0;

            models[i]->quantize(quant_config);

            auto quantized_weights = models[i]->get_weights();
            float quantized_memory = quantized_weights.size() * sizeof(int8_t);

            float compression_ratio = original_memory / quantized_memory;
            std::cout << "Quantization memory usage ratio: " << compression_ratio << "x" << std::endl;

            models[i]->dequantize();

            models[i]->quantize_model();
            models[i]->dequantize_model();

            std::cout << std::endl;
        }

        std::cout << "=== Performance Benchmark ===" << std::endl;
        benchmark_inference_speed(models);
    }

private:
    void benchmark_inference_speed(const std::vector<std::unique_ptr<edge_ai::BaseModel>>& models) {
        auto test_data = std::vector<std::vector<float>>({
            {5.1f, 3.5f, 1.4f, 0.2f},
            {6.3f, 3.3f, 4.7f, 1.6f},
            {6.3f, 3.3f, 6.0f, 2.5f}
        });

        for (size_t i = 0; i < models.size(); ++i) {
            int num_iterations = 1000;
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iterations; ++iter) {
                models[i]->predict(test_data);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            float avg_inference_time_us = static_cast<float>(duration.count()) / num_iterations;
            float avg_inference_time_ms = avg_inference_time_us / 1000.0f;

            std::cout << "Model " << i << " avg inference time: " << avg_inference_time_ms << " ms" << std::endl;
            std::cout << "Model " << i << " throughput: " << (1000000.0f / avg_inference_time_us) << " fps" << std::endl;
        }
    }
};

int main() {
    try {
        IrisDatasetDemo demo;
        demo.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}