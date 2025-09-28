#include "../src/include/model_base.h"
#include "../src/ml/cnn.h"
#include "../src/utils/model_factory.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <wiringPi.h>

class FaceLockSystem {
public:
    FaceLockSystem() {
        if (wiringPiSetup() == -1) {
            std::cerr << "Failed to setup wiringPi" << std::endl;
        }

        pinMode(2, OUTPUT);
        digitalWrite(2, LOW);
    }

    void run() {
        auto model = ModelFactory::create_cnn();

        CNNConfig config;
        config.conv_layers.push_back({3, 16, 3, 1, 0});
        config.conv_layers.push_back({16, 32, 3, 1, 0});
        config.dense_layers.push_back({32 * 30 * 30, 128});
        config.dense_layers.push_back({128, 2});

        CNN cnn_model(config);
        model = std::make_unique<CNN>(config);

        TrainingConfig train_config;
        train_config.learning_rate = 0.001f;
        train_config.max_epochs = 100;
        train_config.verbose = true;

        edge_ai::Dataset face_data;
        face_data.X = {
            {0.1f, 0.9f, 0.1f, 0.9f, 0.8f},
            {0.9f, 0.1f, 0.9f, 0.1f, 0.8f},
            {0.2f, 0.8f, 0.3f, 0.7f, 0.9f},
            {0.8f, 0.2f, 0.7f, 0.3f, 0.9f}
        };
        face_data.y = {1, 0, 1, 0};
        face_data.num_samples = 4;
        face_data.num_features = 5;

        std::cout << "Training face detection model..." << std::endl;
        model->train(face_data, train_config);

        model->save_model("face_detection_model.bin");
        std::cout << "Model saved as face_detection_model.bin" << std::endl;

        std::cout << "Starting face lock system..." << std::endl;

        while (true) {
            detect_face_and_control_lock(model);
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

private:
    void detect_face_and_control_lock(const std::unique_ptr<edge_ai::BaseModel>& model) {
        auto test_image = std::vector<std::vector<float>>(1, std::vector<float>(5, 0.5f));

        auto predictions = model->predict(test_image);
        bool face_detected = predictions[0] == 1;

        std::cout << "Face detected: " << (face_detected ? "Yes" : "No") << std::endl;

        if (face_detected) {
            std::cout << "Access Granted" << std::endl;
            digitalWrite(2, HIGH);
            std::this_thread::sleep_for(std::chrono::seconds(1));
            digitalWrite(2, LOW);
        } else {
            std::cout << "Access Denied" << std::endl;
            digitalWrite(2, LOW);
        }
    }
};

int main() {
    try {
        FaceLockSystem system;
        system.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}