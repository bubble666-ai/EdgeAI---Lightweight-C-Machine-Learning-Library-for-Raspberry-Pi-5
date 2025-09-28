// Simple validation script to check code structure
#include <iostream>
#include <fstream>
#include <string>

bool check_file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

bool check_syntax_validation(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return false;

    std::string line;
    int brace_count = 0;
    int bracket_count = 0;
    bool in_comment = false;

    while (std::getline(file, line)) {
        // Simple brace counting validation
        for (char c : line) {
            if (c == '{') brace_count++;
            if (c == '}') brace_count--;
            if (c == '[') bracket_count++;
            if (c == ']') bracket_count--;
        }

        // Check for obvious syntax errors
        if (line.find(";;") != std::string::npos) {
            std::cout << "Error: Double semicolon in " << filename << std::endl;
            return false;
        }
        if (line.find("}") != std::string::npos && brace_count < 0) {
            std::cout << "Error: Unmatched closing brace in " << filename << std::endl;
            return false;
        }
    }

    return brace_count == 0 && bracket_count == 0;
}

int main() {
    std::cout << "=== EdgeAI Code Validation ===" << std::endl;

    std::vector<std::string> required_files = {
        "src/include/model_base.h",
        "src/include/model_base.cpp",
        "src/ml/logistic_regression.h",
        "src/ml/logistic_regression.cpp",
        "src/ml/decision_tree.h",
        "src/ml/decision_tree.cpp",
        "src/ml/knn.h",
        "src/ml/knn.cpp",
        "src/ml/cnn.h",
        "src/ml/cnn.cpp",
        "src/utils/quantization.h",
        "src/utils/quantization.cpp",
        "src/utils/model_factory.h",
        "src/utils/model_factory.cpp",
        "examples/face_lock.cpp",
        "examples/iris_demo.cpp",
        "tests/test_algorithms.cpp",
        "CMakeLists.txt",
        "README.md"
    };

    bool all_files_exist = true;
    for (const auto& file : required_files) {
        if (!check_file_exists(file)) {
            std::cout << "✗ Missing file: " << file << std::endl;
            all_files_exist = false;
        } else {
            std::cout << "✓ Found file: " << file << std::endl;
        }
    }

    std::cout << "\n=== Syntax Validation ===" << std::endl;

    bool syntax_valid = true;
    for (const auto& file : required_files) {
        if (file.find(".h") != std::string::npos || file.find(".cpp") != std::string::npos) {
            if (check_syntax_validation(file)) {
                std::cout << "✓ Syntax OK: " << file << std::endl;
            } else {
                std::cout << "✗ Syntax error in: " << file << std::endl;
                syntax_valid = false;
            }
        }
    }

    if (all_files_exist && syntax_valid) {
        std::cout << "\n✓ All files present and syntax is valid!" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ Some files are missing or have syntax errors!" << std::endl;
        return 1;
    }
}