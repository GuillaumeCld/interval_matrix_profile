#include <iostream>
#include <vector>
#include <chrono> // For timing
#include <random> // For random number generation
#include <algorithm>
#include <cmath>
#include <matrix_profile.hpp>
#include <utils.hpp>

double vector_norm(const std::vector<double>& vec) {
    double norm = 0.0;
    for (const auto& elem : vec) {
        norm += elem * elem;
    }
    return std::sqrt(norm);
}



double compute_vector_difference_norm(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cerr << "Error: Vectors have different sizes." << std::endl;
        return -1.0; // Error code
    }
    std::vector<double> difference(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        difference[i] = vec1[i] - vec2[i];
    }
    return vector_norm(difference);
}

void compareStumpy(){
  std::vector<double> ts;
  readFile<double>("Data/ts_sst.txt", ts, "%f");
  std::vector<double> mp_ref;
  readFile<double>("Data/mp_sst.txt", mp_ref, "%f");

  int window_size = 7;
  auto mpOutput = computeMatrixProfileSTOMP(ts, window_size);
  std::vector<double> matrix_profile_stomp = std::get<0>(mpOutput);

  double difference_norm = compute_vector_difference_norm(mp_ref, matrix_profile_stomp);
  std::cout << "Norm of the difference: " << difference_norm << std::endl;

}

// Function to test computation speed of matrix profile
void testMatrixProfileComputationSpeed(int vector_size, int window_size) {
    // Generate random vector

    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    std::cout << "Random gen" << std::endl;

    for (int i = 0; i < vector_size; ++i) {
        data.push_back(dis(gen));
    }
    std::cout << "Brute Force" << std::endl;

    // Measure computation time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mpOutput = computeMatrixProfileBruteForce(data, window_size);
    std::vector<double> matrix_profile = std::get<0>(mpOutput);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Compute duration
    std::chrono::duration<double> duration = end_time - start_time;

    // Output results
    std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
    std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;
    
    std::cout << "STOMP" << std::endl;

    // Measure computation time
    start_time = std::chrono::high_resolution_clock::now();
    mpOutput = computeMatrixProfileSTOMP(data, window_size);
    std::vector<double> matrix_profile_stomp = std::get<0>(mpOutput);
    end_time = std::chrono::high_resolution_clock::now();

    // Compute duration
    duration = end_time - start_time;

    // Output results
    std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
    std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;  
  
    if (std::equal(matrix_profile.begin(), matrix_profile.end(), matrix_profile_stomp.begin()))
      std::cout << "success" << std::endl;
    double difference_norm = compute_vector_difference_norm(matrix_profile, matrix_profile_stomp);
    std::cout << "Norm of the difference: " << difference_norm << std::endl;
    
    std::vector<double>::iterator result;
    result = std::max_element(matrix_profile.begin(), matrix_profile.end());
    std::cout << " max " << *result << '\n';
    result = std::max_element(matrix_profile_stomp.begin(), matrix_profile_stomp.end());
    std::cout << " max " << *result << '\n';

    result = std::min_element(matrix_profile.begin(), matrix_profile.end());
    std::cout << " min " << *result << '\n';
    result = std::min_element(matrix_profile_stomp.begin(), matrix_profile_stomp.end());
    std::cout << " min " << *result << '\n';
  }


int main() {
  compareStumpy();

  // testMatrixProfileComputationSpeed(20000, 64);

  return 0;
}
