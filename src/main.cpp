#include <iostream>
#include <vector>
#include <chrono> // For timing
#include <random> // For random number generation
#include <algorithm>
#include <cmath>
#include <span>
#include <fstream>
#include <matrix_profile.hpp>
#include <utils.hpp>


using value_type= double;

void testDistance(){
  printf("testDistance\n");
    // Test case 1: Same vector
    std::array<double, 3> v1{1.0, 2.0, 3.0};
    double result1 = euclideanDistance(std::span<double, 3>(v1), std::span<double, 3>(v1));
    assert(result1 == 0.0);

    // Test case 2: Different vectors
    std::array<double, 3> v2{4.0, 5.0, 6.0};
    double result2 = euclideanDistance(std::span<double, 3>(v1), std::span<double, 3>(v2));
    assert(result2 == std::sqrt(27.0));

    // Test case 3: Zero vectors
    std::array<double, 3> v3{0.0, 0.0, 0.0};
    double result3 = euclideanDistance(std::span<double, 3>(v3), std::span<double, 3>(v3));
    assert(result3 == 0.0);

    // Test case 4: Symetric vectors 
    double result4 = euclideanDistance(std::span<double, 3>(v2), std::span<double, 3>(v1));
    assert(result4 == result2);

    std::cout << "All tests passed!\n";
}



void compareStumpy(){
  printf("compareStumpy\n");
  std::vector<double> ts;
  readFile<double>("../Data/ts_sst.txt", ts, "%lf");
  std::vector<double> mp_ref;
  readFile<double>("../Data/mp_sst.txt", mp_ref, "%lf");
  int window_size = 7;

  std:double norm_ref = vector_norm(mp_ref);

  printf("Brute Force\n");
  auto mpOutput = computeMatrixProfileBruteForce(ts, window_size);
  std::vector<double> matrix_profile_brute_force = std::get<0>(mpOutput);
  std::vector<int> index_profile_brute_force = std::get<1>(mpOutput);

  printf("STOMP\n");
  mpOutput = computeMatrixProfileSTOMP(ts, window_size);
  std::vector<double> matrix_profile_stomp = std::get<0>(mpOutput);
  std::vector<int> index_profile_stomp = std::get<1>(mpOutput);


  double difference_norm = compute_vector_difference_norm(mp_ref, matrix_profile_stomp);
  std::cout << "Norm of the difference with STOMP: " << difference_norm << std::endl;
  if (std::equal(mp_ref.begin(), mp_ref.end(), matrix_profile_stomp.begin())) {
    std::cout << " success" << std::endl;
  } else {
    std::cout << "failure" << std::endl;
  }
  difference_norm = compute_vector_difference_norm(mp_ref, matrix_profile_brute_force);
  std::cout << "Norm of the difference with BruteForce: " << difference_norm<< std::endl;
  if(std::equal(mp_ref.begin(), mp_ref.end(), matrix_profile_stomp.begin())) {
    std::cout << " success" << std::endl;
  } else {
    std::cout << "failure" << std::endl;
  }
    

  
  std::vector<double>::iterator max;
  std::vector<double>::iterator min;
  max = std::max_element(mp_ref.begin(), mp_ref.end());
  min = std::min_element(mp_ref.begin(), mp_ref.end());

  std::cout << "Ref: max " << *max << ", min " << *min << '\n';

  max = std::max_element(matrix_profile_stomp.begin(), matrix_profile_stomp.end());
  min = std::min_element(matrix_profile_stomp.begin(), matrix_profile_stomp.end());
  std::cout << "STOMP: max " << *max << ", min " << *min << '\n';

  max = std::max_element(matrix_profile_brute_force.begin(), matrix_profile_brute_force.end()); 
  min = std::min_element(matrix_profile_brute_force.begin(), matrix_profile_brute_force.end());
  std::cout << "BruteForce: max " << *max << ", min " << *min << '\n';


  max = std::max_element(ts.begin(), ts.end()); 
  min = std::min_element(ts.begin(), ts.end());
  std::cout << "ts: max " << *max << ", min " << *min << '\n';




  // Write the result to a file
  std::ofstream outputFile("../Data/matrix_profile_bf.txt");
  if (outputFile.is_open()) {
      for (const auto& value : matrix_profile_brute_force) {
          outputFile << value << "\n";
      }
      outputFile.close();
        std::cout << "Matrix profile has been written to matrix_profile.txt\n";
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }
  // Write the result to a file
  std::ofstream outputFile1("../Data/matrix_profile_stomp.txt");
  if (outputFile1.is_open()) {
      for (const auto& value : matrix_profile_stomp) {
          outputFile1 << value << "\n";
      }
      outputFile1.close();
        std::cout << "Matrix profile has been written to matrix_profile.txt\n";
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }
    std::ofstream outputFile2("../Data/index_profile_stomp.txt");
  if (outputFile2.is_open()) {
      for (const auto& value : index_profile_stomp) {
          outputFile2 << value << "\n";
      }
      outputFile2.close();
        std::cout << "Matrix profile has been written to matrix_profile.txt\n";
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }
    std::ofstream outputFile3("../Data/index_profile_bf.txt");
    if (outputFile3.is_open()) {
      for (const auto& value : index_profile_stomp) {
          outputFile3 << value << "\n";
      }
      outputFile3.close();
        std::cout << "Matrix profile has been written to matrix_profile.txt\n";
    } else {
        std::cerr << "Unable to open file for writing.\n";
    }
}

// Function to test computation speed of matrix profile
void testMatrixProfileComputationSpeed(int vector_size, int window_size) {
    // Generate random vector

    std::vector<value_type> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

    std::cout << "Random gen" << std::endl;

    for (int i = 0; i < vector_size; ++i) {
        data.push_back(dis(gen));
    }


    // std::cout << "Brute Force" << std::endl;

    // // Measure computation time
    // auto start_time = std::chrono::high_resolution_clock::now();
    // auto mpOutput = computeMatrixProfileBruteForce(data, window_size);
    // auto matrix_profile = std::get<0>(mpOutput);
    // auto end_time = std::chrono::high_resolution_clock::now();
    
    // // // Compute duration
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // // // Output results
    // std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
    // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;
    

    // std::cout << "Brute Force v2" << std::endl;

    // // Measure computation time
    // start_time = std::chrono::high_resolution_clock::now();
    // mpOutput = brute_force_v2(data, window_size);
    // matrix_profile = std::get<0>(mpOutput);
    // end_time = std::chrono::high_resolution_clock::now();
    
    // // // Compute duration
    // duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    // // // Output results
    // std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
    // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

    // std::cout << "STOMP" << std::endl;

    // // Measure computation time
    // auto start_time = std::chrono::high_resolution_clock::now();
    // auto mpOutput = computeMatrixProfileSTOMP(data, window_size);
    // std::vector<double> matrix_profile_stomp = std::get<0>(mpOutput);
    // auto end_time = std::chrono::high_resolution_clock::now();

    // // Compute duration
    // auto duration = end_time - start_time;

 
    // // Output results
    // std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
    // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;  
  

    // start_time = std::chrono::high_resolution_clock::now();
    // mpOutput = computeMatrixProfileSTOMP(data, window_size);
    // std::vector<double> matrix_profile_stompv2 = std::get<0>(mpOutput);
    // end_time = std::chrono::high_resolution_clock::now();
    // duration = end_time - start_time;
    // std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
    // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

    // std::cout << "BlockSTOMP" << std::endl;
    // // Measure computation time
    // start_time = std::chrono::high_resolution_clock::now();
    // mpOutput = blockSTOMP_v2(data, window_size, 5000, 5000);
    // std::vector<double> matrix_profile_blockstomp = std::get<0>(mpOutput);
    // end_time = std::chrono::high_resolution_clock::now();
    // // Compute duration
    // duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

    std::cout << "BlockSTOMP v2" << std::endl;
    // Measure computation time
    auto start_time = std::chrono::high_resolution_clock::now();
    auto mpOutput = blockSTOMP_v2(data, window_size, 5000, 5000);
    auto matrix_profile_blockstomp = std::get<0>(mpOutput);
    auto end_time = std::chrono::high_resolution_clock::now();
    // Compute duration
    auto duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

    std::cout << "BlockSTOMP v3" << std::endl;
    // Measure computation time
    start_time = std::chrono::high_resolution_clock::now();
    mpOutput = blockSTOMP_v3(data, window_size, 5000, 5000);
    matrix_profile_blockstomp = std::get<0>(mpOutput);
    end_time = std::chrono::high_resolution_clock::now();
    // Compute duration
    duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

    std::cout << "BlockSTOMP v4" << std::endl;
    // Measure computation time
    start_time = std::chrono::high_resolution_clock::now();
    mpOutput = blockSTOMP_v4(data, window_size, 5000, 5000);
    matrix_profile_blockstomp = std::get<0>(mpOutput);
    end_time = std::chrono::high_resolution_clock::now();
    // Compute duration
    duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

    std::cout << "BlockSTOMP v5" << std::endl;
    // Measure computation time
    start_time = std::chrono::high_resolution_clock::now();
    mpOutput = blockSTOMP_v5(data, window_size, 5000, 5000);
    matrix_profile_blockstomp = std::get<0>(mpOutput);
    end_time = std::chrono::high_resolution_clock::now();
    // Compute duration
    duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

  //   double difference_norm = compute_vector_difference_norm(matrix_profile, matrix_profile_stomp);
  //   std::cout << "Norm of the difference: " << difference_norm << std::endl;
  
  //   difference_norm = compute_vector_difference_norm(matrix_profile, matrix_profile_stompv2);
  //   std::cout << "Norm of the difference: " << difference_norm << std::endl;

  //   std::vector<double>::iterator result;
  //   result = std::max_element(matrix_profile.begin(), matrix_profile.end());
  //   std::cout << " max " << *result << '\n';
  //   result = std::max_element(matrix_profile_stomp.begin(), matrix_profile_stomp.end());
  //   std::cout << " max " << *result << '\n';

  //   result = std::min_element(matrix_profile.begin(), matrix_profile.end());
  //   std::cout << " min " << *result << '\n';
  //   result = std::min_element(matrix_profile_stomp.begin(), matrix_profile_stomp.end());
  //   std::cout << " min " << *result << '\n';
  
  //   result = std::max_element(matrix_profile_stompv2.begin(), matrix_profile_stompv2.end());
  //   std::cout << " max " << *result << '\n';
  //   result = std::min_element(matrix_profile_stompv2.begin(), matrix_profile_stompv2.end());
  //   std::cout << " min " << *result << '\n';
  }


int main() {
  // testDistance();
  //  compareStumpy();

  testMatrixProfileComputationSpeed(132000, 64);
  // test_stompv2(1000, 1);

  return 0;
}
