#include <iostream>
#include <vector>
#include <chrono> // For timing
#include <random> // For random number generation
#include <algorithm>
#include <cmath>
#include <span>
#include <fstream>
#include <matrix_profile.hpp>
#include <seasonal_matrix_profile.hpp>
#include <interval_matrix_profile.hpp>
#include <utils.hpp>

using value_type = double;

void testDistance()
{
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

void compareStumpy()
{
  printf("compareStumpy\n");
  std::vector<double> ts;
  readFile<double>("../Data/ts_sst.txt", ts, "%lf");
  std::vector<double> mp_ref;
  readFile<double>("../Data/mp_sst.txt", mp_ref, "%lf");
  int window_size = 7;

std:
  double norm_ref = vector_norm(mp_ref);

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
  if (std::equal(mp_ref.begin(), mp_ref.end(), matrix_profile_stomp.begin()))
  {
    std::cout << " success" << std::endl;
  }
  else
  {
    std::cout << "failure" << std::endl;
  }
  difference_norm = compute_vector_difference_norm(mp_ref, matrix_profile_brute_force);
  std::cout << "Norm of the difference with BruteForce: " << difference_norm << std::endl;
  if (std::equal(mp_ref.begin(), mp_ref.end(), matrix_profile_stomp.begin()))
  {
    std::cout << " success" << std::endl;
  }
  else
  {
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
  if (outputFile.is_open())
  {
    for (const auto &value : matrix_profile_brute_force)
    {
      outputFile << value << "\n";
    }
    outputFile.close();
    std::cout << "Matrix profile has been written to matrix_profile.txt\n";
  }
  else
  {
    std::cerr << "Unable to open file for writing.\n";
  }
  // Write the result to a file
  std::ofstream outputFile1("../Data/matrix_profile_stomp.txt");
  if (outputFile1.is_open())
  {
    for (const auto &value : matrix_profile_stomp)
    {
      outputFile1 << value << "\n";
    }
    outputFile1.close();
    std::cout << "Matrix profile has been written to matrix_profile.txt\n";
  }
  else
  {
    std::cerr << "Unable to open file for writing.\n";
  }
  std::ofstream outputFile2("../Data/index_profile_stomp.txt");
  if (outputFile2.is_open())
  {
    for (const auto &value : index_profile_stomp)
    {
      outputFile2 << value << "\n";
    }
    outputFile2.close();
    std::cout << "Matrix profile has been written to matrix_profile.txt\n";
  }
  else
  {
    std::cerr << "Unable to open file for writing.\n";
  }
  std::ofstream outputFile3("../Data/index_profile_bf.txt");
  if (outputFile3.is_open())
  {
    for (const auto &value : index_profile_stomp)
    {
      outputFile3 << value << "\n";
    }
    outputFile3.close();
    std::cout << "Matrix profile has been written to matrix_profile.txt\n";
  }
  else
  {
    std::cerr << "Unable to open file for writing.\n";
  }
}

// Function to test computation speed of matrix profile
void testMatrixProfileComputationSpeed(int vector_size, int window_size)
{
  // Generate random vector

  std::vector<value_type> data;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

  std::cout << "Random gen" << std::endl;

  for (int i = 0; i < vector_size; ++i)
  {
    data.push_back(dis(gen));
  }

  std::cout << "Brute Force" << std::endl;

  // Measure computation time
  auto start_time = std::chrono::high_resolution_clock::now();
  auto mpOutput = computeMatrixProfileBruteForce(data, window_size);
  auto matrix_profile = std::get<0>(mpOutput);
  auto end_time = std::chrono::high_resolution_clock::now();

  // // Compute duration
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

  // // Output results
  std::cout << "Matrix profile computed for vector of size " << vector_size << " with window size " << window_size << std::endl;
  std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

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
  start_time = std::chrono::high_resolution_clock::now();
  mpOutput = blockSTOMP_v2(data, window_size, 5000, 5000);
  auto matrix_profile_blockstomp = std::get<0>(mpOutput);
  end_time = std::chrono::high_resolution_clock::now();
  // Compute duration
  duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

  // std::cout << "BlockSTOMP v3" << std::endl;
  // // Measure computation time
  // start_time = std::chrono::high_resolution_clock::now();
  // mpOutput = blockSTOMP_v3(data, window_size, 5000, 5000);
  // matrix_profile_blockstomp = std::get<0>(mpOutput);
  // end_time = std::chrono::high_resolution_clock::now();
  // // Compute duration
  // duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

  // std::cout << "BlockSTOMP v4" << std::endl;
  // // Measure computation time
  // start_time = std::chrono::high_resolution_clock::now();
  // mpOutput = blockSTOMP_v4(data, window_size, 5000, 5000);
  // matrix_profile_blockstomp = std::get<0>(mpOutput);
  // end_time = std::chrono::high_resolution_clock::now();
  // // Compute duration
  // duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

  // std::cout << "BlockSTOMP v5" << std::endl;
  // // Measure computation time
  // start_time = std::chrono::high_resolution_clock::now();
  // mpOutput = blockSTOMP_v5(data, window_size, 5000, 5000);
  // matrix_profile_blockstomp = std::get<0>(mpOutput);
  // end_time = std::chrono::high_resolution_clock::now();
  // // Compute duration
  // duration =  std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  // std::cout << "Computation time: " << duration.count() << " seconds" << std::endl;

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

auto smp_computation_speed(int vector_size, int period_size, int n_seasons, int window_size)
{
  // Generate random vector

  std::vector<value_type> data;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0); // Random numbers between 0 and 1

  std::cout << "Random gen" << std::endl;

  for (int i = 0; i < vector_size; ++i)
  {
    data.push_back(dis(gen));
  }
  int exclude = 2;
  const int n_sequence = data.size() - window_size + 1;
  int season_size = period_size / n_seasons;
  int n_period = vector_size / period_size;
  std::vector<std::vector<std::pair<int, int>>> seasons;
  for (int i = 0; i < n_seasons; ++i)
  {
    std::vector<std::pair<int, int>> season;
    for (int j = 0; j < vector_size; j += period_size)
    {
      const int start = j + i * season_size;
      const int end = std::min(start + season_size, n_sequence);
      season.push_back(std::make_pair(start, end));
    }
    seasons.push_back(season);
  }

  // Calculate the matrix profile using the three methods
  auto start_time = std::chrono::high_resolution_clock::now();
  auto result_brute_force = seasonal_matrix_profile_brute_force(data, window_size, exclude, seasons);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  std::cout << "Brute force execution time: " << duration.count() << " seconds" << std::endl;

  start_time = std::chrono::high_resolution_clock::now();
  auto result_brute_force_blocking = seasonal_matrix_profile_brute_force_blocking(data, window_size, exclude, seasons);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  std::cout << "Brute force blocking execution time: " << duration.count() << " seconds" << std::endl;

  start_time = std::chrono::high_resolution_clock::now();
  auto result_stomp = seasonal_matrix_profile_STOMP_blocking(data, window_size, exclude, seasons);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  std::cout << "STOMP execution time: " << duration.count() << " seconds" << std::endl;
}

void write_vector_to_file(const std::string& filename, const std::vector<double>& vec) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    for (const auto& value : vec) {
        file << value << "\n";
    }
    file.close();
}

auto test_climate_series()
{
  std::vector<double> data;
  readFile<double>("../Data/SST/medi.txt", data, "%lf");
  // Define the window size, exclude value, and seasons
  const int window_size = 7;
  const int exclude = 7;
  const int k = 3;
  const int n = data.size() - window_size + 1;
  std::vector<std::vector<std::pair<int, int>>> seasons;
  seasons.push_back(build_season_vector<double>("../Data/SST/seasons_sst_0.txt", n));
  seasons.push_back(build_season_vector<double>("../Data/SST/seasons_sst_1.txt", n));
  seasons.push_back(build_season_vector<double>("../Data/SST/seasons_sst_2.txt", n));
  seasons.push_back(build_season_vector<double>("../Data/SST/seasons_sst_3.txt", n));
  int interval_length = 90;
  std::vector<int> period_starts;
  readFile<int>("../Data/SST/periods_start_sst.txt", period_starts, "%d");

  // // Calculate the matrix profile using the three methods
  // auto start_time = std::chrono::high_resolution_clock::now();
  // auto result = seasonal_matrix_profile_brute_force(data, window_size, exclude, seasons);
  // std::vector<double> smp_bf = std::get<0>(result);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "SMP BF execution time: " << duration.count() << " ms" << std::endl;

  // start_time = std::chrono::high_resolution_clock::now();
  // result = seasonal_matrix_profile_STOMP_blocking(data, window_size, exclude, seasons);
  // std::vector<double> smp_stomp = std::get<0>(result);
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "SMP STOMP execution time: " << duration.count() << " ms" << std::endl;

  // start_time = std::chrono::high_resolution_clock::now();
  // result = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
  // std::vector<double> imp_bf = std::get<0>(result);
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "IMP BF execution time: " << duration.count() << " ms" << std::endl;
  
  // start_time = std::chrono::high_resolution_clock::now();
  // result = interval_matrix_profile_STOMP_bf(data, window_size, period_starts, interval_length, exclude);
  // std::vector<double> imp_stomp = std::get<0>(result);
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "IMP STOMP execution time: " << duration.count() << " ms" << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();
  auto result = interval_matrix_profile_STOMP_ep(data, window_size, period_starts, interval_length, exclude);
  std::vector<double> imp_stomp_ep = std::get<0>(result);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "IMP STOMP 2 execution time: " << duration.count() << " ms" << std::endl;

  start_time = std::chrono::high_resolution_clock::now();
  result = interval_matrix_profile_STOMP_kNN(data, window_size, period_starts, interval_length, exclude, k);
  std::vector<double> imp_stomp_kNN = std::get<0>(result);
  end_time = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  std::cout << "IMP STOMP kNN execution time: " << duration.count() << " ms" << std::endl;

  // start_time = std::chrono::high_resolution_clock::now();
  // result = computeMatrixProfileBruteForce(data, window_size);
  // std::vector<double> mp_bf = std::get<0>(result);
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "MP BF execution time: " << duration.count() << " ms" << std::endl;

  // start_time = std::chrono::high_resolution_clock::now();
  // result = blockSTOMP_v2(data, window_size, 5000, 5000);
  // std::vector<double> mp_stomp = std::get<0>(result);
  // end_time = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "MP STOMP execution time: " << duration.count() << " ms" << std::endl;

  // write_vector_to_file("../Data/SST/mp_bf.txt", mp_bf);
  // write_vector_to_file("../Data/SST/mp_stomp.txt", mp_stomp);
  // write_vector_to_file("../Data/SST/smp_bf.txt", smp_bf);
  // write_vector_to_file("../Data/SST/smp_stomp.txt", smp_stomp);
  // write_vector_to_file("../Data/SST/imp_bf.txt", imp_bf);
  // write_vector_to_file("../Data/SST/imp_stomp.txt", imp_stomp);
  write_vector_to_file("../Data/SST/imp_stomp_ep.txt", imp_stomp_ep);
  write_vector_to_file("../Data/SST/imp_stomp_kNN.txt", imp_stomp_kNN);
}

int main()
{
  // testDistance();
  //  compareStumpy();
  // testMatrixProfileComputationSpeed(20000, 7);
  // smp_computation_speed(2000000, 1000, 4, 64);
  // test_stompv2(1000, 1);
  test_climate_series();

  return 0;
}
