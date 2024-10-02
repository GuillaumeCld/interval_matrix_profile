#include <iostream>
#include <vector>
#include <chrono> // For timing
#include <random> // For random number generation
#include <algorithm>
#include <cmath>
#include <span>
#include <fstream>
#include <interval_matrix_profile.hpp>
#include <utils.hpp>

void m_impact(const int window_size)
{

    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    const int n_year = 100;
    const int period_length = 365;
    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < period_length; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * period_length);
    }

    const int exclude = window_size;
    const int interval_length = 90; 


    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp = std::get<0>(result);
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = modified_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result);
    std::vector<int> imp_stomp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "STOMP " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_block = std::get<0>(result);
    std::vector<int> imp_block_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;
}

void year_impact(int n_year)
{

    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < 365; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * 365);
    }

    const int interval_length = 90;
    const int window_size = 7;
    const int exclude = window_size;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp = std::get<0>(result);
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = modified_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result);
    std::vector<int> imp_stomp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "STOMP " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_block = std::get<0>(result);
    std::vector<int> imp_block_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;
}

void interval_impact(const int interval_length)
{

    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    const int n_year = 100;
    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < 365; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * 365);
    }

    const int window_size = 7;
    const int exclude = window_size;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_brute_force(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp = std::get<0>(result);
    std::vector<int> imp_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BF " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = modified_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_stomp = std::get<0>(result);
    std::vector<int> imp_stomp_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "STOMP " << duration.count() << " ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_block = std::get<0>(result);
    std::vector<int> imp_block_index = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "BLOCK " << duration.count() << " ms" << std::endl;
}


void parallel_time(const int nthreads)
{
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    const int n_year = 100;
    const int period_length = 365*24;
    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < period_length; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * period_length);
    }

    const int window_size = 7*24;
    const int exclude = window_size;

    const int interval_length = 90*24; 



    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = interval_matrix_profile_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_block = std::get<0>(result);
    std::vector<int> imp_block_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_block = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "BLOCK " << duration_block.count() << " s" << std::endl;
}


void k_impact(const int k)
{
    std::vector<double> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    const int n_year = 50;
    const int period_length = 365*24;
    std::vector<int> period_starts;
    for (int year = 0; year < n_year; ++year)
    {
        for (int i = 0; i < period_length; ++i)
        {
            data.push_back(dis(gen));
        }
        period_starts.push_back(year * period_length);
    }

    const int window_size = 24*7;
    const int exclude = window_size;
    const int interval_length = 90*24; 


    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = BIMP_kNN(data, window_size, period_starts, interval_length, exclude, k, false);
    std::vector<double> imp_block = std::get<0>(result);
    std::vector<int> imp_block_index = std::get<1>(result);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_block = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "kNN BLOCK " << duration_block.count() << " ms" << std::endl;   


    start_time = std::chrono::high_resolution_clock::now();
    result = interval_matrix_profile_STOMP(data, window_size, period_starts, interval_length, exclude);
    std::vector<double> imp_block_stomp = std::get<0>(result);
    std::vector<int> imp_block_index_stomp = std::get<1>(result);
    end_time = std::chrono::high_resolution_clock::now();
    duration_block = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Block " << duration_block.count() << " ms" << std::endl;
}

int main(int argc, char const *argv[])
{

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <test> <param>" << std::endl;
        return 1;
    }
    const int test = std::atoi(argv[1]);
    if (test == 1)
    {
        const int window_size = std::atoi(argv[2]);
        m_impact(window_size);
    }
    else if (test == 2)
    {
        const int n_year = std::atoi(argv[2]);
        year_impact(n_year);
    }
    else if (test == 3)
    {
        const int interval_length = std::atoi(argv[2]);
        interval_impact(interval_length);
    }
    else if (test == 4)
    {
        const int nthreads = std::atoi(argv[2]);
        parallel_time(nthreads);
    }
    else if (test == 5)
    {
        const int k = std::atoi(argv[2]);
        k_impact(k);
    }
    else
    {
        std::cerr << "Invalid test" << std::endl;
        return 1;
    }
    return 0;
}