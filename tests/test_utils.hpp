#pragma once
#include <vector>
#include <random>

template <class T>
std::vector<T> generateRandomTimeSeries(int n, int seed = 123)
{
    std::vector<T> time_series;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < n; i++)
    {
        time_series.push_back(dis(gen));
    }
    return time_series;
}