#pragma once

#include <vector>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include <filesystem>

// Reads input time series from file
template <class T>
void readFile(const char *filename, std::vector<T> &v, const char *format)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL)
    {
        printf("unable to open file\n");
        exit(EXIT_FAILURE);
    }

    T elem;
    while (!feof(file))
    {
        fscanf(file, format, &elem);
        v.push_back(elem);
    }
    v.pop_back();
    fclose(file);
}

template <class T>
auto vector_norm(const std::vector<T> &vec) -> T
{
    auto norm = T(0);
    for (const auto &elem : vec)
    {
        norm += elem * elem;
    }
    return std::sqrt(norm);
}

template <class T>
auto compute_vector_difference_norm(const std::vector<T> &vec1, const std::vector<T> &vec2) -> T
{
    assert(vec1.size() == vec2.size());
    std::vector<T> diff(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), diff.begin(), std::minus<T>());
    return vector_norm(diff);
}
template <class T>
auto build_season_vector(const char *filename, const int n)
{
    FILE *file = fopen(filename, "r");

    if (file == NULL)
    {
        printf("unable to open file\n");
        exit(EXIT_FAILURE);
    }

    std::vector<std::pair<int, int>> season;
    std::pair<int, int> pair_index;
    int iter = 1;
    int elem;
    while (!feof(file))
    {
        fscanf(file, "%d", &elem);

        if (iter % 2 == 0)
        {
            pair_index.second = std::min(elem, n);
            season.push_back(pair_index);
        }
        else
        {
            if (elem >= n)
            {
                break;
            }
            pair_index.first = elem;
        }
        ++iter;
    }
    fclose(file);
    return season;
}

template <class T>
void write_vector_to_file(const std::string& filename, const std::vector<T>& vec) {
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