#include <vector>
#include <stdio.h>
#include <iostream>
#include <cassert>
#include <filesystem>

//Reads input time series from file
template<class T>
void readFile(const char* filename, std::vector<T>& v, const char *format) 
{
    FILE* file = fopen(filename, "r");

    if (file == NULL) {
        printf("unable to open file\n");
        exit(EXIT_FAILURE);
    }
    
    T elem;
    while(! feof(file)){
            fscanf(file, format, &elem);
            v.push_back(elem);
        }
    v.pop_back();
    fclose(file);
}



template<class T>
auto vector_norm(const std::vector<T>& vec) -> T{
    auto norm = T(0);
    for (const auto& elem : vec) {
        norm += elem * elem;
    }
    return std::sqrt(norm);
}

template<class T>
auto compute_vector_difference_norm(const std::vector<T>& vec1, const std::vector<T>& vec2) -> T{
    assert(vec1.size() == vec2.size());
    std::vector<T> diff(vec1.size());
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), diff.begin(), std::minus<T>());
    return vector_norm(diff);
}
