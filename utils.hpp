#include <vector>
#include <stdio.h>

//Reads input time series from file
template<class T>
void readFile(const char* filename, std::vector<T>& v, const char *format) 
{
    FILE* file = fopen(filename, "r");

    assertm(file == NULL, "unable to open file %s", filename);
    
    auto elem;
    while(! feof(file)){
            fscanf(f, format, &elem);
            v.push_back(elem);
        }
    v.pop_back();
    fclose(file);
}
    