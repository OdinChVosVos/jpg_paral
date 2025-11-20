#ifndef DCT_H
#define DCT_H

#include <vector>
#include <cmath>
#include <array>

class DCT {
public:
    static constexpr int BLOCK_SIZE = 8;
    static constexpr double PI = 3.14159265358979323846;
    
    static void forward_dct(const double input[BLOCK_SIZE][BLOCK_SIZE],
                           double output[BLOCK_SIZE][BLOCK_SIZE]);
    
    static void inverse_dct(const double input[BLOCK_SIZE][BLOCK_SIZE],
                           double output[BLOCK_SIZE][BLOCK_SIZE]);
    
    static void forward_dct_optimized(const double input[BLOCK_SIZE][BLOCK_SIZE],
                                     double output[BLOCK_SIZE][BLOCK_SIZE]);
    
private:
    static std::array<std::array<double, BLOCK_SIZE>, BLOCK_SIZE> init_cosine_table();
    static const std::array<std::array<double, BLOCK_SIZE>, BLOCK_SIZE> cosine_table;
    
    static inline double C(int u) {
        return (u == 0) ? (1.0 / std::sqrt(2.0)) : 1.0;
    }
};

#endif
