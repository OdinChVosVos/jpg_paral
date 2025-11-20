#include "dct.h"
#include <cmath>
#include <omp.h>

const std::array<std::array<double, DCT::BLOCK_SIZE>, DCT::BLOCK_SIZE> 
    DCT::cosine_table = DCT::init_cosine_table();

std::array<std::array<double, DCT::BLOCK_SIZE>, DCT::BLOCK_SIZE> DCT::init_cosine_table() {
    std::array<std::array<double, DCT::BLOCK_SIZE>, DCT::BLOCK_SIZE> table;
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        for (int x = 0; x < BLOCK_SIZE; ++x) {
            table[u][x] = std::cos((2.0 * x + 1.0) * u * PI / (2.0 * BLOCK_SIZE));
        }
    }
    return table;
}

void DCT::forward_dct(const double input[BLOCK_SIZE][BLOCK_SIZE],
                     double output[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        for (int v = 0; v < BLOCK_SIZE; ++v) {
            double sum = 0.0;
            
            for (int x = 0; x < BLOCK_SIZE; ++x) {
                for (int y = 0; y < BLOCK_SIZE; ++y) {
                    double cos_u = std::cos((2.0 * x + 1.0) * u * PI / 16.0);
                    double cos_v = std::cos((2.0 * y + 1.0) * v * PI / 16.0);
                    sum += input[x][y] * cos_u * cos_v;
                }
            }
            
            output[u][v] = 0.25 * DCT::C(u) * DCT::C(v) * sum;
        }
    }
}

void DCT::forward_dct_optimized(const double input[BLOCK_SIZE][BLOCK_SIZE],
                               double output[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        for (int v = 0; v < BLOCK_SIZE; ++v) {
            double sum = 0.0;
            
            for (int x = 0; x < BLOCK_SIZE; ++x) {
                for (int y = 0; y < BLOCK_SIZE; ++y) {
                    sum += input[x][y] * cosine_table[u][x] * cosine_table[v][y];
                }
            }
            
            output[u][v] = 0.25 * DCT::C(u) * DCT::C(v) * sum;
        }
    }
}

void DCT::inverse_dct(const double input[BLOCK_SIZE][BLOCK_SIZE],
                     double output[BLOCK_SIZE][BLOCK_SIZE]) {
    for (int x = 0; x < BLOCK_SIZE; ++x) {
        for (int y = 0; y < BLOCK_SIZE; ++y) {
            double sum = 0.0;
            
            for (int u = 0; u < BLOCK_SIZE; ++u) {
                for (int v = 0; v < BLOCK_SIZE; ++v) {
                    double cos_u = cosine_table[u][x];
                    double cos_v = cosine_table[v][y];
                    sum += C(u) * C(v) * input[u][v] * cos_u * cos_v;
                }
            }
            
            output[x][y] = 0.25 * sum;
        }
    }
}
