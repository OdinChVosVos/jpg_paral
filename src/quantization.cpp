#include "quantization.h"
#include <algorithm>

const int Quantization::LUMINANCE_QUANT_TABLE[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE] = {
    {16, 11, 10, 16,  24,  40,  51,  61},
    {12, 12, 14, 19,  26,  58,  60,  55},
    {14, 13, 16, 24,  40,  57,  69,  56},
    {14, 17, 22, 29,  51,  87,  80,  62},
    {18, 22, 37, 56,  68, 109, 103,  77},
    {24, 35, 55, 64,  81, 104, 113,  92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103,  99}
};

const int Quantization::CHROMINANCE_QUANT_TABLE[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE] = {
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};

void Quantization::scale_quantization_table(
    const int input[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
    int output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
    int quality) {
    
    quality = std::clamp(quality, 1, 100);
    
    int scale_factor;
    if (quality < 50) {
        scale_factor = 5000 / quality;
    } else {
        scale_factor = 200 - quality * 2;
    }
    
    for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
        for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
            int temp = (input[i][j] * scale_factor + 50) / 100;
            output[i][j] = std::clamp(temp, 1, 255);
        }
    }
}

void Quantization::quantize(const double input[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
                           int output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
                           const int quant_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
                           int quality) {
    int scaled_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    scale_quantization_table(quant_table, scaled_table, quality);
    
    for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
        for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
            output[i][j] = static_cast<int>(std::round(input[i][j] / scaled_table[i][j]));
        }
    }
}

void Quantization::dequantize(const int input[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
                             double output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
                             const int quant_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
                             int quality) {
    int scaled_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    scale_quantization_table(quant_table, scaled_table, quality);
    
    for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
        for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
            output[i][j] = input[i][j] * scaled_table[i][j];
        }
    }
}
