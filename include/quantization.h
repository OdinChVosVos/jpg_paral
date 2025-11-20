#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include "dct.h"

class Quantization {
public:
    static const int LUMINANCE_QUANT_TABLE[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    static const int CHROMINANCE_QUANT_TABLE[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];

    static void scale_quantization_table(
        const int input[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
        int output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE],
        int quality);
};

#endif
