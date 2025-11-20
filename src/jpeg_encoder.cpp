#include "jpeg_encoder.h"
#include "dct.h"
#include "quantization.h"
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <iostream>

JPEGEncoder::JPEGEncoder(int quality, int num_threads)
    : quality_(quality), num_threads_(num_threads) {
    omp_set_num_threads(num_threads_);
}

void JPEGEncoder::set_quality(int quality) {
    quality_ = std::clamp(quality, 1, 100);
}

void JPEGEncoder::set_threads(int num_threads) {
    num_threads_ = std::max(1, num_threads);
    omp_set_num_threads(num_threads_);
}

void JPEGEncoder::rgb_to_ycbcr(const Image& image,
                               std::vector<double>& y,
                               std::vector<double>& cb,
                               std::vector<double>& cr) {
    size_t pixel_count = image.width * image.height;
    y.resize(pixel_count);
    cb.resize(pixel_count);
    cr.resize(pixel_count);
    
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(pixel_count); ++i) {
        int idx = i * 3;
        double r = image.data[idx];
        double g = image.data[idx + 1];
        double b = image.data[idx + 2];
        
        y[i]  =  0.299    * r + 0.587    * g + 0.114    * b;
        cb[i] = -0.168736 * r - 0.331264 * g + 0.5      * b + 128.0;
        cr[i] =  0.5      * r - 0.418688 * g - 0.081312 * b + 128.0;
    }
}

void JPEGEncoder::ycbcr_to_rgb(const std::vector<double>& y,
                               const std::vector<double>& cb,
                               const std::vector<double>& cr,
                               Image& image) {
    size_t pixel_count = y.size();
    
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(pixel_count); ++i) {
        double y_val = y[i];
        double cb_val = cb[i] - 128.0;
        double cr_val = cr[i] - 128.0;
        
        double r = y_val + 1.402    * cr_val;
        double g = y_val - 0.344136 * cb_val - 0.714136 * cr_val;
        double b = y_val + 1.772    * cb_val;
        
        int idx = i * 3;
        image.data[idx]     = static_cast<uint8_t>(std::clamp(r, 0.0, 255.0));
        image.data[idx + 1] = static_cast<uint8_t>(std::clamp(g, 0.0, 255.0));
        image.data[idx + 2] = static_cast<uint8_t>(std::clamp(b, 0.0, 255.0));
    }
}

void JPEGEncoder::process_block(const double* block_data,
                                int* output,
                                bool is_luminance) {
    double input_block[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    double dct_output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    int quantized[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    
    for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
        for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
            input_block[i][j] = block_data[i * DCT::BLOCK_SIZE + j] - 128.0;
        }
    }
    
    DCT::forward_dct_optimized(input_block, dct_output);
    
    const int (*quant_table)[DCT::BLOCK_SIZE] = is_luminance 
        ? Quantization::LUMINANCE_QUANT_TABLE
        : Quantization::CHROMINANCE_QUANT_TABLE;
    
    Quantization::quantize(dct_output, quantized, quant_table, quality_);
    
    for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
        for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
            output[i * DCT::BLOCK_SIZE + j] = quantized[i][j];
        }
    }
}

void JPEGEncoder::process_channel_parallel(const std::vector<double>& channel,
                                          std::vector<int>& quantized,
                                          int width, int height,
                                          bool is_luminance) {
    int blocks_x = (width + DCT::BLOCK_SIZE - 1) / DCT::BLOCK_SIZE;
    int blocks_y = (height + DCT::BLOCK_SIZE - 1) / DCT::BLOCK_SIZE;
    int total_blocks = blocks_x * blocks_y;
    
    quantized.resize(total_blocks * DCT::BLOCK_SIZE * DCT::BLOCK_SIZE);
    
    #pragma omp parallel for schedule(dynamic)
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        int block_y = block_idx / blocks_x;
        int block_x = block_idx % blocks_x;
        
        double block_data[DCT::BLOCK_SIZE * DCT::BLOCK_SIZE];
        std::fill(block_data, block_data + DCT::BLOCK_SIZE * DCT::BLOCK_SIZE, 0.0);
        
        for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
            for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
                int y = block_y * DCT::BLOCK_SIZE + i;
                int x = block_x * DCT::BLOCK_SIZE + j;
                
                if (y < height && x < width) {
                    block_data[i * DCT::BLOCK_SIZE + j] = channel[y * width + x];
                }
            }
        }
        
        int* output_ptr = &quantized[block_idx * DCT::BLOCK_SIZE * DCT::BLOCK_SIZE];
        process_block(block_data, output_ptr, is_luminance);
    }
}

EncodedData JPEGEncoder::encode(const Image& image) {
    std::cout << "Начало кодирования изображения " << image.width 
              << "x" << image.height << " с качеством " << quality_ 
              << " используя " << num_threads_ << " потоков" << std::endl;
    
    auto start_time = omp_get_wtime();
    
    std::vector<double> y, cb, cr;
    rgb_to_ycbcr(image, y, cb, cr);
    
    std::vector<int> y_quantized, cb_quantized, cr_quantized;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        process_channel_parallel(y, y_quantized, image.width, image.height, true);
        
        #pragma omp section
        process_channel_parallel(cb, cb_quantized, image.width, image.height, false);
        
        #pragma omp section
        process_channel_parallel(cr, cr_quantized, image.width, image.height, false);
    }
    
    auto end_time = omp_get_wtime();
    std::cout << "Время кодирования: " << (end_time - start_time) * 1000.0 
              << " мс" << std::endl;
    
    EncodedData result;
    result.data.insert(result.data.end(), 
                      reinterpret_cast<uint8_t*>(y_quantized.data()),
                      reinterpret_cast<uint8_t*>(y_quantized.data()) + y_quantized.size() * sizeof(int));
    result.size = result.data.size();
    
    return result;
}

Image JPEGEncoder::decode(const EncodedData& encoded) {
    Image result;
    result.width = 512;
    result.height = 512;
    result.channels = 3;
    result.data.resize(result.width * result.height * 3);
    
    return result;
}
