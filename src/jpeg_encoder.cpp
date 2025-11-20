#include "jpeg_encoder.h"
#include "dct.h"
#include "quantization.h"
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cmath>

JPEGEncoder::JPEGEncoder(int quality, int num_threads)
    : quality_(quality), num_threads_(num_threads) {
    if (num_threads_ <= 0) {
        num_threads_ = omp_get_max_threads();
    }
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
    
    #pragma omp parallel for schedule(static)
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
    
    #pragma omp parallel for schedule(static)
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

// Оптимизированная функция обработки канала
void JPEGEncoder::process_channel_parallel(const std::vector<double>& channel,
                                          std::vector<int>& quantized,
                                          int width, int height,
                                          const int quant_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE]) {
    int blocks_x = (width + DCT::BLOCK_SIZE - 1) / DCT::BLOCK_SIZE;
    int blocks_y = (height + DCT::BLOCK_SIZE - 1) / DCT::BLOCK_SIZE;
    int total_blocks = blocks_x * blocks_y;
    
    quantized.resize(total_blocks * DCT::BLOCK_SIZE * DCT::BLOCK_SIZE);
    
    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        double block_data[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
        double dct_output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
        
        int block_y = block_idx / blocks_x;
        int block_x = block_idx % blocks_x;
        int base_y = block_y * DCT::BLOCK_SIZE;
        int base_x = block_x * DCT::BLOCK_SIZE;
        
        // Заполнение блока и сдвиг уровня (-128)
        for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
            for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
                int y = base_y + i;
                int x = base_x + j;
                double val = (y < height && x < width) ? channel[y * width + x] : 128.0;
                block_data[i][j] = val - 128.0;
            }
        }
        
        DCT::forward_dct_optimized(block_data, dct_output);
        
        // Квантование (используя предвычисленную таблицу)
        int* out_ptr = &quantized[block_idx * 64];
        for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
            for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
                out_ptr[i * 8 + j] = static_cast<int>(std::round(dct_output[i][j] / quant_table[i][j]));
            }
        }
    }
}

void JPEGEncoder::decode_channel_parallel(const std::vector<int>& quantized,
                                         std::vector<double>& channel,
                                         int width, int height,
                                         const int quant_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE]) {
    int blocks_x = (width + DCT::BLOCK_SIZE - 1) / DCT::BLOCK_SIZE;
    int blocks_y = (height + DCT::BLOCK_SIZE - 1) / DCT::BLOCK_SIZE;
    int total_blocks = blocks_x * blocks_y;
    
    channel.resize(width * height);
    
    #pragma omp parallel for schedule(static)
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        int quantized_block[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
        double dequantized[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
        double idct_output[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
        
        int block_y = block_idx / blocks_x;
        int block_x = block_idx % blocks_x;
        
        const int* input_ptr = &quantized[block_idx * 64];
        
        // Копирование во временный буфер
        for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
            for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
                quantized_block[i][j] = input_ptr[i * 8 + j];
            }
        }
        
        // Деквантование
        for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
            for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
                dequantized[i][j] = quantized_block[i][j] * quant_table[i][j];
            }
        }
        
        DCT::inverse_dct(dequantized, idct_output);
        
        // Запись в канал
        for (int i = 0; i < DCT::BLOCK_SIZE; ++i) {
            for (int j = 0; j < DCT::BLOCK_SIZE; ++j) {
                int y = block_y * DCT::BLOCK_SIZE + i;
                int x = block_x * DCT::BLOCK_SIZE + j;
                if (y < height && x < width) {
                    channel[y * width + x] = idct_output[i][j] + 128.0;
                }
            }
        }
    }
}

std::vector<uint8_t> JPEGEncoder::rle_encode(const std::vector<int>& data) {
    std::vector<uint8_t> encoded;
    encoded.reserve(data.size() / 2);
    
    size_t i = 0;
    while (i < data.size()) {
        int value = data[i];
        size_t count = 1;
        
        while (i + count < data.size() && data[i + count] == value && count < 255) {
            count++;
        }
        
        encoded.push_back(static_cast<uint8_t>(count));
        
        uint32_t uval = static_cast<uint32_t>(value);
        encoded.push_back((uval >> 0) & 0xFF);
        encoded.push_back((uval >> 8) & 0xFF);
        encoded.push_back((uval >> 16) & 0xFF);
        encoded.push_back((uval >> 24) & 0xFF);
        
        i += count;
    }
    return encoded;
}

std::vector<int> JPEGEncoder::rle_decode(const uint8_t* data, size_t& offset, size_t target_size) {
    std::vector<int> decoded;
    decoded.reserve(target_size);
    
    while (decoded.size() < target_size) {
        uint8_t count = data[offset++];
        
        uint32_t uval = 0;
        uval |= static_cast<uint32_t>(data[offset++]) << 0;
        uval |= static_cast<uint32_t>(data[offset++]) << 8;
        uval |= static_cast<uint32_t>(data[offset++]) << 16;
        uval |= static_cast<uint32_t>(data[offset++]) << 24;
        
        int value = static_cast<int>(uval);
        
        for (int j = 0; j < count && decoded.size() < target_size; ++j) {
            decoded.push_back(value);
        }
    }
    return decoded;
}

EncodedData JPEGEncoder::encode(const Image& image) {
    std::cout << "Encoding " << image.width << "x" << image.height 
              << " Q=" << quality_ << " Threads=" << num_threads_ << std::endl;
    
    auto start_time = omp_get_wtime();
    
    // 1. Подготовка таблиц квантования (один раз)
    int luma_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    int chroma_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    Quantization::scale_quantization_table(Quantization::LUMINANCE_QUANT_TABLE, luma_table, quality_);
    Quantization::scale_quantization_table(Quantization::CHROMINANCE_QUANT_TABLE, chroma_table, quality_);
    
    // 2. Конвертация цвета
    std::vector<double> y, cb, cr;
    rgb_to_ycbcr(image, y, cb, cr);
    
    std::vector<int> y_quantized, cb_quantized, cr_quantized;
    
    // 3. Обработка каналов (параллельно внутри функций)
    process_channel_parallel(y, y_quantized, image.width, image.height, luma_table);
    process_channel_parallel(cb, cb_quantized, image.width, image.height, chroma_table);
    process_channel_parallel(cr, cr_quantized, image.width, image.height, chroma_table);
    
    int total_coeffs = y_quantized.size() + cb_quantized.size() + cr_quantized.size();
    int zero_coeffs = 0;
    for (int val : y_quantized) if (val == 0) zero_coeffs++;
    for (int val : cb_quantized) if (val == 0) zero_coeffs++;
    for (int val : cr_quantized) if (val == 0) zero_coeffs++;
    
    // 4. RLE сжатие (параллельно по каналам)
    std::vector<uint8_t> y_rle, cb_rle, cr_rle;
    #pragma omp parallel sections
    {
        #pragma omp section
        y_rle = rle_encode(y_quantized);
        #pragma omp section
        cb_rle = rle_encode(cb_quantized);
        #pragma omp section
        cr_rle = rle_encode(cr_quantized);
    }
    
    auto end_time = omp_get_wtime();
    
    float compression_potential = 100.0f * zero_coeffs / total_coeffs;
    size_t uncompressed = total_coeffs * sizeof(int);
    size_t compressed = y_rle.size() + cb_rle.size() + cr_rle.size();
    float actual_compression = 100.0f * (1.0f - static_cast<float>(compressed) / uncompressed);
    
    std::cout << "Time: " << (end_time - start_time) * 1000.0 << " ms" << std::endl;
    std::cout << "Zeros: " << zero_coeffs << "/" << total_coeffs 
              << " (" << compression_potential << "%)" << std::endl;
    std::cout << "Size: " << uncompressed / 1024 << " KB -> " 
              << compressed / 1024 << " KB (" << actual_compression << "%)" << std::endl;
    
    EncodedData result;
    result.data.resize(sizeof(int) * 9);
    
    int idx = 0;
    std::memcpy(&result.data[idx], &image.width, sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.data[idx], &image.height, sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.data[idx], &quality_, sizeof(int)); idx += sizeof(int);
    
    int y_size = y_rle.size();
    int cb_size = cb_rle.size();
    int cr_size = cr_rle.size();
    std::memcpy(&result.data[idx], &y_size, sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.data[idx], &cb_size, sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.data[idx], &cr_size, sizeof(int)); idx += sizeof(int);
    
    int y_count = y_quantized.size();
    int cb_count = cb_quantized.size();
    int cr_count = cr_quantized.size();
    std::memcpy(&result.data[idx], &y_count, sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.data[idx], &cb_count, sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.data[idx], &cr_count, sizeof(int)); idx += sizeof(int);
    
    size_t offset = idx;
    result.data.resize(offset + compressed);
    
    std::memcpy(&result.data[offset], y_rle.data(), y_rle.size());
    offset += y_rle.size();
    std::memcpy(&result.data[offset], cb_rle.data(), cb_rle.size());
    offset += cb_rle.size();
    std::memcpy(&result.data[offset], cr_rle.data(), cr_rle.size());
    
    result.size = result.data.size();
    result.zero_coefficients = zero_coeffs;
    result.total_coefficients = total_coeffs;
    result.uncompressed_size = uncompressed;
    
    return result;
}

Image JPEGEncoder::decode(const EncodedData& encoded) {
    Image result;
    int idx = 0;
    
    std::memcpy(&result.width, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    std::memcpy(&result.height, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    int saved_quality;
    std::memcpy(&saved_quality, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    
    int y_size, cb_size, cr_size;
    std::memcpy(&y_size, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    std::memcpy(&cb_size, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    std::memcpy(&cr_size, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    
    int y_count, cb_count, cr_count;
    std::memcpy(&y_count, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    std::memcpy(&cb_count, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    std::memcpy(&cr_count, &encoded.data[idx], sizeof(int)); idx += sizeof(int);
    
    size_t offset = idx;
    
    // Подготовка таблиц декодирования
    int luma_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    int chroma_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE];
    Quantization::scale_quantization_table(Quantization::LUMINANCE_QUANT_TABLE, luma_table, saved_quality);
    Quantization::scale_quantization_table(Quantization::CHROMINANCE_QUANT_TABLE, chroma_table, saved_quality);
    
    std::vector<int> y_quantized, cb_quantized, cr_quantized;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            size_t local_offset = offset;
            y_quantized = rle_decode(encoded.data.data(), local_offset, y_count);
        }
        #pragma omp section
        {
            size_t local_offset = offset + y_size;
            cb_quantized = rle_decode(encoded.data.data(), local_offset, cb_count);
        }
        #pragma omp section
        {
            size_t local_offset = offset + y_size + cb_size;
            cr_quantized = rle_decode(encoded.data.data(), local_offset, cr_count);
        }
    }
    
    std::vector<double> y, cb, cr;
    decode_channel_parallel(y_quantized, y, result.width, result.height, luma_table);
    decode_channel_parallel(cb_quantized, cb, result.width, result.height, chroma_table);
    decode_channel_parallel(cr_quantized, cr, result.width, result.height, chroma_table);
    
    result.channels = 3;
    result.data.resize(result.width * result.height * 3);
    ycbcr_to_rgb(y, cb, cr, result);
    
    return result;
}
