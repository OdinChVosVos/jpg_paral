#ifndef JPEG_ENCODER_H
#define JPEG_ENCODER_H

#include "dct.h" // Для BLOCK_SIZE
#include <vector>
#include <cstdint>
#include <string>

struct Image {
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;
};

struct EncodedData {
    std::vector<uint8_t> data;
    size_t size;
    int zero_coefficients;
    int total_coefficients;
    size_t uncompressed_size;
};

class JPEGEncoder {
public:
    JPEGEncoder(int quality = 85, int num_threads = 4);
    
    EncodedData encode(const Image& image);
    Image decode(const EncodedData& encoded);
    
    void set_quality(int quality);
    void set_threads(int num_threads);
    
private:
    int quality_;
    int num_threads_;
    
    void rgb_to_ycbcr(const Image& image,
                     std::vector<double>& y,
                     std::vector<double>& cb,
                     std::vector<double>& cr);
    
    void ycbcr_to_rgb(const std::vector<double>& y,
                     const std::vector<double>& cb,
                     const std::vector<double>& cr,
                     Image& image);
    
    // Упрощенные функции принимают готовую таблицу квантования
    void process_channel_parallel(const std::vector<double>& channel,
                                 std::vector<int>& quantized,
                                 int width, int height,
                                 const int quant_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE]);
    
    void decode_channel_parallel(const std::vector<int>& quantized,
                                std::vector<double>& channel,
                                int width, int height,
                                const int quant_table[DCT::BLOCK_SIZE][DCT::BLOCK_SIZE]);
    
    std::vector<uint8_t> rle_encode(const std::vector<int>& data);
    std::vector<int> rle_decode(const uint8_t* data, size_t& offset, size_t target_size);
};

#endif
