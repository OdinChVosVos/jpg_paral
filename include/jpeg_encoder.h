#ifndef JPEG_ENCODER_H
#define JPEG_ENCODER_H

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
    
    void process_channel_parallel(const std::vector<double>& channel,
                                 std::vector<int>& quantized,
                                 int width, int height,
                                 bool is_luminance);
    
    void process_block(const double* block_data,
                      int* output,
                      bool is_luminance);
};

#endif
