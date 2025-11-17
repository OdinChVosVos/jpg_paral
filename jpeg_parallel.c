// jpeg_parallel.c
#include "jpeg_parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Стандартные таблицы квантования JPEG
static const uint8_t QUANTIZATION_TABLE_LUMINANCE[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

static const uint8_t QUANTIZATION_TABLE_CHROMINANCE[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

// Константы для DCT
static const double PI = 3.14159265358979323846;
static const double SQRT2 = 1.41421356237309504880;

// Масштабирование таблиц квантования по качеству
void scale_quantization_table(const uint8_t *base_table, uint8_t *scaled_table, int quality) {
    int scale;
    if (quality < 50) {
        scale = 5000 / quality;
    } else {
        scale = 200 - quality * 2;
    }
    
    for (int i = 0; i < 64; i++) {
        int val = (base_table[i] * scale + 50) / 100;
        if (val < 1) val = 1;
        if (val > 255) val = 255;
        scaled_table[i] = (uint8_t)val;
    }
}

// RGB -> YCbCr конвертация (параллельная)
YCbCrImage* rgb_to_ycbcr_parallel(Image *img, int num_threads) {
    YCbCrImage *ycbcr = (YCbCrImage*)malloc(sizeof(YCbCrImage));
    ycbcr->width = img->width;
    ycbcr->height = img->height;
    
    int total_pixels = img->width * img->height;
    ycbcr->Y = (float*)malloc(total_pixels * sizeof(float));
    ycbcr->Cb = (float*)malloc(total_pixels * sizeof(float));
    ycbcr->Cr = (float*)malloc(total_pixels * sizeof(float));
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total_pixels; i++) {
        int idx = i * img->channels;
        float R = img->data[idx];
        float G = img->data[idx + 1];
        float B = img->data[idx + 2];
        
        ycbcr->Y[i] = 0.299f * R + 0.587f * G + 0.114f * B;
        ycbcr->Cb[i] = -0.168736f * R - 0.331264f * G + 0.5f * B + 128.0f;
        ycbcr->Cr[i] = 0.5f * R - 0.418688f * G - 0.081312f * B + 128.0f;
    }
    
    return ycbcr;
}

// Fast DCT алгоритм (упрощенная версия AAN)
void dct_8x8(float block[64], float output[64]) {
    float temp[64];
    
    // Применяем 1D DCT к строкам
    for (int i = 0; i < 8; i++) {
        for (int u = 0; u < 8; u++) {
            float sum = 0.0f;
            float cu = (u == 0) ? (1.0f / SQRT2) : 1.0f;
            
            for (int x = 0; x < 8; x++) {
                sum += block[i * 8 + x] * cos((2 * x + 1) * u * PI / 16.0);
            }
            temp[i * 8 + u] = 0.5f * cu * sum;
        }
    }
    
    // Применяем 1D DCT к столбцам
    for (int j = 0; j < 8; j++) {
        for (int v = 0; v < 8; v++) {
            float sum = 0.0f;
            float cv = (v == 0) ? (1.0f / SQRT2) : 1.0f;
            
            for (int y = 0; y < 8; y++) {
                sum += temp[y * 8 + j] * cos((2 * y + 1) * v * PI / 16.0);
            }
            output[v * 8 + j] = 0.5f * cv * sum;
        }
    }
}

// Квантование блока
void quantize_block(float dct_block[64], int16_t output[64], const uint8_t qtable[64]) {
    for (int i = 0; i < 64; i++) {
        output[i] = (int16_t)round(dct_block[i] / qtable[i]);
    }
}

// Обратное квантование
void dequantize_block(int16_t input[64], float output[64], const uint8_t qtable[64]) {
    for (int i = 0; i < 64; i++) {
        output[i] = input[i] * qtable[i];
    }
}

// Обратное DCT
void idct_8x8(float dct_block[64], float output[64]) {
    float temp[64];
    
    // Применяем 1D IDCT к строкам
    for (int i = 0; i < 8; i++) {
        for (int x = 0; x < 8; x++) {
            float sum = 0.0f;
            
            for (int u = 0; u < 8; u++) {
                float cu = (u == 0) ? (1.0f / SQRT2) : 1.0f;
                sum += cu * dct_block[i * 8 + u] * cos((2 * x + 1) * u * PI / 16.0);
            }
            temp[i * 8 + x] = 0.5f * sum;
        }
    }
    
    // Применяем 1D IDCT к столбцам
    for (int j = 0; j < 8; j++) {
        for (int y = 0; y < 8; y++) {
            float sum = 0.0f;
            
            for (int v = 0; v < 8; v++) {
                float cv = (v == 0) ? (1.0f / SQRT2) : 1.0f;
                sum += cv * temp[v * 8 + j] * cos((2 * y + 1) * v * PI / 16.0);
            }
            output[y * 8 + j] = 0.5f * sum;
        }
    }
}

// Обработка блоков с параллелизацией
void process_blocks_parallel(YCbCrImage *ycbcr, int16_t **quantized_blocks, 
                             int quality, int num_threads) {
    int blocks_wide = (ycbcr->width + 7) / 8;
    int blocks_high = (ycbcr->height + 7) / 8;
    int total_blocks = blocks_wide * blocks_high;
    
    // Подготовка таблиц квантования
    uint8_t qtable_lum[64], qtable_chr[64];
    scale_quantization_table(QUANTIZATION_TABLE_LUMINANCE, qtable_lum, quality);
    scale_quantization_table(QUANTIZATION_TABLE_CHROMINANCE, qtable_chr, quality);
    
    // Выделяем память для квантованных коэффициентов (Y, Cb, Cr)
    quantized_blocks[0] = (int16_t*)malloc(total_blocks * 64 * sizeof(int16_t));
    quantized_blocks[1] = (int16_t*)malloc(total_blocks * 64 * sizeof(int16_t));
    quantized_blocks[2] = (int16_t*)malloc(total_blocks * 64 * sizeof(int16_t));
    
    omp_set_num_threads(num_threads);
    
    // Параллельная обработка блоков
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int by = 0; by < blocks_high; by++) {
        for (int bx = 0; bx < blocks_wide; bx++) {
            int block_idx = by * blocks_wide + bx;
            
            // Обработка Y, Cb, Cr компонент
            float *channels[3] = {ycbcr->Y, ycbcr->Cb, ycbcr->Cr};
            const uint8_t *qtables[3] = {qtable_lum, qtable_chr, qtable_chr};
            
            for (int c = 0; c < 3; c++) {
                float block[64];
                float dct_block[64];
                
                // Извлекаем блок 8x8
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        int img_x = bx * 8 + x;
                        int img_y = by * 8 + y;
                        
                        if (img_x < ycbcr->width && img_y < ycbcr->height) {
                            block[y * 8 + x] = channels[c][img_y * ycbcr->width + img_x] - 128.0f;
                        } else {
                            block[y * 8 + x] = 0.0f;  // Padding
                        }
                    }
                }
                
                // Применяем DCT
                dct_8x8(block, dct_block);
                
                // Квантование
                quantize_block(dct_block, &quantized_blocks[c][block_idx * 64], qtables[c]);
            }
        }
    }
}

// Сжатие изображения с параллелизацией
CompressedImage* jpeg_compress_parallel(Image *img, int quality, int num_threads) {
    printf("Starting parallel JPEG compression with %d threads...\n", num_threads);
    double start_time = omp_get_wtime();
    
    // 1. RGB -> YCbCr конвертация
    double conv_start = omp_get_wtime();
    YCbCrImage *ycbcr = rgb_to_ycbcr_parallel(img, num_threads);
    double conv_time = omp_get_wtime() - conv_start;
    printf("Color conversion: %.4f seconds\n", conv_time);
    
    // 2. Обработка блоков (DCT + квантование)
    double dct_start = omp_get_wtime();
    int16_t *quantized_blocks[3];
    process_blocks_parallel(ycbcr, quantized_blocks, quality, num_threads);
    double dct_time = omp_get_wtime() - dct_start;
    printf("DCT + Quantization: %.4f seconds\n", dct_time);
    
    // 3. Упрощенное "сжатие" (в реальности нужно Huffman/Arithmetic кодирование)
    int blocks_wide = (ycbcr->width + 7) / 8;
    int blocks_high = (ycbcr->height + 7) / 8;
    int total_blocks = blocks_wide * blocks_high;
    
    CompressedImage *compressed = (CompressedImage*)malloc(sizeof(CompressedImage));
    compressed->size = total_blocks * 64 * 3 * sizeof(int16_t);  // Упрощенный размер
    compressed->data = (uint8_t*)malloc(compressed->size);
    compressed->width = img->width;
    compressed->height = img->height;
    compressed->quality = quality;
    
    // Копируем квантованные данные (в реальности здесь должно быть энтропийное кодирование)
    memcpy(compressed->data, quantized_blocks[0], total_blocks * 64 * sizeof(int16_t));
    memcpy(compressed->data + total_blocks * 64 * sizeof(int16_t), 
           quantized_blocks[1], total_blocks * 64 * sizeof(int16_t));
    memcpy(compressed->data + 2 * total_blocks * 64 * sizeof(int16_t), 
           quantized_blocks[2], total_blocks * 64 * sizeof(int16_t));
    
    // Освобождаем память
    free(quantized_blocks[0]);
    free(quantized_blocks[1]);
    free(quantized_blocks[2]);
    free(ycbcr->Y);
    free(ycbcr->Cb);
    free(ycbcr->Cr);
    free(ycbcr);
    
    double total_time = omp_get_wtime() - start_time;
    printf("Total compression time: %.4f seconds\n", total_time);
    printf("Compression ratio: %.2fx\n", 
           (float)(img->width * img->height * img->channels) / compressed->size);
    
    return compressed;
}

// Декомпрессия с параллелизацией
Image* jpeg_decompress_parallel(CompressedImage *compressed, int num_threads) {
    printf("Starting parallel JPEG decompression with %d threads...\n", num_threads);
    double start_time = omp_get_wtime();
    
    int blocks_wide = (compressed->width + 7) / 8;
    int blocks_high = (compressed->height + 7) / 8;
    int total_blocks = blocks_wide * blocks_high;
    
    // Восстанавливаем квантованные блоки
    int16_t *quantized_blocks[3];
    quantized_blocks[0] = (int16_t*)compressed->data;
    quantized_blocks[1] = (int16_t*)(compressed->data + total_blocks * 64 * sizeof(int16_t));
    quantized_blocks[2] = (int16_t*)(compressed->data + 2 * total_blocks * 64 * sizeof(int16_t));
    
    // Подготовка таблиц квантования
    uint8_t qtable_lum[64], qtable_chr[64];
    scale_quantization_table(QUANTIZATION_TABLE_LUMINANCE, qtable_lum, compressed->quality);
    scale_quantization_table(QUANTIZATION_TABLE_CHROMINANCE, qtable_chr, compressed->quality);
    
    // Создаем YCbCr изображение
    YCbCrImage ycbcr;
    ycbcr.width = compressed->width;
    ycbcr.height = compressed->height;
    ycbcr.Y = (float*)malloc(compressed->width * compressed->height * sizeof(float));
    ycbcr.Cb = (float*)malloc(compressed->width * compressed->height * sizeof(float));
    ycbcr.Cr = (float*)malloc(compressed->width * compressed->height * sizeof(float));
    
    omp_set_num_threads(num_threads);
    
    // Параллельная обработка блоков (обратное квантование + IDCT)
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int by = 0; by < blocks_high; by++) {
        for (int bx = 0; bx < blocks_wide; bx++) {
            int block_idx = by * blocks_wide + bx;
            
            float *channels[3] = {ycbcr.Y, ycbcr.Cb, ycbcr.Cr};
            const uint8_t *qtables[3] = {qtable_lum, qtable_chr, qtable_chr};
            
            for (int c = 0; c < 3; c++) {
                float dct_block[64];
                float idct_block[64];
                
                // Обратное квантование
                dequantize_block(&quantized_blocks[c][block_idx * 64], dct_block, qtables[c]);
                
                // IDCT
                idct_8x8(dct_block, idct_block);
                
                // Записываем в изображение
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        int img_x = bx * 8 + x;
                        int img_y = by * 8 + y;
                        
                        if (img_x < ycbcr.width && img_y < ycbcr.height) {
                            channels[c][img_y * ycbcr.width + img_x] = idct_block[y * 8 + x] + 128.0f;
                        }
                    }
                }
            }
        }
    }
    
    // YCbCr -> RGB конвертация
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = compressed->width;
    img->height = compressed->height;
    img->channels = 3;
    img->data = (uint8_t*)malloc(img->width * img->height * img->channels);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < img->width * img->height; i++) {
        float Y = ycbcr.Y[i];
        float Cb = ycbcr.Cb[i] - 128.0f;
        float Cr = ycbcr.Cr[i] - 128.0f;
        
        float R = Y + 1.402f * Cr;
        float G = Y - 0.344136f * Cb - 0.714136f * Cr;
        float B = Y + 1.772f * Cb;
        
        // Clamp
        R = (R < 0) ? 0 : (R > 255) ? 255 : R;
        G = (G < 0) ? 0 : (G > 255) ? 255 : G;
        B = (B < 0) ? 0 : (B > 255) ? 255 : B;
        
        img->data[i * 3] = (uint8_t)R;
        img->data[i * 3 + 1] = (uint8_t)G;
        img->data[i * 3 + 2] = (uint8_t)B;
    }
    
    free(ycbcr.Y);
    free(ycbcr.Cb);
    free(ycbcr.Cr);
    
    double total_time = omp_get_wtime() - start_time;
    printf("Total decompression time: %.4f seconds\n", total_time);
    
    return img;
}

void free_compressed_image(CompressedImage *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

void free_image(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}
