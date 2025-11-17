#ifndef JPEG_PARALLEL_H
#define JPEG_PARALLEL_H

#include <stdint.h>
#include <stdbool.h>

// Структура для хранения изображения
typedef struct {
    uint8_t *data;      // RGB данные
    int width;
    int height;
    int channels;       // 3 для RGB
} Image;

// Структура для хранения YCbCr данных
typedef struct {
    float *Y;           // Яркость
    float *Cb;          // Синяя цветоразность
    float *Cr;          // Красная цветоразность
    int width;
    int height;
} YCbCrImage;

// Структура для сжатых данных
typedef struct {
    uint8_t *data;
    size_t size;
    int width;
    int height;
    int quality;
} CompressedImage;

// Основные функции
CompressedImage* jpeg_compress_parallel(Image *img, int quality, int num_threads);
Image* jpeg_decompress_parallel(CompressedImage *compressed, int num_threads);
void free_compressed_image(CompressedImage *img);
void free_image(Image *img);

#endif
