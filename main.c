#include "jpeg_parallel.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Загрузка простого PPM изображения (для тестирования)
Image* load_ppm(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    char magic[3];
    int width, height, max_val;
    fscanf(fp, "%2s\n%d %d\n%d\n", magic, &width, &height, &max_val);
    
    if (magic[0] != 'P' || magic[1] != '6') {
        fprintf(stderr, "Not a valid PPM file\n");
        fclose(fp);
        return NULL;
    }
    
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = 3;
    img->data = (uint8_t*)malloc(width * height * 3);
    
    fread(img->data, 1, width * height * 3, fp);
    fclose(fp);
    
    printf("Loaded image: %dx%d\n", width, height);
    return img;
}

// Сохранение PPM
void save_ppm(const char *filename, Image *img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->data, 1, img->width * img->height * img->channels, fp);
    fclose(fp);
    
    printf("Saved image: %s\n", filename);
}

// Генерация тестового изображения
Image* generate_test_image(int width, int height) {
    Image *img = (Image*)malloc(sizeof(Image));
    img->width = width;
    img->height = height;
    img->channels = 3;
    img->data = (uint8_t*)malloc(width * height * 3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            img->data[idx] = (x * 255) / width;         // R gradient
            img->data[idx + 1] = (y * 255) / height;    // G gradient
            img->data[idx + 2] = 128;                   // B constant
        }
    }
    
    return img;
}

// Вычисление PSNR
double calculate_psnr(Image *original, Image *reconstructed) {
    double mse = 0.0;
    int total_pixels = original->width * original->height * original->channels;
    
    for (int i = 0; i < total_pixels; i++) {
        double diff = original->data[i] - reconstructed->data[i];
        mse += diff * diff;
    }
    
    mse /= total_pixels;
    
    if (mse == 0) return INFINITY;
    
    return 10.0 * log10((255.0 * 255.0) / mse);
}

// Тестирование масштабируемости
void test_scalability(Image *img, int quality) {
    printf("\n=== Scalability Test (Quality: %d) ===\n", quality);
    printf("Image size: %dx%d\n\n", img->width, img->height);
    
    int thread_counts[] = {1, 2, 4, 8};
    double baseline_time = 0.0;
    
    for (int i = 0; i < 4; i++) {
        int threads = thread_counts[i];
        
        double start = omp_get_wtime();
        CompressedImage *compressed = jpeg_compress_parallel(img, quality, threads);
        Image *decompressed = jpeg_decompress_parallel(compressed, threads);
        double end = omp_get_wtime();
        
        double total_time = end - start;
        
        if (i == 0) {
            baseline_time = total_time;
        }
        
        double speedup = baseline_time / total_time;
        double efficiency = (speedup / threads) * 100.0;
        
        double psnr = calculate_psnr(img, decompressed);
        
        printf("Threads: %d | Time: %.4fs | Speedup: %.2fx | Efficiency: %.1f%% | PSNR: %.2f dB\n",
               threads, total_time, speedup, efficiency, psnr);
        
        free_compressed_image(compressed);
        free_image(decompressed);
    }
}

int main(int argc, char **argv) {
    printf("=== Parallel JPEG Compression Library ===\n");
    printf("OpenMP Threads available: %d\n\n", omp_get_max_threads());
    
    // Тест 1: Малое изображение
    printf("TEST 1: Small image (512x512)\n");
    Image *small_img = generate_test_image(512, 512);
    test_scalability(small_img, 90);
    free_image(small_img);
    
    // Тест 2: Среднее изображение
    printf("\n\nTEST 2: Medium image (2048x2048)\n");
    Image *medium_img = generate_test_image(2048, 2048);
    test_scalability(medium_img, 90);
    free_image(medium_img);
    
    // Тест 3: Большое изображение
    printf("\n\nTEST 3: Large image (4096x4096)\n");
    Image *large_img = generate_test_image(4096, 4096);
    test_scalability(large_img, 90);
    
    // Сохраняем результат для визуальной проверки
    CompressedImage *compressed = jpeg_compress_parallel(large_img, 90, 4);
    Image *decompressed = jpeg_decompress_parallel(compressed, 4);
    save_ppm("output_test.ppm", decompressed);
    
    free_compressed_image(compressed);
    free_image(decompressed);
    free_image(large_img);
    
    // Тест разных уровней качества
    printf("\n\n=== Quality Comparison (1024x1024, 4 threads) ===\n");
    Image *test_img = generate_test_image(1024, 1024);
    int qualities[] = {50, 75, 90, 95};
    
    for (int i = 0; i < 4; i++) {
        CompressedImage *comp = jpeg_compress_parallel(test_img, qualities[i], 4);
        Image *decomp = jpeg_decompress_parallel(comp, 4);
        double psnr = calculate_psnr(test_img, decomp);
        
        printf("Quality: %d | Size: %zu bytes | PSNR: %.2f dB\n", 
               qualities[i], comp->size, psnr);
        
        free_compressed_image(comp);
        free_image(decomp);
    }
    
    free_image(test_img);
    
    printf("\n=== All tests completed ===\n");
    return 0;
}
