#include "jpeg_encoder.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

Image generate_test_image(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.data.resize(width * height * 3);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            img.data[idx]     = (x * 255) / width;
            img.data[idx + 1] = (y * 255) / height;
            img.data[idx + 2] = 128;
        }
    }
    
    return img;
}

void benchmark_threads(const Image& img, int quality) {
    std::cout << "\n=== Бенчмарк: влияние количества потоков ===" << std::endl;
    std::cout << "Изображение: " << img.width << "x" << img.height << std::endl;
    std::cout << "Качество: " << quality << std::endl;
    
    for (int threads : {1, 2, 4, 8}) {
        JPEGEncoder encoder(quality, threads);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto encoded = encoder.encode(img);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Потоков: " << threads 
                  << " | Время: " << duration.count() << " мс"
                  << " | Размер: " << encoded.size / 1024 << " КБ" << std::endl;
    }
}

void benchmark_quality(const Image& img, int threads) {
    std::cout << "\n=== Бенчмарк: влияние качества ===" << std::endl;
    std::cout << "Изображение: " << img.width << "x" << img.height << std::endl;
    std::cout << "Потоков: " << threads << std::endl;
    
    for (int quality : {10, 30, 50, 75, 95}) {
        JPEGEncoder encoder(quality, threads);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto encoded = encoder.encode(img);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Качество: " << quality 
                  << " | Время: " << duration.count() << " мс"
                  << " | Размер: " << encoded.size / 1024 << " КБ" << std::endl;
    }
}

int main() {
    std::cout << "=== Тестирование параллельной библиотеки JPEG ===" << std::endl;
    
    std::vector<std::pair<int, int>> sizes = {
        {512, 512},
        {1024, 1024},
        {2048, 2048},
        {4096, 4096}
    };
    
    for (const auto& [width, height] : sizes) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Тестирование размера: " << width << "x" << height << std::endl;
        
        auto img = generate_test_image(width, height);
        
        benchmark_threads(img, 85);
        
        if (width == 1024) {
            benchmark_quality(img, 4);
        }
    }
    
    std::cout << "\n=== Тестирование завершено ===" << std::endl;
    
    return 0;
}
