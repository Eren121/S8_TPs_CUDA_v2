#include "Image.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_DIFF_PRINT 30

void comparerVersions(ImageBW *imgOutCuda, ImageBW *imgOut) {
    const size_t nRows = imgOut->nRows, nCols = imgOut->nCols;
    int distance = 0;
    int tmpCuda, tmpCpu;
    int tmpDiff;
    size_t row, col;
    size_t nbDiffs = 0;
  
    for(row = 0; row < nRows; ++row) {
      for(col = 0; col < nCols; ++col) {
        tmpCuda = *imageBWAt(imgOutCuda, row, col);
        tmpCpu = *imageBWAt(imgOut, row, col);
        tmpDiff = abs(tmpCuda - tmpCpu);

        if(nbDiffs < MAX_DIFF_PRINT + 1 && tmpDiff != 0) {
            ++nbDiffs;
            
            if(nbDiffs == MAX_DIFF_PRINT + 1) {
                printf("...\n");
            }
            else {
                printf("Difference [row: %zu, col: %zu]: CUDA=%d, CPU=%d.\n", row, col, tmpCuda, tmpCpu);
                distance += tmpDiff;    
            }
        }
      }
    }
  
    puts("Comparaison des deux méthodes...");
    printf("Distance d'erreur: %d\n", distance);
}
  
void printCudaInfo() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
  
    printf("Free CUDA memory / Total CUDA memory: %zu/%zu\n", free, total);
}

  
void imageRGBInit(ImageRGB *imageRGB, size_t nRows, size_t nCols) {
    array2DInit(imageRGB, sizeof(PixelRGB), nRows, nCols);
}

__host__ __device__
Pixel* imageRGBAt(ImageRGB *imageRGB, ImageRGBColorComponent component, size_t row, size_t col) {
    assert(component >= 0 && component < 3);

    Pixel *pixelRGB = (Pixel*)array2DAt(imageRGB, row, col);
    return &pixelRGB[component];
}

const Pixel* imageRGBGet(const ImageRGB *imageRGB, ImageRGBColorComponent component, size_t row, size_t col) {
    assert(component >= 0 && component < 3);

    const Pixel *pixelRGB = (const Pixel*)array2DGet(imageRGB, row, col);
    return &pixelRGB[component];
}

const char* imagePixelRGBToString(const void *pixelRGBPointer) {
    static const char format[] = STR(PIXEL_RGB_FORMAT_STRING(PIXEL_FORMAT_STRING));
    const Pixel *pixelRGB = (const Pixel*)pixelRGBPointer;
    
    snprintf(array2DTmpString, sizeof(array2DTmpString),
        format,
        pixelRGB[RED], pixelRGB[GREEN], pixelRGB[BLUE]);
    
    return array2DTmpString;
}

void imageRGBPrint(const ImageRGB *imageRGB) {
    array2DPrint(imageRGB, imagePixelRGBToString);
}

void imageBWInit(ImageBW *imageBW, size_t nRows, size_t nCols) {
    array2DInit(imageBW, sizeof(Pixel), nRows, nCols);
}

__host__ __device__
Pixel* imageBWAt(ImageBW *imageBW, size_t row, size_t col) {
    Pixel *pixel = (Pixel*)array2DAt(imageBW, row, col);
    return pixel;
}

__host__ __device__
Pixel imageBWGet(const ImageBW *imageBW, size_t row, size_t col) {
    Pixel pixel = *(const Pixel*)array2DGet(imageBW, row, col);
    return pixel;
}

void imageBWPrint(const ImageBW *imageBW) {
    array2DPrint(imageBW, imagePixelBWToString);
}

const char* imagePixelBWToString(const void *pixelBWPointer) {
    const Pixel pixel = *(Pixel*)pixelBWPointer;
    
    snprintf(array2DTmpString, sizeof(array2DTmpString),
        STR(PIXEL_FORMAT_STRING),
        pixel);
    
    return array2DTmpString;
}

void* imageBWRandomPixel(size_t row, size_t col) {
    static Pixel pixel;
    pixel = (Pixel)((float)rand() / RAND_MAX * MAX_PIXEL);
    return &pixel;
}

void* imageRGBRandomPixel(size_t row, size_t col) {
    static PixelRGB pixelRGB;
    pixelRGB[RED] = (Pixel)((float)rand() / RAND_MAX * MAX_PIXEL);
    pixelRGB[GREEN] = (Pixel)((float)rand() / RAND_MAX * MAX_PIXEL);
    pixelRGB[BLUE] = (Pixel)((float)rand() / RAND_MAX * MAX_PIXEL);
    return &pixelRGB;
}