#ifndef IMAGE_H
#define IMAGE_H

#include "Array2D.h"

///
/// Images RVB (ou RGB, rouge-vert-bleu) et BW (ou NB, noir-blanc)
///

#define MAX_PIXEL 255

// Valeur d'un pixel ou d'une composante d'un pixel RGB entre 0 et 255
#define PIXEL_FORMAT_STRING %3d
#define PIXEL_RGB_FORMAT_STRING(x) (x x x)

typedef enum {
    RED = 0,
    GREEN = 1,
    BLUE = 2
} ImageRGBColorComponent;

typedef Array2D ImageRGB;
typedef Array2D ImageBW;

typedef unsigned char Pixel;
typedef Pixel PixelRGB[3];


void comparerVersions(ImageBW *imgOutCuda, ImageBW *imgOut);
void printCudaInfo();

void imageRGBInit(ImageRGB *imageRGB, size_t nRows, size_t nCols);

__host__ __device__
Pixel* imageRGBAt(ImageRGB *imageRGB, ImageRGBColorComponent component, size_t row, size_t col);

const Pixel* imageRGBGet(const ImageRGB *imageRGB, ImageRGBColorComponent component, size_t row, size_t col);
const char* imagePixelRGBToString(const void *pixelRGB);
void imageRGBPrint(const ImageRGB *imageRGB);
#define imageRGBDestroy array2DDestroy

void imageBWInit(ImageBW *imageBW, size_t nRows, size_t nCols);

__host__ __device__
Pixel* imageBWAt(ImageBW *imageBW, size_t row, size_t col);

__host__ __device__
Pixel imageBWGet(const ImageBW *imageBW, size_t row, size_t col);

const char* imagePixelBWToString(const void *pixelBW);
void imageBWPrint(const ImageBW *imageBW);
#define imageBWDestroy array2DDestroy


/// Pour la génération aléatoire d'images

void* imageBWRandomPixel(size_t row, size_t col);
void* imageRGBRandomPixel(size_t row, size_t col);

#endif