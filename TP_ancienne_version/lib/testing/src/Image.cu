#include "Image.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void testImage() {

    // Données de l'image en noir et blanc
    const Pixel dataBW[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 16
    };

    // Données de l'image en RGB
    const PixelRGB dataRGB[] = {
        {10, 20, 30}, {40, 50, 60},
        {70, 80, 90}, {100, 110, 120},
        {0xAA, 0xBB, 0xCC}, {0xDD, 0xEE, 0xFF}
    };
    
    ImageBW imageBW;    
    ImageRGB imageRGB;

    puts("Image noir et blanc { 1...9 }:");
    imageBWInit(&imageBW, 3, 3);
    memcpy(imageBWAt(&imageBW, 0, 0), dataBW, sizeof(dataBW));
    imageBWPrint(&imageBW);
    imageBWDestroy(&imageBW);
    
    puts("Image RGB : ");
    imageRGBInit(&imageRGB, 3, 2);
    memcpy(imageRGBAt(&imageRGB, RED, 0, 0), dataRGB, sizeof(dataRGB));
    imageRGBPrint(&imageRGB);
    imageRGBDestroy(&imageRGB);
}