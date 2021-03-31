#include "Array2D.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

void testArray2D() {
    const int dataInt[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    const float dataFloat[] = {
        1.0f, 10.0f,
        -1.0f, -2.0f,
        999.0f, 1.234f
    };
    
    Array2D tab;    
    
    puts("Tableau { 1...9 }:");
    array2DInit(&tab, sizeof(int), 3, 3);
    memcpy(array2DAt(&tab, 0, 0), dataInt, sizeof(dataInt));
    array2DPrint(&tab, array2DIntToString);
    array2DDestroy(&tab);

    puts("Tableau de flottants:");
    array2DInit(&tab, sizeof(float), 3, 2);
    memcpy(array2DAt(&tab, 0, 0), dataFloat, sizeof(dataFloat));
    array2DPrint(&tab, array2DFloatToString);
    array2DDestroy(&tab);
}