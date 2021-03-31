#include "Array2D.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "Logging.h"

#define MAX_DIFF_PRINT 30

char array2DTmpString[ARRAY_2D_TMP_STRING_SIZE];

void array2DInit(Array2D *tab, size_t elemSize, size_t nRows, size_t nCols) {
    assert(tab);
    
    tab->elemSize = elemSize;
    tab->nRows = nRows;
    tab->nCols = nCols;

    tab->data = (unsigned char*)malloc(elemSize * nRows * nCols);
    if(!tab->data) {
        fatalMalloc(tab->data);
    }

    tab->hostData = NULL;
}

void array2DPrint(const Array2D *tab, const char* (*elemToString)(const void *elem)) {
    const size_t maxRow = ARRAY_2D_MAX_ELEMS_LINE, maxCol = ARRAY_2D_MAX_ELEMS_LINE; // Do not print if there is too much data
    const size_t nCols = tab->nCols, nRows = tab->nRows;
    size_t col, row;
    const char* elemAsString = NULL;

    fputs("[", stdout);

    for(row = 0; row < nRows; ++row) {

        if(row >= maxRow) {
            printf(" ... %zu more rows", nRows - row);
            break;
        }
        
        fputs("[", stdout);

        for(col = 0; col < nCols; ++col) {

            if(col >= maxCol) {
                printf(" ... %zu more columns", nCols - col);
                break;
            }

            if(col != 0) { fputs(" ", stdout); }

            elemAsString = (const char*)elemToString(array2DGet(tab, row, col));
            fputs(elemAsString, stdout);
        }

        fputs("]", stdout);
        if(row != nRows - 1) { fputs(",\n ", stdout); }
    }
    
    puts("]");
}


size_t array2DSizeof(const Array2D *tab) {
    return tab->elemSize * tab->nCols * tab->nRows;
}

void array2DGenerate(Array2D *tab, void* (*generator)(size_t row, size_t col)) {
    size_t row, col;
    
    for(row = 0; row < tab->nRows; ++row) {
        for(col = 0; col < tab->nCols; ++col) {
            memcpy(array2DAt(tab, row, col), generator(row, col), tab->elemSize);
        }
    }
}

__host__ __device__
void* array2DAt(Array2D *tab, size_t row, size_t col) {
    const size_t nCols = tab->nCols, nRows = tab->nRows, elemSize = tab->elemSize;

    assert(row < nRows);
    assert(col < nCols);

    return &tab->data[elemSize * (row * nCols + col)];
}

__host__ __device__
const void* array2DGet(const Array2D *tab, size_t row, size_t col) {
    const size_t nCols = tab->nCols, nRows = tab->nRows, elemSize = tab->elemSize;

    assert(row < nRows);
    assert(col < nCols);

    return &tab->data[elemSize * (row * nCols + col)];
}

void array2DDestroy(Array2D *tab) {
    assert(tab);

    free(tab->data);
    tab->data = NULL;

    tab->nCols = 0;
    tab->nRows = 0;
    tab->elemSize = 0;
}

const char* array2DIntToString(const void *intPointer) {
    int elem = *(int*)intPointer;
    snprintf(array2DTmpString, sizeof(array2DTmpString), "%d", elem);
    return array2DTmpString;
}

const char* array2DFloatToString(const void *floatPointer) {
    float elem = *(float*)floatPointer;
    snprintf(array2DTmpString, sizeof(array2DTmpString), "%f", elem);
    return array2DTmpString;
}


void array2DCudaSetup(Array2D *tab, bool load) {
    assert(tab->hostData == NULL);

    const size_t nBytes = array2DSizeof(tab);

    tab->hostData = tab->data;

    cudaCheck(cudaMalloc(&tab->data, nBytes));

    if(load) {
        cudaCheck(cudaMemcpy(tab->data, tab->hostData, nBytes, cudaMemcpyHostToDevice));
    }
}

void array2DCudaFinalize(Array2D *tab, bool save) {
    assert(tab->hostData != NULL);

    const size_t nBytes = array2DSizeof(tab);

    if(save) {
        cudaCheck(cudaMemcpy(tab->hostData, tab->data, nBytes, cudaMemcpyDeviceToHost));
    }

    cudaCheck(cudaFree(tab->data));

    tab->data = tab->hostData;
    tab->hostData = NULL;
}
 
void array2DCompare_float(Array2D *gpu, Array2D *cpu) {
    const size_t nRows = gpu->nRows, nCols = gpu->nCols;
    float distance = 0;
    float tmpCuda, tmpCpu;
    float tmpDiff;
    size_t row, col;
    size_t nbDiffs = 0;
  
    for(row = 0; row < nRows; ++row) {
      for(col = 0; col < nCols; ++col) {
        tmpCuda = *(float*)array2DAt(gpu, row, col);
        tmpCpu = *(float*)array2DAt(cpu, row, col);
        tmpDiff = fabs(tmpCuda - tmpCpu);

        if(nbDiffs < MAX_DIFF_PRINT + 1 && tmpDiff != 0) {
            ++nbDiffs;
            
            if(nbDiffs == MAX_DIFF_PRINT + 1) {
                printf("...\n");
            }
            else {
                printf("Difference [row: %zu, col: %zu]: CUDA=%f, CPU=%f.\n", row, col, tmpCuda, tmpCpu);
                distance += tmpDiff;    
            }
        }
      }
    }
  
    puts("Comparaison des deux méthodes...");
    printf("Distance d'erreur: %f\n", distance);
}