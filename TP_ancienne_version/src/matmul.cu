///
/// Multiplication de matrices
///

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "Logging.h"
#include "Time.h"

#define DEF_SIZE 41
#define MAX_CPU_SIZE 1000

// Pour la generation aleatoire des valeurs
#define MAX_VAL 100
#define MIN_VAL 1

// Calcul a = b * c
__host__
void h_matmul(Array2D a, Array2D b, Array2D c) {
    const size_t nRows = a.nRows, nCols = a.nCols, count = b.nCols;
    size_t i, j, k;
    float tmpValue, aValue, bValue;

    assert(b.nCols == c.nRows);
    assert(a.nRows == b.nRows);
    assert(a.nCols == c.nCols);

    for(i = 0; i < nRows; ++i) {
        
        for(j = 0; j < nCols; ++j) {
            tmpValue = 0;
        
            for(k = 0; k < count; ++k) {
                aValue = *(float*)array2DAt(&b, i, k);
                bValue = *(float*)array2DAt(&c, k, j);

                tmpValue += aValue * bValue;
            }
            
            *(float*)array2DAt(&a, i, j) = tmpValue;
        }
    }
}

__global__
void d_matmul(Array2D a, Array2D b, Array2D c) {
    const size_t nRows = a.nRows, nCols = a.nCols, count = b.nCols;
    const size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k;
    float tmpValue, aValue, bValue;

    if(i < nRows && j < nCols) {

        for(k = 0; k < count; ++k) {

            aValue = *(float*)array2DAt(&b, i, k);
            bValue = *(float*)array2DAt(&c, k, j);

            tmpValue += aValue * bValue;
        }

        *(float*)array2DAt(&a, i, j) = tmpValue;
    }
}

void* fillRandomly(size_t row, size_t col) {
    static float value;
	value = (float)rand() / RAND_MAX * MAX_VAL + MIN_VAL;
    return &value;
}


// Wrapper le call CUDA dans une fonction
// retourne le temps d'exécution en millisecondes
float matmulCuda(Array2D a, Array2D b, Array2D c) {
    const size_t nRows = a.nRows, nCols = a.nCols;
    const size_t blockSize = 16, sizeX = nCols, sizeY = nRows;
    const dim3 gridDim((sizeX + blockSize - 1) / blockSize, (sizeY + blockSize - 1) / blockSize);
    const dim3 blockDim(blockSize, blockSize);

    float millis;
    cudaEvent_t start, stop;
  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    // Envoi des paramètres
    array2DCudaSetup(&a, false);
    array2DCudaSetup(&b, true);
    array2DCudaSetup(&c, true);

    // Launch kernel
    cudaCheck(cudaEventRecord(start));
    d_matmul<<<gridDim, blockDim>>>(a, b, c);
    cudaCheck(cudaEventRecord(stop));
  
    // Récupération du résultat
    // permet aussi de libérer la mémoire GPU
    array2DCudaFinalize(&a, true);
    array2DCudaFinalize(&b, false);
    array2DCudaFinalize(&c, false);

    // Aficher le temps écoulé
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&millis, start, stop));
    printf("Temps écoulé avec CUDA: %fs\n", millis / 1000.0f);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    return millis;
}


int main(int argc, char **argv) {

    /// Init

    size_t size;    
    Elapsed elapsed; // pour la mesure du temps CPU
  
    Array2D h_a, b, c;
    Array2D d_a;

    if(argc > 1) {
        size = atoi(argv[1]);
    } else {
        size = DEF_SIZE;
    }
    
    printCudaInfo();
  
    printf("Mallocing memory\n");
    
    // Init matrices
    array2DInit(&h_a, sizeof(float), size, size);
    array2DInit(&d_a, sizeof(float), size, size);
    array2DInit(&b, sizeof(float), size, size);
    array2DInit(&c, sizeof(float), size, size);

    // Fill the matrices randomly
    srand(time(0) + clock() + rand());
    array2DGenerate(&b, fillRandomly);
    array2DGenerate(&c, fillRandomly);
  
    printf("b = \n");
    array2DPrint(&b, array2DFloatToString);
  
    printf("c = \n");
    array2DPrint(&c, array2DFloatToString);

    //// Conversion CPU d'abord pour vérifier
  
    if(size <= MAX_CPU_SIZE ) {
      elapsedFrom(&elapsed);
      h_matmul(h_a, b, c);
      elapsedTo(&elapsed);

      puts("[Version CPU] a = ");
      array2DPrint(&h_a, array2DFloatToString);
      elapsedPrint(&elapsed);  
    }
    else {
      printf("La matrice est très grande, évitons d'exécuter sur CPU...\n");
    }
  
    //// Conversion CUDA

    // Launch kernel
    matmulCuda(d_a, b, c);

    // Afficher le résultat
    puts("[Version CUDA] a = ");
    array2DPrint(&d_a, array2DFloatToString);

    array2DCompare_float(&d_a, &h_a);
    
    //// Free memory

    imageBWDestroy(&h_a);
    imageBWDestroy(&d_a);
    imageBWDestroy(&b);
    imageBWDestroy(&c);

    return EXIT_SUCCESS;
  }