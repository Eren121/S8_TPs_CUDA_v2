///
/// Flou sur une image en Noir et Blanc
///

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "Logging.h"
#include "Time.h"

#define DEF_SIZE 41
#define MAX_CPU_SIZE 1000

// Version du flou sur CPU
__host__
void h_blur(ImageBW *imageOut, const ImageBW *imageIn, ImageBW *tmp, size_t blurSize) {
    const int semiBlurSize = blurSize / 2;
    const size_t nRows = imageIn->nRows, nCols = imageIn->nCols;
    size_t row, col;
    int i, tmpPixel, nbBlur;
    signed int tmpCol, tmpRow;

    // Flou vertical
    for(row = 0; row < nRows; ++row) {
        for(col = 0; col < nCols; ++col) {
                
            nbBlur = 0;
            tmpPixel = 0;

            for(i = -semiBlurSize; i <= semiBlurSize; ++i) {
                
                tmpRow = row + i;
                
                if(tmpRow >= 0 && tmpRow < nRows) {
                    
                    tmpPixel += imageBWGet(imageIn, tmpRow, col);
                    ++nbBlur;
                }
            }

            assert(nbBlur > 0);
            tmpPixel /= nbBlur;

            *imageBWAt(tmp, row, col) = (Pixel)tmpPixel;
        }
    }


    // + Flou horizontal (donc flou final)
    for(row = 0; row < nRows; ++row) {
        for(col = 0; col < nCols; ++col) {
                
            nbBlur = 0;
            tmpPixel = 0;

            for(i = -semiBlurSize; i <= semiBlurSize; ++i) {
                
                tmpCol = col + i;
                
                if(tmpCol >= 0 && tmpCol < nCols) {
                    
                    tmpPixel += imageBWGet(tmp, row, tmpCol);
                    ++nbBlur;
                }
            }

            assert(nbBlur > 0);
            tmpPixel /= nbBlur;
            
            *imageBWAt(imageOut, row, col) = (Pixel)tmpPixel;
        }
    }
}

void* imageBWCheckerboard(size_t row, size_t col) {
    static Pixel pixel;
    pixel = (row + col) % 2;
    return &pixel;
}

typedef enum {
    BLUR_VERTICAL,
    BLUR_HORIZONTAL
} BlurDirection;

// Version du flou sur GPU
// Comme on ne peut pas synchroniser tous les blocks ensemble,
// On devra appeler le blur verticalement puis horizontalement
__global__
void d_blur(ImageBW imageOut, const ImageBW imageIn, size_t blurSize, BlurDirection blurDirection) {
    const int semiBlurSize = blurSize / 2;
    const size_t nRows = imageIn.nRows, nCols = imageIn.nCols;
    int i, tmpPixel, nbBlur;
    signed int tmpCol, tmpRow;

    // De quel pixel s'occupe le thread actuel
    const size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    // Attention à bien être dans l'image
    // Ne pas faire de return; car invalide avec __syncthreads() !

    if(blurDirection == BLUR_VERTICAL && row < nRows && col < nCols) {
        
        // Flou vertical

        nbBlur = 0;
        tmpPixel = 0;

        for(i = -semiBlurSize; i <= semiBlurSize; ++i) {
            
            tmpRow = row + i;
            
            if(tmpRow >= 0 && tmpRow < nRows) {
                
                tmpPixel += imageBWGet(&imageIn, tmpRow, col);
                ++nbBlur;
            }
        }

        assert(nbBlur > 0);
        tmpPixel /= nbBlur;

        *imageBWAt(&imageOut, row, col) = (Pixel)tmpPixel;
    }
    
    else if(blurDirection == BLUR_HORIZONTAL && row < nRows && col < nCols) {

        // + Flou horizontal (donc flou final)

        nbBlur = 0;
        tmpPixel = 0;

        for(i = -semiBlurSize; i <= semiBlurSize; ++i) {
            
            tmpCol = col + i;
            
            if(tmpCol >= 0 && tmpCol < nCols) {
                
                tmpPixel += imageBWGet(&imageIn, row, tmpCol);
                ++nbBlur;
            }
        }

        assert(nbBlur > 0);
        tmpPixel /= nbBlur;
        
        *imageBWAt(&imageOut, row, col) = (Pixel)tmpPixel;
    }
}

// Wrapper le call CUDA dans une fonction
// Plus simple car on doit appeler 2 fois le kernel
// Retourne le nombre de secondes écoulées
float blurCuda(ImageBW *imgOutCuda, ImageBW *imgIn, ImageBW *imgTmp, size_t blurSize) {
    const size_t size = imgIn->nRows;
    const size_t blockSize = 16;
    const dim3 gridDim((size + blockSize - 1) / blockSize, (size + blockSize - 1) / blockSize);
    const dim3 blockDim(blockSize, blockSize);

    float millis;
    cudaEvent_t start, stop;
  
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    // Envoi des paramètres
    array2DCudaSetup(imgOutCuda, false);
    array2DCudaSetup(imgIn, true);
    array2DCudaSetup(imgTmp, false);

    // Launch kernel
    cudaCheck(cudaEventRecord(start));
    d_blur<<<gridDim, blockDim>>>(*imgTmp, *imgIn, blurSize, BLUR_HORIZONTAL);
    d_blur<<<gridDim, blockDim>>>(*imgOutCuda, *imgTmp, blurSize, BLUR_VERTICAL);
    cudaCheck(cudaEventRecord(stop));
  
    // Récupération du résultat
    array2DCudaFinalize(imgOutCuda, true);
    array2DCudaFinalize(imgIn, false);
    array2DCudaFinalize(imgTmp, false);

    // Aficher le temps écoulé
    cudaCheck(cudaEventSynchronize(stop));
    cudaCheck(cudaEventElapsedTime(&millis, start, stop));
    printf("Temps écoulé avec CUDA: %fs\n", millis / 1000.0f);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));

    return millis;
}

// Premier argument: taille de l'image (DEF_SIZE si l'argument n'est pas renseigné)
int main(int argc, char **argv) {
    /// Init
    const size_t blurSize = 10;
    size_t size;
    
    float millis;
    Elapsed elapsed; // pour la mesure du temps CPU
  
    ImageBW imgIn, imgTmp;
    ImageBW imgOut, imgOutCuda; // Résultats pour les versions normales et CUDA
    
    if(argc > 1) {
        size = atoi(argv[1]);
    } else {
        size = DEF_SIZE;
    }
    
    printCudaInfo();
  
    printf("Mallocing memory\n");
  
    imageBWInit(&imgIn, size, size);
    imageBWInit(&imgTmp, size, size);
    imageBWInit(&imgOut, size, size);
    imageBWInit(&imgOutCuda, size, size);
    
    // Random seed
    srand(time(0) + clock() + rand());
    array2DGenerate(&imgIn, imageBWRandomPixel);
  
    puts("Image d'entrée à flouter:");
    imageBWPrint(&imgIn);
  
    //// Conversion CPU d'abord pour vérifier
  
    if(size <= MAX_CPU_SIZE ) {
      elapsedFrom(&elapsed);
      h_blur(&imgOut, &imgIn, &imgTmp, blurSize);
      elapsedTo(&elapsed);

      puts("[Version CPU] Image noir et blanc résultante de la conversion:");
      imageBWPrint(&imgOut);
      elapsedPrint(&elapsed);  
    }
    else {
      printf("L'image est très grande, évitons d'exécuter sur CPU...\n");
    }
  
    //// Conversion CUDA

    // Launch kernel
    millis = blurCuda(&imgOutCuda, &imgIn, &imgTmp, blurSize);

    // Afficher le résultat
    puts("[Version CUDA] Image noir et blanc résultante de la conversion:");
    imageBWPrint(&imgOutCuda);

    // Aficher le temps écoulé
    printf("Temps écoulé avec CUDA: %fs\n", millis / 1000.0f);
  
    comparerVersions(&imgOutCuda, &imgOut);
  
    //// Free memory

    imageBWDestroy(&imgIn);
    imageBWDestroy(&imgOut);
    imageBWDestroy(&imgOutCuda);
    imageBWDestroy(&imgTmp);

    return EXIT_SUCCESS;
  }