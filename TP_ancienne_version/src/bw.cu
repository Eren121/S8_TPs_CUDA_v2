#include <stdio.h>
#include <stdlib.h>
#include "Image.h"
#include "Logging.h"
#include "Time.h"

// 100000^2 B = 10GB
#define DEF_SIZE 10000

///
/// Convertit une image de RVB (rouge-vert-bleu) vers NB (noir-blanc)
///

// Mixe les valeurs RGB selon la méthode voulue
__host__ __device__
Pixel pixelToGrey(PixelRGB pixel) {
  const Pixel red = pixel[RED];
  const Pixel green = pixel[GREEN];
  const Pixel blue = pixel[BLUE];

  return (red + green + blue) / 3;
}

// Version host (CPU)
__host__
void h_imageRGBToBW(ImageBW *imgOut, ImageRGB *imgIn) {
  size_t row, col;

  for(row = 0; row < imgIn->nRows; ++row) {
    for(col = 0; col < imgIn->nCols; ++col) {
      *imageBWAt(imgOut, row, col) = pixelToGrey(imageRGBAt(imgIn, RED, row, col));
    }
  }
}

// Version device (CUDA => GPU)
// On passe par copie
// Car CUDA s'occupe de la copie des pointeurs de la structure
// Comme la structure est petite, le coût est faible
// à éviter si la structure est trop grande (sans compter les données pointées mais juste la structure en elle-même)
__global__
void d_imageRGBToBW(ImageBW imgOut, ImageRGB imgIn) {
  const size_t nRows = imgIn.nRows, nCols = imgIn.nCols;
  size_t row, col;

  row = blockDim.y * blockIdx.y + threadIdx.y;
  col = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < nRows && col < nCols) {
    *imageBWAt(&imgOut, row, col) = pixelToGrey(imageRGBAt(&imgIn, RED, row, col));
  }
}

int main(int argc, char **argv) {
  /// Init
  size_t size;

  if(argc > 1) {
      size = atoi(argv[1]);
  } else {
      size = DEF_SIZE;
  }

  Elapsed elapsed; // pour la mesure du temps CPU

  float millis;
  cudaEvent_t start, stop; // pour la mesure du temps GPU

  ImageRGB imgIn;
  ImageBW imgOut, imgOutCuda; // Résultats pour les versions normales et CUDA
  Pixel *d_pixelsBW = NULL, *h_pixelsBW = NULL;
  PixelRGB *d_pixelsRGB = NULL, *h_pixelsRGB = NULL;

  printCudaInfo();

  printf("Mallocing memory\n");

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  imageRGBInit(&imgIn, size, size);
  imageBWInit(&imgOut, size, size);
  imageBWInit(&imgOutCuda, size, size);

  // On ne copie pas la structure, on actualise juste les pointeurs internes à la structure
  // Pour la copie des données (plus simple)

  cudaCheck(cudaMalloc(&d_pixelsRGB, array2DSizeof(&imgIn)));
  cudaCheck(cudaMalloc(&d_pixelsBW, array2DSizeof(&imgOut)));
  
  // Random seed
  srand(time(0) + clock() + rand());
  array2DGenerate(&imgIn, imageRGBRandomPixel);

  puts("Image RGB d'entrée à transformer en noir et blanc:");
  imageRGBPrint(&imgIn);

  //// Conversion CPU d'abord pour vérifier

  if(size < 1000) {
    elapsedFrom(&elapsed);
    h_imageRGBToBW(&imgOut, &imgIn);
    elapsedTo(&elapsed);

    puts("[Version CPU] Image noir et blanc résultante de la conversion:");
    imageBWPrint(&imgOut);
    elapsedPrint(&elapsed);  
  }
  else {
    printf("L'image est très grande, évitons d'exécuter sur CPU...\n");
  }

  //// Conversion CUDA

  // Sauvegarde des pointeurs d'origine
  h_pixelsRGB = (PixelRGB*)imgIn.data;
  h_pixelsBW = (Pixel*)imgOutCuda.data;

  // Pas besoin d'envoyer les données NB car c'est la sortie
  cudaCheck(cudaMemcpy((void*)d_pixelsRGB, h_pixelsRGB, array2DSizeof(&imgIn), cudaMemcpyHostToDevice));

  // Adapter les structures pour le passage au kernel
  imgIn.data = (unsigned char*)d_pixelsRGB;
  imgOutCuda.data = (unsigned char*)d_pixelsBW;

  // Launch kernel
  size_t blockSize = 16;
  dim3 gridDim((size + blockSize - 1) / blockSize, (size + blockSize - 1) / blockSize);
  dim3 blockDim(blockSize, blockSize);

  cudaCheck(cudaEventRecord(start));
  d_imageRGBToBW<<<gridDim, blockDim>>>(imgOutCuda, imgIn);
  cudaCheck(cudaEventRecord(stop));

  // Récupération du résultat
  cudaCheck(cudaMemcpy((void*)h_pixelsBW, d_pixelsBW, array2DSizeof(&imgOutCuda), cudaMemcpyDeviceToHost));

  cudaCheck(cudaEventSynchronize(stop));
  cudaCheck(cudaEventElapsedTime(&millis, start, stop));
  printf("Temps écoulé avec CUDA: %fs\n", millis / 1000.0f);

  // Récupération des pointeurs d'origine
  imgIn.data = (unsigned char*)h_pixelsRGB;
  imgOutCuda.data = (unsigned char*)h_pixelsBW;

  puts("[Version CUDA] Image noir et blanc résultante de la conversion:");
  imageBWPrint(&imgOutCuda);

  comparerVersions(&imgOutCuda, &imgOut);

  //// Free memory

  cudaCheck(cudaFree(d_pixelsRGB));
  cudaCheck(cudaFree(d_pixelsBW));
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // ne pas détruire ici imgOutCuda, car doit être utilisé avec cudaFree

  imageRGBDestroy(&imgIn);
  imageBWDestroy(&imgOut);
  imageBWDestroy(&imgOutCuda);

  return EXIT_SUCCESS;
}
