#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// la taille des vecteurs a traiter
#define DEF_SIZE 10000

// Pour la generation aleatoire des valeurs
#define MAX_VAL 100
#define MIN_VAL 1

// Initialisation aleatoire d'un vecteur
void vecAleatoire(float *v, int n) {
  int i;
  for(i=0;i<n;i++){
	v[i]= (float)rand()/RAND_MAX*MAX_VAL + MIN_VAL;
  }
}

// Affiche un vecteur
void vecAff(float *v, int n){
  int i;
  printf("[");
  for(i=0;i<n-1;i++) printf("%f ",v[i]);
  printf("%f]",v[n-1]);
}

// Compute vector sum C = A + B
void vecAdd(float *h_A, float *h_B, float *h_C, int n) {
  int i;
  for (i = 0; i<n; i++) h_C[i] = h_A[i] + h_B[i];
}

__global__
void vecFarhenheitToCelsius(float *d_C, float *d_F, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    float celsius = d_C[i];
    float farhenheit = celsius * 1.8000 + 32.00;
    d_F[i] = farhenheit;
  }
}

void comparer(float *gpu, float *cpu, size_t size) {
  float distance;
  size_t i;

  for(i = 0; i < size; ++i) {
    distance += fabs(gpu[i] - cpu[i]);
  }

  printf("distance of size %zx: %f\n", size, distance);
  printf("Average: %f distance per element\n", distance / size);
}

int main(int argc, char **argv) {
  // Tableaux Celsius, Farhenheit
  float *h_C, *h_F;
  float *d_C, *d_F;

  int i; 
  int size;

  if(argc > 1) {
      size = atoi(argv[1]);
  } else {
      size = DEF_SIZE;
  }

  printf("Mallocing memory\n");
  
  h_C = (float*)malloc(sizeof(float) * size);  	
  h_F = (float*)malloc(sizeof(float) * size);  	
  
  for(i = 0; i < size; ++i) {
    h_C[i] = 0;
    h_F[i] = 0;
  }
  
  cudaMalloc(&d_C, sizeof(float) * size);  	
  cudaMalloc(&d_F, sizeof(float) * size);
  
  //Random seed	
  srand(time(0)+clock()+rand());

  vecAleatoire(h_C, size);

  // Copie host -> device
  cudaMemcpy(d_C, h_C, size * sizeof(float), cudaMemcpyHostToDevice);

  // Addition CUDA
  vecFarhenheitToCelsius<<<(size + 255) / 256, 256>>>(d_C, d_F, size); 

  // Copie device -> host
  cudaMemcpy(h_F, d_F, size * sizeof(float), cudaMemcpyDeviceToHost);  

  cudaFree(d_C);
  cudaFree(d_F);

  // a desactiver pour des vecteurs de taille importante

  if(size < 100) {
    printf("Celsius: ");
    vecAff(h_C, size);

    printf("+\nFarhenheit:");
    vecAff(h_F, size);
  }

  //comparer(h_C, h_F, size);

  free(h_C);
  free(h_F);
}
