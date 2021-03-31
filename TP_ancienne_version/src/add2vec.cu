#include <stdio.h>
#include <stdlib.h>

// la taille des vecteurs a traiter
#define SIZE 10

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
void vecAddCuda(float *d_A, float *d_B, float *d_C, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) d_C[i] = d_A[i] + d_B[i];
}

int main() {
  // Memory allocation for h_A, h_B, and h_C
  // I/O to read h_A and h_B, N elements
  float *h_A, *h_B, *h_C;
  float *d_A, *d_B, *d_C;
  int i; 
  int size=SIZE ;
  printf("Mallocing memory\n");
  h_A=(float*)malloc(sizeof(float)*size);  	
  h_B=(float*)malloc(sizeof(float)*size);  	
  h_C=(float*)malloc(sizeof(float)*size);  	

  for(i = 0; i < size; ++i) {
    h_A[i] = 0;
    h_B[i] = 0;
    h_C[i] = 0;
  }
  
  cudaMalloc(&d_A, sizeof(float)*size);  	
  cudaMalloc(&d_B, sizeof(float)*size);  	
  cudaMalloc(&d_C, sizeof(float)*size);  	
  
  // generation aleatoire des vecteurs A et B
  //Random seed	
  srand(time(0)+clock()+rand());

  vecAleatoire(h_A,size);
  vecAleatoire(h_B,size);
  //vecAdd(h_A, h_B, h_C, size);

  // Copie host -> device
  cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);  
  cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);  

  // Addition CUDA
  vecAddCuda<<<(size+255)/256, 256>>>(d_A, d_B, d_C, size); 

  // Copie device -> host
  cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);  

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // a desactiver pour des vecteurs de taille importante
  vecAff(h_A,size);
  printf("+\n");
  vecAff(h_B,size);
  printf("=\n");
  vecAff(h_C,size);
  
}
