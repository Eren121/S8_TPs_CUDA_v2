#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>

// Version: optimal en lecture / écriture des pixels par l'utilisation d'une mémoire partagée,
// Limitée dans la taille par 1024x1024

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define NBTHREADS_MAX 1024

#define min(a, b) ((a) <= (b) ? (a) : (b))
#define max(a, b) ((a) >= (b) ? (a) : (b))

#define check(error) { checkCudaError((error), __FILE__, __LINE__); }
void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA: %s:%d %s\n", file, line, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

// Kernel flou horizontal
// flou = taille du flou dans une direction, moins le pixel central
// Par exemple si tailleFlou = 3, alors flou = 1
__global__
void blur_hor(unsigned char *d_in, unsigned char *d_out, int flou) {
    // Pas besoin de passer la taille, on la connaît
    int w = blockDim.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    int tot[3] = {0, 0, 0}; // Int et pas unsigned char pour gérer les débordements le temps de l'accumulation des pixels voisins
    __shared__ unsigned char s[NBTHREADS_MAX][3];
    int k;
    int mink, maxk; // Positions limites des voisins
    int r; // Itérer RGB

    for(r = 0; r < 3; ++r) {
        s[j][r] = d_in[(i * w + j) * 3 + r];
    }

    __syncthreads();

    mink = max(0, j - flou);
    maxk = min(w - 1, j + flou);
    for(k = mink; k <= maxk; ++k) {

        for(r = 0; r < 3; ++r) {
            tot[r] += s[k][r];
        }
    }

    // En utilisant mink et maxk, aussi on n'a aussi pas besoin
    // de compter le nombre de pixels voisins individuellement
    for(r = 0; r < 3; ++r) {
        tot[r] /= (maxk - mink + 1);
        d_out[(i * w + j) * 3 + r] = (unsigned char)tot[r];
    }
}

// Kernel flou vertical
__global__
void blur_ver(unsigned char *d_in, unsigned char *d_out, int flou) {
    int w = gridDim.x; // On doit connaître w pour localiser l'offset dans la mémoire aux coordonnées (i, j) row-major
    int h = blockDim.x;
    int i = threadIdx.x;
    int j = blockIdx.x;
    int tot[3] = {0, 0, 0};
    __shared__ unsigned char s[NBTHREADS_MAX][3];
    int k;
    int mink, maxk;
    int r;

    for(r = 0; r < 3; ++r) {
        s[i][r] = d_in[(i * w + j) * 3 + r];
    }

    __syncthreads();

    mink = max(0, i - flou);
    maxk = min(h - 1, i + flou);

    for(k = mink; k <= maxk; ++k) {
        for(r = 0; r < 3; ++r) {
            tot[r] += s[k][r];
        }
    }

    for(r = 0; r < 3; ++r) {
        tot[r] /= (maxk - mink + 1);
        d_out[(i * w + j) * 3 + r] = (unsigned char)tot[r];
    }
}

// Wrapper du kernel
float blurGpu(int w, int h, int flou, unsigned char *h_in, unsigned char *h_out) {
    // Pas besoin d'allouer tmp sur l'hôte, seul le device l'utilise
    unsigned char *d_tmp = NULL;
    unsigned char *d_in = NULL;
    unsigned char *d_out = NULL;

    check(cudaMalloc(&d_tmp, w * h * 3));
    check(cudaMalloc(&d_in, w * h * 3));
    check(cudaMalloc(&d_out, w * h * 3));

    // Copie de l'hôte vers le device
    check(cudaMemcpy(d_in, h_in, w * h * 3, cudaMemcpyHostToDevice));

    blur_hor<<<h, w>>>(d_in, d_tmp, flou);
    blur_ver<<<w, h>>>(d_tmp, d_out, flou);
    check(cudaDeviceSynchronize());

    // Copie du device vers l'hôte
    check(cudaMemcpy(h_out, d_out, w * h * 3, cudaMemcpyDeviceToHost));

    check(cudaFree(d_out));
    check(cudaFree(d_in));
    check(cudaFree(d_tmp));
    return 0.0f;
}

int main(int argc, char **argv) {

    int w, h, comps; // comps = 3 pour RGB, 4 pour RGBA.
    unsigned char *h_in, *h_out;

    h_in = stbi_load("data/test.tga", &w, &h, &comps, 3);
    printf("Image %dx%d, %d composantes\n", w, h, comps);

    h_out = (unsigned char*)malloc(w * h * 3);
    blurGpu(w, h, 16, h_in, h_out);

    stbi_write_tga("generated/blurry_v3.tga", w, h, comps, h_out);

    free(h_out);
    free(h_in);
    return 0;
}