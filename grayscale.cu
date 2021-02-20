#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>

#define R 255, 0, 0
#define G 0, 255, 0
#define B 0, 0, 255

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// nbre de threads dans une dimension
#define NBTHREADS 1024

#define check(error) { checkCudaError((error), __FILE__, __LINE__); }
void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA: %s:%d %s\n", file, line, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

// Version CPU
float grayscaleCpu(int w, int h, unsigned char *in, unsigned char *out) {
    float secs;
    struct timeb tap, tav;
    ftime(&tav);

    int i, j, off;
    unsigned char r, g, b, mix;

    for(i = 0; i < w; ++i) {
        for(j = 0; j < h; ++j) {
            off = (i * h + j);
            r = in[off * 3];
            g = in[off * 3 + 1];
            b = in[off * 3 + 2];

            mix = (r + g + b) / 3;
            out[off] = mix;
        }
    }

    ftime(&tap);
    secs = (double)((tap.time*1000+tap.millitm)-(tav.time*1000+tav.millitm))/1000 ;
    return secs;
};

// Kernel
__global__
void grayscale(int w, int h, unsigned char *in, unsigned char *out) {
    int i, j, off;
    unsigned char r, g, b, mix;

    i = threadIdx.y + blockDim.y * blockIdx.y;
    j = threadIdx.x + blockDim.x * blockIdx.x;
    off = (i * h + j);

    if(off < w * h) {
        r = in[off * 3];
        g = in[off * 3 + 1];
        b = in[off * 3 + 2];

        mix = (r + g + b) / 3;
        out[off] = mix;
    }
}

// Wrapper du Kernel
float grayscaleGpu(int w, int h, unsigned char *h_in, unsigned char *h_out) {
    cudaEvent_t start, stop;
    float millis = 0;
    int nbBlocks;
    int n = w * h;
    unsigned char *d_in, *d_out;


    check(cudaMalloc(&d_in, n * 3));
    check(cudaMalloc(&d_out, n));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    nbBlocks = (n - 1) / NBTHREADS + 1;

    check(cudaMemcpy(d_in, h_in, n * 3, cudaMemcpyHostToDevice));

    check(cudaEventRecord(start));
    grayscale<<<nbBlocks, NBTHREADS>>>(w, h, d_in, d_out);
    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&millis, start, stop));

    check(cudaMemcpy(h_out, d_out, n, cudaMemcpyDeviceToHost));


    check(cudaEventDestroy(stop));
    check(cudaEventDestroy(start));
    check(cudaFree(d_out));
    check(cudaFree(d_in));

    return millis / 1000.0f;
}

int main(int argc, char **argv) {
    int w, h, comps; // comps = 3 pour RGB, 4 pour RGBA
    unsigned char *in, *out;

    in = stbi_load("data/test.tga", &w, &h, &comps, 3);
    printf("Image %dx%d, %d composantes\n", w, h, comps);

    out = (unsigned char*)malloc(w * h);


    printf("Temps CPU: %.3f\n", grayscaleCpu(w, h, in, out));
    stbi_write_tga("grayscaled_cpu.tga", w, h, 1, out);

    // Reset pour être sûr que c'est bien CUDA qui traite l'image
    memset(out, 0, w * h);

    printf("Temps GPU: %.3f\n", grayscaleGpu(w, h, in, out));
    stbi_write_tga("grayscaled_gpu.tga", w, h, 1, out);

    free(out);
    free(in);
    return 0;
}