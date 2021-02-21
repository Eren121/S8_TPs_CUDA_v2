#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>


// Hypercube
// Version: optimisé pour une partagée
// On sépare le gros tableau en multiples sous-tableaux qu'on réduit de la même manière
// Pas limité en taille

#define pow2(x) (1<<(x))

// Nombre de threads par bloc
#define NBTHREADS_MAX 1024

#define check(error) { checkCudaError((error), __FILE__, __LINE__); }
void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA: %s:%d %s\n", file, line, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

// Kernel de la somme hypercube
// stride: puissances de NBTHREADS pour savoir à quelle itération on réduit
__global__
void kernel_hypercube(int *t, int tailleTotal, int stride) {
    __shared__ int s[NBTHREADS_MAX];

    const int off = stride * blockIdx.x * blockDim.x; // Position absolue de la 1ière case pour ce block
    const int x = threadIdx.x;
    const int nbits = (int)ceil(log2((double)blockDim.x));
    int d, mask, in, out;

    // Chaque thread copie dans la mémoire partagée
    if(threadIdx.x * stride + off < tailleTotal) {
        s[threadIdx.x] = t[threadIdx.x * stride + off];
    }
    else {
        // on met 0, comme ça quand on réduit, cela ne change pas le résultat sans avoir besoin de vérifier les bords
        // (possiblement en dehors pour le dernier block uniquement)
        s[threadIdx.x] = 0;
    }

    __syncthreads();

    // Réduction optimisée sur 1024 éléments
    for(d = 1; d <= nbits; ++d) {
        if (x < pow2(nbits - d)) {
            mask = x << d;
            in = mask | pow2(d - 1);
            out = mask;
            s[out] += s[in];
        }

        // On doit synchroniser mêmes les threads en dehors sinon deadlock
        __syncthreads();
    }

    // Copie de la somme total de s dans la 1ière case du tableau
    if(threadIdx.x == 0) {
        t[off] = s[0];
    }
}

// Wrap l'appel du kernel
// Retourne le nombre de millisecondes écoulées
float hypercube(int *h_t, int taille) {
    float millis;
    int nbBlocks;
    int nBytes = sizeof(int) * taille;
    int *d_t = NULL;
    int stride = 1;
    cudaEvent_t start, stop;

    check(cudaMalloc(&d_t, nBytes));
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaMemcpy(d_t, h_t, nBytes, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    do {

        printf("Réduction: stride=%d\n", stride);
        nbBlocks = (int)ceil((double)taille / stride / NBTHREADS_MAX);
        printf("nbBlocks=%d\n", nbBlocks);
        kernel_hypercube<<<nbBlocks, NBTHREADS_MAX>>>(d_t, taille, stride);
        check(cudaDeviceSynchronize());
        stride *= NBTHREADS_MAX;
    } while(stride < taille);

    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&millis, start, stop));

    check(cudaMemcpy(h_t, d_t, nBytes, cudaMemcpyDeviceToHost));


    check(cudaFree(d_t));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    return millis;
}

// Somme séquentielle
int somme(int *arr, int taille) {
    long i, tot = 0;
    for(i = 0; i < taille; ++i) {
        tot += arr[i];
    }

    return tot;
}

void fillRandomly(int *t, int taille) {
    int i;
    for(i = 0; i < taille; ++i) {
        t[i] = rand() % 3;
    }
}

void printArr(int *t, int taille) {
    int i;
    for(i = 0; i < taille; ++i) {
        printf("%d ", t[i]);
    }

    printf("\n");
}

// Toujours la même graine pour qu'on puisse avoir des résultats reproductibles

int main(int argc, char **argv) {
    float millis;
    size_t nBytes;
    int *h_arr = NULL;
    struct timeb tav, tap;
    double te;
    long somCpu;
    size_t taille = argc < 2 ? 1000000 : strtol(argv[1], NULL, 10);

    nBytes = sizeof(int) * taille;
    h_arr = (int*)malloc(nBytes);

    if(!h_arr) {
        fprintf(stderr, "Erreur d'allocation mémoire host\n");
        exit(1);
    }


    srand(1234);
    fillRandomly(h_arr, taille);
    if(taille < 100) printArr(h_arr, taille);

    ftime(&tav);
    somCpu = somme(h_arr, taille);
    ftime(&tap);
    te = (double)((tap.time*1000+tap.millitm)-(tav.time*1000+tav.millitm))/1000;

    printf("%ld éléments, %.3fMo\n", taille, taille / 512.0 / 1024.0 * sizeof(int));
    printf("SequentielCPU: %ld, %.3lfs\n", somCpu, te);


    // somme_hypercube() change le tableau, on remet comme avant
    srand(1234);
    fillRandomly(h_arr, taille);

    millis = hypercube(h_arr, taille);
    if(taille < 100) printArr(h_arr, taille);
    printf("HypercubeCUDA: %d, %.3fs\n", h_arr[0], millis / 1000.0f);

    free(h_arr);
    return 0;
}
