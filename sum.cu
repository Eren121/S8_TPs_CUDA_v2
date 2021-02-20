#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>

// 1 pour optimiser en utilisant une mémoire partagée
#define SHARED 1

#define pow2(x) (1<<(x))
#define N 300000000

#define NTESTS 1000

// Nombre de threads par bloc
#define NBTHREADS_MAX 256

#define check(error) { checkCudaError((error), __FILE__, __LINE__); }
void checkCudaError(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA: %s:%d %s\n", file, line, cudaGetErrorString(code));
        exit(EXIT_FAILURE);
    }
}

// Kernel de la somme hypercube
// Ne fonctionne que pour un seul bloc
__global__
void kernel_hypercube(int *t, int taille) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int nbits = (int)ceil(log2((double)taille));
    int d, mask, in, out;

    for(d = 1; d <= nbits; ++d) {
        if (x < pow2(nbits - d)) {
            mask = x << d;
            in = mask | pow2(d - 1);
            out = mask;

            if (in < taille) {
                t[out] += t[in];
            }
        }

        // On doit synchroniser mêmes les threads en dehors sinon deadlock
        __syncthreads();
    }
}

// Kernel pour une dimension
__global__
void kernel_hypercube_uneDim(int *t, int taille, int d, int nbits) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    int mask, in, out;

    // On copie de XXX1000 vers XXX0000
    // Les XXX correspondent à la valeur de x
    // suivi de 1 pour l'entrée et 0 pour la sortie
    // suivi d'uniquement des 0 car la réduction a été faite sur les 0

    if (x < pow2(nbits - d)) {
        mask = x << d;
        in = mask | pow2(d - 1);
        out = mask;

        if (in < taille) {
            t[out] += t[in];
        }
    }
}

// Algorithme de l'hypercube pour la somme
// Version séquentielle pour vérifier que l'algorithme est fonctionnel
int somme_hypercube(int *t, int taille) {
    const int nbits = (int)ceil(log2((double)taille));
    int d, x, mask, in, out, tot = 0;

    for(d = 1; d <= nbits; ++d) {
        for(x = 0; x < pow2(nbits - d); ++x) {
            mask = x << d;
            in = mask | pow2(d - 1);
            out = mask;

            if(in < taille) {
                t[out] += t[in];
            }
        }
    }

    tot = t[0];
    return tot;
}

// Somme séquentielle
int somme(int *arr, int taille) {
    int i, tot = 0;
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
    cudaEvent_t start, stop;
    float millis;
    int nBytes = sizeof(int) * N;
    int *h_arr = (int*)malloc(nBytes);
    int *d_arr = NULL;
    int nbBlocks;
    int nbits = (int)ceil(log2((double)N)), d;
    struct timeb tav, tap;
    double te;
    int somCpu;
    int dim = N;

    if (argc >= 2) dim = strtol(argv[1], NULL, 10);

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaMalloc(&d_arr, nBytes));

    srand(1234);
    fillRandomly(h_arr, N);

    if(dim < 100) printArr(h_arr, dim);


    ftime(&tav);
    somCpu = somme(h_arr, dim);
    ftime(&tap);
    te = (double)((tap.time*1000+tap.millitm)-(tav.time*1000+tav.millitm))/1000 ;

    printf("%d éléments, %.3fMo\n", dim, dim / 512.0 / 1024.0 * sizeof(int));
    printf("SequentielCPU: %d, %.3lfs\n", somCpu, te);


    // somme_hypercube() change le tableau, on remet comme avant
    srand(1234);
    fillRandomly(h_arr, dim);


    check(cudaMemcpy(d_arr, h_arr, nBytes, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    nbBlocks = (dim - 1) / NBTHREADS_MAX + 1;
    if(nbBlocks == 1) {
        // S'il n'y a qu'un seul bloc, pas de problème de synchronisation
        kernel_hypercube<<<nbBlocks, NBTHREADS_MAX>>>(d_arr, dim);
    }
    else {
        // Sinon on doit séparer dimension par dimension
        // Car on ne peut pas synchroniser des threads de blocs différents sur le device
        for(d = 1; d <= nbits; ++d) {
            kernel_hypercube_uneDim<<<nbBlocks, NBTHREADS_MAX>>>(d_arr, dim, d, nbits);
            check(cudaDeviceSynchronize());
        }
    }

    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&millis, start, stop));

    check(cudaMemcpy(h_arr, d_arr, nBytes, cudaMemcpyDeviceToHost));

    if(dim < 100) printArr(h_arr, dim);
    printf("HypercubeCUDA: %d, %.3fs\n", h_arr[0], millis / 1000.0f);

    check(cudaFree(d_arr));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    free(h_arr);
    return 0;
}
