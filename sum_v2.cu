#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/timeb.h>


// Hypercube
// Version: pas de mémoire partagée
// On réduit tout sur une dimension à chaque appel (non optimisé pour une mémoire partagée par block)
// Pas de distinction entre des threads du même block
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

// Note: l'utilisation de mémoire partagée ne paraît pas optimale ici
// nvprof ne montre pas d'amélioration significative (voir des baisses de performance)
// Kernel de la somme hypercube
// Ne fonctionne que pour un seul bloc
__global__
void kernel_hypercube(int *t, int taille) {
    __shared__ int s[NBTHREADS_MAX];

    const int x = threadIdx.x;
    const int nbits = (int)ceil(log2((double)taille));
    int d, mask, in, out;

    if(x < taille) {
        s[x] = t[x];
    }
    __syncthreads();

    for(d = 1; d <= nbits; ++d) {
        if (x < pow2(nbits - d)) {
            mask = x << d;
            in = mask | pow2(d - 1);
            out = mask;

            if (in < taille) {
                s[out] += s[in];
            }
        }

        // NE Fonctionne que sur 1024 à cause de __syncthreads():
        // Syncthreads ne marche que pour les threads du même block (donc <= 1024 threads)
        // On doit synchroniser mêmes les threads en dehors sinon deadlock
        __syncthreads();
    }

    if(x < taille) {
        t[x] = s[x];
    }
}

// Kernel pour une dimension
// Réduit 1 dimension de l'hypercube
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
    int nBytes;
    int *h_arr = NULL;
    int *d_arr = NULL;
    int nbBlocks;
    int nbits, d;
    struct timeb tav, tap;
    double te;
    int somCpu;
    const int taille = argc < 2 ? 1000000 : strtol(argv[1], NULL, 10);

    nBytes = sizeof(int) * taille;
    nbits = (int)ceil(log2((double)taille));
    h_arr = (int*)malloc(nBytes);

    if(!h_arr) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        exit(1);
    }

    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));

    check(cudaMalloc(&d_arr, nBytes));

    srand(1234);
    fillRandomly(h_arr, taille);

    if(taille < 100) printArr(h_arr, taille);


    ftime(&tav);
    somCpu = somme(h_arr, taille);
    ftime(&tap);
    te = (double)((tap.time*1000+tap.millitm)-(tav.time*1000+tav.millitm))/1000 ;

    printf("%d éléments, %.3fMo\n", taille, taille / 512.0 / 1024.0 * sizeof(int));
    printf("SequentielCPU: %d, %.3lfs\n", somCpu, te);


    // somme_hypercube() change le tableau, on remet comme avant
    srand(1234);
    fillRandomly(h_arr, taille);


    check(cudaMemcpy(d_arr, h_arr, nBytes, cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    nbBlocks = (taille - 1) / NBTHREADS_MAX + 1;
    if(nbBlocks == 1) {
        // S'il n'y a qu'un seul bloc, pas de problème de synchronisation
        kernel_hypercube<<<nbBlocks, NBTHREADS_MAX>>>(d_arr, taille);
    }
    else {
        // Sinon on doit séparer dimension par dimension
        // Car on ne peut pas synchroniser des threads de blocs différents sur le device
        for(d = 1; d <= nbits; ++d) {
            kernel_hypercube_uneDim<<<nbBlocks, NBTHREADS_MAX>>>(d_arr, taille, d, nbits);
            check(cudaDeviceSynchronize());
        }
    }

    check(cudaEventRecord(stop));
    check(cudaEventSynchronize(stop));
    check(cudaEventElapsedTime(&millis, start, stop));

    check(cudaMemcpy(h_arr, d_arr, nBytes, cudaMemcpyDeviceToHost));

    if(taille < 100) printArr(h_arr, taille);
    printf("HypercubeCUDA: %d, %.3fs\n", h_arr[0], millis / 1000.0f);

    check(cudaFree(d_arr));
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    free(h_arr);
    return 0;
}
