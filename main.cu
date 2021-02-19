#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define pow2(x) (1<<(x))

// Algorithme de l'hypercube pour la somme
// Version séquentielle pour vérifier que l'algorithme est fonctionnel
int somme_hypercube(int *t, int taille) {
    const int nbits = (int)ceil(log2(taille));
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
        t[i] = rand() % 10;
    }
}

void printArr(int *t, int taille) {
    int i;
    for(i = 0; i < taille; ++i) {
        printf("%d ", t[i]);
    }

    printf("\n");
}

#define N 10000000

int main() {
    int *arr = (int*)malloc(sizeof(*arr) * N);

    srand(time(NULL));
    fillRandomly(arr, N);

    if(N < 100) printArr(arr, N);

    printf("Sequentiel: %d\n", somme(arr, N));
    printf("Hypercube: %d\n", somme_hypercube(arr, N));

    free(arr);
    return 0;
}
