
#include <stdio.h>
#include <sys/timeb.h>
#include <math.h>
#include <cuda.h>

// nbre de threads dans une dimension (on travaille en 1D)
#define NBTHREADS 1024

// Pour la generation aleatoire des valeurs
#define MAX_VAL 10
#define MIN_VAL 0

void vecAleatoire(int *v, int n);
void vecAff(int *v, int n);

// voisin dans la dimension d du processeur p
unsigned int voisin(unsigned int p, unsigned int d);

// calcul de la somme d'un vecteur sur CPU
// pour les tests
int somCPU(int *t, int n);

// calcul de la somme sur l'hypercube (ne fonctionne que pour un vecteur comportant moins de NBTHREADS elements !!!)
__global__ void somHypercubeKernel(int* d_t, int d, int total);
// calcul de la somme sur l'hypercube dans la dimension dc (suppose que toutes les dimensions inferieures ont ete calculees)
__global__ void somHypercubeUneDimensionKernel(int* d_t, int d, int dc, int total);
// fonction qui appel les noyaux
float somHypercube(int* h_t, int d);

int GPUInfo();

int main(int argc, char* argv[]){
    int dim=3, n;
    int *tab;
    int somme;
    float ms;
    // pour mesurer le temps sur CPU
    struct timeb tav, tap ;
    double te;
    // recuperation des parametres en ligne de commandes
    if (argc==2) dim= strtol(argv[1], NULL, 10);
    // allocation et initialisation du tableau
    // pour calculer le max de n=2^dim valeurs, nous avons besoin d'un vecteur de taille 2*n
    n=(int)pow(2,dim);
    // l'occupation memoire du vecteur en Mo
    float tailleMo=sizeof(int)*n/float(512*1024);
    tab=(int*)malloc(sizeof(int)*2*n);
    vecAleatoire(tab,2*n);
    // la partie droite du tableau est egale a 0
    for(int i=0;i<n;i++) tab[n+i]=0;
    // quel GPU ?
    GPUInfo();
    // calcul de la somme sur CPU
    ftime(&tav);
    somme=somCPU(tab,n);
    ftime(&tap);
    te = (double)((tap.time*1000+tap.millitm)-(tav.time*1000+tav.millitm))/1000 ;
    // affichage du tableau
    /* vecAff(tab,2*n);
    printf("\n"); */
    // calcul de la somme sur GPU
    printf("----\nHypercube de dimension %d, soit %d valeurs dans le vecteur (%f Mo).\n", dim, n, tailleMo);
    printf("Temps d'execution sur CPU : %f ms.\n",te);
    ms=somHypercube(tab,dim);
    printf("Temps d'execution sur GPU : %f ms.\n",ms);
    // le resultat peut etre a gauche ou a droite, en fonction de la parite de dim
    printf("SommeCPU : %d, sommeGPU : %d (ecart GPU : %d)\n",somme, tab[n*(dim%2)], tab[n*(dim%2)+n-1]-tab[n*(dim%2)]);
    // affichage du tableau resultat
    // vecAff(tab,2*n);
}

// calcul la somme de tous les elements contenus dans le vecteur h_t
float somHypercube(int* h_t, int d){
    int n = (int)pow(2,d);
    // la taille du vecteur est de 2*n elements
    long size = 2*n*sizeof(int);
    int nbBlocs;
    int *d_t;
    // pour mesurer le temps en cuda
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocations du vecteur
    printf("Allocation de %ld octets (%f Mo) sur le GPU.\n",size,(float)size/1024/1024);
    if (cudaMalloc((void **) &d_t, size)!=cudaSuccess) {
        printf ("Pb allocation !!!\n");
        exit(1);
    }
    // copies hote vers device
    cudaMemcpy(d_t, h_t, size, cudaMemcpyHostToDevice);

    // le calcul sur GPU
    nbBlocs=(n-1)/NBTHREADS+1;
    printf("Appel du noyau <<<%d blocs, %d>>>.\n", nbBlocs, NBTHREADS);
    // 2 cas de figures : (1) un seul bloc ou (2) plus d'un blocs
    cudaEventRecord(start);
    if(nbBlocs==1) somHypercubeKernel<<<nbBlocs,NBTHREADS>>>(d_t, d, n);
    else {
        // on appelle le noyau pour chaque dimension
        // afin de s'assurer que tous les blocs soient resolus avant de passer a la dimension suivante
        for(int i=0;i<d;i++) {
            // printf("somHypercubeUneDimensionKernel<<<%d,%d>>>(d_t,%d,%d,%d)\n",nbBlocs,NBTHREADS,d,i,n);
            somHypercubeUneDimensionKernel<<<nbBlocs,NBTHREADS>>>(d_t, d, i, n);
            // attente de la fin du noyau dans la dimension courante
            cudaDeviceSynchronize();
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // copie device vers hote
    cudaMemcpy(h_t, d_t, size, cudaMemcpyDeviceToHost);
    // liberation de la memoire
    cudaFree(d_t);
    return milliseconds;
}

// calcul de la somme sur l'hypercube (ne fonctionne que pour un vecteur comportant moins de NBTHREADS elements !!!)
__global__ void somHypercubeKernel(int* d_t, int d, int total){
    int p,voisin;
    int val;
    p=threadIdx.x+blockDim.x*blockIdx.x;
    // attention pour eviter le conflit d'ecriture, la longueur de d_t est egale a 2*total
    // suivant la parite de i, on utilise la partie gauche ou droite du tableau pour la lecture et inversement pour l'ecriture
    if (p<total) {
        for(int i=0;i<d;i++){
            voisin=p^(((unsigned int)1)<<i);
            val=d_t[total*(i%2)+voisin];
            d_t[total*((i+1)%2)+p]=d_t[total*(i%2)+p]+val;
            __syncthreads();
        }
    }
}

// calcul de la somme sur l'hypercube dans la dimension dc (suppose que toutes les dimensions inferieures ont ete calculees)
// ce noyau devrait etre appele d fois depuis l'hote !!!
__global__ void somHypercubeUneDimensionKernel(int* d_t, int d, int i, int total) {
    int p,voisin;
    int val;
    p=threadIdx.x+blockDim.x*blockIdx.x;
    // attention pour eviter le conflit d'ecriture, la longueur de d_t est egale a 2*total
    // suivant la parite de i, on utilise la partie gauche ou droite du tableau pour la lecture et inversement pour l'ecriture
    if (p<total) {
        // voisin=p^(((unsigned int)1)<<i);
        voisin=p^(1<<i);
        val=d_t[total*(i%2)+voisin];
        d_t[total*((i+1)%2)+p]=d_t[total*(i%2)+p]+val;
        __syncthreads();
    }
}


// rappel sur les fonctions C pour manipuler les bits directement
// https://zestedesavoir.com/tutoriels/755/le-langage-c-1/notions-avancees/manipulation-des-bits/
unsigned int voisin(unsigned int p, unsigned int d) {
    // ou exclusif entre le numero du processeur
    // et le bit a 1 en position d
    return (p^(((unsigned int)1)<<d));
}

// calcul de la somme d'un vecteur sur CPU
int somCPU(int *t, int n) {
    int somme=0;
    for(int i=0;i<n;i++) somme+=t[i];
    return somme;
}

// Initialisation aleatoire d'un vecteur
void vecAleatoire(int *v, int n) {
    int i;
    for(i=0;i<n;i++){
        v[i]= (int)((double)rand()/RAND_MAX*MAX_VAL) + MIN_VAL;
    }
}

// Affiche un vecteur
void vecAff(int *v, int n){
    int i;
    printf("[");
    for(i=0;i<n-1;i++) printf("%d ",v[i]);
    printf("%f]",v[n-1]);
}

int GPUInfo(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                printf("No CUDA GPU has been detected");
                return -1;
            } else if (deviceCount == 1) {
                printf("There is 1 device supporting CUDA\n");
                printf("Device %d, name: %s\n",  dev, deviceProp.name);
                printf("Computational Capabilities: %d.%d\n", deviceProp.major, deviceProp.minor);
                printf("Maximum global memory size: %ld bytes\n", deviceProp.totalGlobalMem);
                printf("Maximum shared memory size per block: %ld bytes\n", deviceProp.sharedMemPerBlock);
                printf("Warp size: %d\n", deviceProp.warpSize);
                printf("Maximum number of blocks per multiProcessor: %d\n",deviceProp.maxBlocksPerMultiProcessor);
                printf("Maximum number of threads per multiProcessor: %d\n",deviceProp.maxThreadsPerMultiProcessor);
                printf("Maximum grid size : %d x %d x %d blocks.\n",deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
            } else {
                printf("There are %d devices supporting CUDA\n", deviceCount);
            }
        }
    }
    return deviceCount;
}