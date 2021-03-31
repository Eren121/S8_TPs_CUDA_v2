#ifndef ARRAY_2D_H
#define ARRAY_2D_H

#include <stddef.h>

#define STR_(x) #x
#define STR(x) STR_(x) /* Au cas où il y un appel de macro dans x */

#define ARRAY_2D_MAX_ELEMS_LINE 10

/**
 * Variable temporaire que l'on peut utiliser dans les fonctions pour l'affichage.
 * L'utilisation qu'on n'en fait ne doit pas dépasser la taille donnée.
 */
#define ARRAY_2D_TMP_STRING_SIZE 1000
extern char array2DTmpString[ARRAY_2D_TMP_STRING_SIZE];

typedef struct {
    size_t elemSize;
    size_t nRows;
    size_t nCols;

    /*
     * Ordre de stockage des éléments dans la mémoire: "row-major", comme en C:
     *
     * data = [0, 1, 2,
     *         3, 4, 5,
     *         6, 7, 8];
     *
     * array2DAt(row=2, col=0) = 6;
     */
    unsigned char *data;

    /**
     * Si data pointe vers des données dans le device, 
     * hostData contient le pointeur vers les données de l'hôte
     */
    unsigned char *hostData;

} Array2D;

/**
 * Initialise un tableau 2D.
 */
void array2DInit(Array2D *tab, size_t elemSize, size_t nRows, size_t nCols);

/**
 * Accède à un élément d'un tableau 2D.
 * Retourne un pointeur donc utilisable en lecture et en écriture.
 * 
 * Retourne un pointeur vers l'élément de taille elemSize,
 * void* donc doit être casté pour être utilisé.
 */
__host__ __device__
void* array2DAt(Array2D *tab, size_t row, size_t col);

/**
 * Version Constante de array2DAt()
 */
__host__ __device__
const void* array2DGet(const Array2D *tab, size_t row, size_t col);

/**
 * Affiche un tableau 2D.
 * Pour que l'affichage soit générique, on demande une fonction
 * qui se chargera de convertir chaque élément en chaîne de caractère.
 * Attention, la chaîne n'est pas désallouée. On pourra utiliser
 * une chaîne statique d'une taille suffisante pour tous les éléments possibles pour le type donné.
 * Si la fonction elemToString retourne toujours une chaîne de la même taille,
 * l'affichage sera toujours aligné.

 * Format d'affichage (ressemble à numpy en Python):
 *
 * [[0 1 2],
 *  [3 4 5],
 *  [6 7 8]]
 *
 */
void array2DPrint(const Array2D *tab, const char* (*elemToString)(const void *elem));

/**
 * Retourne la taille totale en bytes (nombre d'éléments * taille d'un élément)
 * prise par les données tab->data
 */
size_t array2DSizeof(const Array2D *tab);

/**
 * Remplit avec une valeur depuis une fonction (une valeur aléatoire par exemple)
 */
void array2DGenerate(Array2D *tab, void* (*generator)(size_t row, size_t col));

/**
 * Libère toutes les ressources allouées par un tableau 2D.
 */
void array2DDestroy(Array2D *tab);

/**
 * Fonctions d'affichage pour certains types communs primitifs
 */
const char* array2DIntToString(const void *intPointer);
const char* array2DFloatToString(const void *floatPointer);

/**
 * Préparer la tableau a un envoi par valeur à CUDA
 * C'est-à-dire remplacer le pointeur des données hôte par un pointeur device
 * 
 * @param load Si on doit copier les données depuis l'hôte vers le device (si donnée d'entrée)
 */
void array2DCudaSetup(Array2D *tab, bool load);

/**
 * Finaliser le tableau après un appel à un kernel cuda.
 * C'est-à-dire remettre le pointeur hôte.
 * 
 * @param save Si on doit récupérer le résultat depuis le device vers l'hôte (si donnée de sortie)
 */
void array2DCudaFinalize(Array2D *tab, bool save);


/**
 * Comparer voir si le résultat est le mêm eavec CUDA et sur GPU
 */
void array2DCompare_float(Array2D *gpu, Array2D *cpu);

#endif