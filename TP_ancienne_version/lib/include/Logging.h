#ifndef LOGGING_H
#define LOGGING_H


/**
 * Termine le programme subitement pour cause d'erreur,
 * avec un message d'erreur écrit sur la sortie d'erreur standard stderr.
 */
void fatal(const char *format, ...);

/**
 * Vérifier toutes les erreurs CUDA
 */
__host__
void cudaCheck(cudaError_t code);

/**
 * Erreur fatale à cause d'une allocation mémoire qui a échoué.
 */
#define fatalMalloc(variableName) do { \
        (void)((void)sizeof(variableName), "La variable n'existe pas"); \
        fatal("L'allocation mémoire a échouée sur la variable %s", #variableName); \
    } while(0)

#endif