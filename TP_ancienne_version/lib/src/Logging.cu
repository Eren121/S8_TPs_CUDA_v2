#include "Logging.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

void fatal(const char *format, ...)
{
    va_list args;
    
    fprintf(stderr, "Erreur fatale\n");

    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    putc('\n', stderr);
    exit(1);
}

__host__
void cudaCheck(cudaError_t code) {
    if(code != cudaSuccess) {
        fatal("Cuda error: %s.\n", cudaGetErrorString(code));
    }
}