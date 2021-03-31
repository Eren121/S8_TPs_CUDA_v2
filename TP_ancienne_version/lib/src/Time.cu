#include "Time.h"
#include <stdio.h>

void elapsedFrom(Elapsed *t) {
    gettimeofday(&t->t1, NULL);
}

void elapsedTo(Elapsed *t) {
    gettimeofday(&t->t2, NULL);
}

double elapsedSeconds(Elapsed *t) {
    double te = (t->t2.tv_sec - t->t1.tv_sec) * 1000.0;      /* sec -> ms */

    te += (t->t2.tv_usec - t->t1.tv_usec) / 1000.0;   /* us -> ms */
    te /= 1000.0; /* ms -> s */

    return te;
}

void elapsedPrint(Elapsed *t) {
    double te = elapsedSeconds(t);

	printf("Temps d'execution : %fs\n",  te);
}