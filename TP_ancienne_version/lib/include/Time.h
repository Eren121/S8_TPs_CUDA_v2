#ifndef TIME_H
#define TIME_H

#include <stdbool.h> /* bool, true, false */
#include <stddef.h> /* size_t */
#include <sys/time.h>  /* struct timeval */

///
/// Gestion du temps
///

typedef struct {
	struct timeval t1, t2;
} Elapsed;


void elapsedFrom(Elapsed *t);
void elapsedTo(Elapsed *t);

/**
 * Récupère le temps écoulé en secondes entre les deux appels à elapsedFrom() et elapsedTo()
 */
double elapsedSeconds(Elapsed *t);

/**
 * Affiche le temps écoulé entre les deux appels à elapsed_from() et elapsed_to()
 */
void elapsedPrint(Elapsed *t);

#endif