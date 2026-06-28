/*********************************************************/
/* TAUCS                                                 */
/* Author: Sivan Toledo                                  */
/*********************************************************/

/*
 * taucs_ccs_permute_symmetrically
 *
 * This routine is part of the upstream TAUCS distribution but was dropped from
 * the reduced (double-only, non-templated) TAUCS subset bundled under
 * gmrflib/taucs/.  GMRFLib calls it unconditionally from problem-setup.c, so it
 * must be present.  Re-implemented here in the subset's plain-double style
 * (matrix uses int *colptr, int *rowind, double *values; no datatype templating,
 * no indshift, real symmetric only), matching the declaration in taucs_private.h.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "taucs.h"

#ifndef TAUCS_CORE
#error "You must define TAUCS_CORE to compile this file"
#endif

#ifdef TAUCS_CORE_GENERAL

taucs_ccs_matrix *taucs_ccs_permute_symmetrically(taucs_ccs_matrix * A, int *perm, int *invperm)
{
	taucs_ccs_matrix *PAPT;
	int n, nnz;
	int *len;
	int i, j, ip, I, J;
	double AIJ;

	(void) perm;					       /* only invperm is used */

	assert(A->flags & TAUCS_SYMMETRIC);
	assert(A->flags & TAUCS_LOWER);

	n = A->n;
	nnz = (A->colptr)[n];

	PAPT = taucs_ccs_create(n, n, nnz, A->flags);
	if (!PAPT)
		return NULL;
	PAPT->flags = A->flags;

	len = (int *) taucs_malloc(n * sizeof(int));
	if (!len) {
		fprintf(stderr, "taucs_ccs_permute_symmetrically: out of memory\n");
		taucs_ccs_free(PAPT);
		return NULL;
	}

	for (j = 0; j < n; j++)
		len[j] = 0;

	for (j = 0; j < n; j++) {
		for (ip = (A->colptr)[j]; ip < (A->colptr)[j + 1]; ip++) {
			i = (A->rowind)[ip];
			I = invperm[i];
			J = invperm[j];
			if (I < J) {
				int T = I;
				I = J;
				J = T;
			}
			len[J]++;
		}
	}

	(PAPT->colptr)[0] = 0;
	for (j = 1; j <= n; j++)
		(PAPT->colptr)[j] = (PAPT->colptr)[j - 1] + len[j - 1];

	for (j = 0; j < n; j++)
		len[j] = (PAPT->colptr)[j];

	for (j = 0; j < n; j++) {
		for (ip = (A->colptr)[j]; ip < (A->colptr)[j + 1]; ip++) {
			i = (A->rowind)[ip];
			AIJ = (A->values)[ip];
			I = invperm[i];
			J = invperm[j];
			if (I < J) {
				int T = I;
				I = J;
				J = T;
			}
			(PAPT->rowind)[len[J]] = I;
			(PAPT->values)[len[J]] = AIJ;
			len[J]++;
		}
	}

	taucs_free(len);
	return PAPT;
}

#endif							       /* TAUCS_CORE_GENERAL */
