
/*********************************************************/

/* TAUCS                                                 */

/* Author: Sivan Toledo                                  */

/*********************************************************/

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

#ifdef TAUCS_DOUBLE_IN_BUILD
	if (A->flags & TAUCS_DOUBLE)
		return taucs_dccs_permute_symmetrically(A, perm, invperm);
#endif

#ifdef TAUCS_SINGLE_IN_BUILD
	if (A->flags & TAUCS_SINGLE)
		return taucs_sccs_permute_symmetrically(A, perm, invperm);
#endif

#ifdef TAUCS_DCOMPLEX_IN_BUILD
	if (A->flags & TAUCS_DCOMPLEX)
		return taucs_zccs_permute_symmetrically(A, perm, invperm);
#endif

#ifdef TAUCS_SCOMPLEX_IN_BUILD
	if (A->flags & TAUCS_SCOMPLEX)
		return taucs_cccs_permute_symmetrically(A, perm, invperm);
#endif

	assert(0);
	return NULL;
}
#endif							       /* TAUCS_CORE_GENERAL */

#ifndef TAUCS_CORE_GENERAL

taucs_ccs_matrix *taucs_dtl(ccs_permute_symmetrically) (taucs_ccs_matrix * A, int *perm, int *invperm) {
	taucs_ccs_matrix *PAPT;
	int n;
	int nnz;

	/*
	 * int* colptr;
	 */
	int *len;
	int i, j, ip, I, J;
	taucs_datatype AIJ;

	assert(A->flags & TAUCS_SYMMETRIC || A->flags & TAUCS_HERMITIAN);
	assert(A->flags & TAUCS_LOWER);

	n = A->n;
	nnz = (A->colptr)[n];

	PAPT = taucs_dtl(ccs_create) (n, n, nnz);
	if (!PAPT)
		return NULL;

	/*
	 * PAPT->flags = TAUCS_SYMMETRIC | TAUCS_LOWER;
	 */
	PAPT->flags = A->flags;

	len = (int *) taucs_malloc(n * sizeof(int));
	/*
	 * colptr = (int*) taucs_malloc(n * sizeof(int));
	 */
	if (!len) {
		taucs_printf("taucs_ccs_permute_symmetrically: out of memory\n");
		taucs_ccs_free(PAPT);
		return NULL;
	}

	for (j = 0; j < n; j++)
		len[j] = 0;

	for (j = 0; j < n; j++) {
		for (ip = (A->colptr)[j]; ip < (A->colptr)[j + 1]; ip++) {
			/*
			 * i = (A->rowind)[ip] - (A->indshift);
			 */
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
			/*
			 * i = (A->rowind)[ip] - (A->indshift);
			 */
			i = (A->rowind)[ip];
			AIJ = (A->taucs_values)[ip];

			I = invperm[i];
			J = invperm[j];

			if (I < J) {
				int T = I;

				I = J;
				J = T;
				if (A->flags & TAUCS_HERMITIAN)
					AIJ = taucs_conj(AIJ);
			}

			/*
			 * (PAPT->rowind)[ len[J] ] = I + (PAPT->indshift);
			 */
			(PAPT->rowind)[len[J]] = I;
			(PAPT->taucs_values)[len[J]] = AIJ;

			len[J]++;
		}
	}

	taucs_free(len);
	return PAPT;
}

#endif							       /* #ifndef TAUCS_CORE_GENERAL */

/*********************************************************/
