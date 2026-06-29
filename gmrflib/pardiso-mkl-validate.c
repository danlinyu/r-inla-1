/*
 * pardiso-mkl-validate.c -- standalone validation of the oneMKL PARDISO adapter
 * (inlaprog/src/pardiso-mkl.c) against dense LAPACK ground truth.
 *
 * Builds a small SPD GMRF precision Q on a banded (linear) graph, runs the GMRFLib
 * PARDISO code path (which, with the adapter linked, drives Intel oneMKL), and
 * compares: log|det Q|, the full solve Q^{-1}b, the L and L^T solves, and the
 * selected inverse Qinv -- the last reported with DIAGONAL and OFF-DIAGONAL error
 * separately, which tells us whether oneMKL's phase=-22 gives off-diagonal Qinv.
 *
 * Exploratory tool on the mkl-pardiso branch; not part of the upstream PR.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GMRFLib/GMRFLib.h"

/* dpotrf_/dpotrs_/dpotri_ are declared by GMRFLib's lapack-interface.h (with the
 * trailing hidden Fortran string-length arg). Only dtrsv_ needs declaring here. */
extern void dtrsv_(const char *, const char *, const char *, int *, double *, int *, double *, int *,
		   FORTRAN_CHARLEN_T, FORTRAN_CHARLEN_T, FORTRAN_CHARLEN_T);

/* inlaprog symbols that libGMRFLib references; stubbed since we link only GMRFLib. */
int inla_ncpu(void)
{
	return 1;
}
void *inla_stiles_get_setup(void *m)
{
	(void) m;
	return NULL;
}

static double q_diag(int i)
{
	return 5.0 + 0.1 * i;				       /* diagonally dominant -> SPD */
}

static double Qfunc(int UNUSED(thread_id), int i, int j, double *UNUSED(values), void *UNUSED(arg))
{
	if (j < 0) {
		return NAN;
	}
	if (i == j) {
		return q_diag(i);
	}
	return -1.0;					       /* neighbour entries */
}

static double maxabs_diff(double *a, double *b, int n)
{
	double m = 0.0;
	for (int i = 0; i < n; i++) {
		double d = fabs(a[i] - b[i]);
		if (d > m) {
			m = d;
		}
	}
	return m;
}

int main(void)
{
	int n = 12, bw = 2;
	int info = 0, one = 1;

	setvbuf(stdout, NULL, _IONBF, 0);		       /* unbuffered: see output up to any crash */

	GMRFLib_openmp = Calloc(1, GMRFLib_openmp_tp);	       /* the engine allocates this at startup */
	GMRFLib_openmp->strategy = GMRFLib_OPENMP_STRATEGY_PARDISO;
	GMRFLib_openmp_implement_strategy(GMRFLib_OPENMP_PLACES_OPTIMIZE, NULL, NULL);
	GMRFLib_smtp = GMRFLib_SMTP_PARDISO;

	GMRFLib_graph_tp *g = NULL;
	GMRFLib_graph_mk_linear(&g, n, bw, 0);

	/* dense Q (column-major) for the reference */
	double *Q = Calloc(n * n, double);
	for (int i = 0; i < n; i++) {
		Q[i + i * n] = Qfunc(0, i, i, NULL, NULL);
		for (int k = 0; k < g->nnbs[i]; k++) {
			int j = g->nbs[i][k];
			Q[i + j * n] = Qfunc(0, i, j, NULL, NULL);
		}
	}

	/* ---- reference: dense Cholesky, logdet, inverse ---- */
	double *L = Calloc(n * n, double);
	for (int i = 0; i < n * n; i++) {
		L[i] = Q[i];
	}
	dpotrf_("L", &n, L, &n, &info, 1);			       /* L (lower) : Q = L L^T */
	double ref_logdet = 0.0;
	for (int i = 0; i < n; i++) {
		ref_logdet += 2.0 * log(L[i + i * n]);
	}

	double *Qi_ref = Calloc(n * n, double);
	for (int i = 0; i < n * n; i++) {
		Qi_ref[i] = Q[i];
	}
	dpotrf_("L", &n, Qi_ref, &n, &info, 1);
	dpotri_("L", &n, Qi_ref, &n, &info, 1);		       /* lower triangle = Q^{-1} */
	for (int i = 0; i < n; i++) {			       /* symmetrise */
		for (int j = 0; j < i; j++) {
			Qi_ref[j + i * n] = Qi_ref[i + j * n];
		}
	}

	/* random-ish rhs */
	double *b = Calloc(n, double);
	for (int i = 0; i < n; i++) {
		b[i] = 1.0 + sin(0.7 * i);
	}

	/* ---- GMRFLib / oneMKL path ---- */
	GMRFLib_pardiso_store_tp *store = NULL;
	GMRFLib_pardiso_init(&store);
	GMRFLib_pardiso_reorder(store, g);
	GMRFLib_pardiso_build(0, store, g, Qfunc, NULL);
	GMRFLib_pardiso_chol(store);

	double mkl_logdet = GMRFLib_pardiso_logdet(store);
	printf("[logdet]    mkl=%.10f  ref=%.10f  diff=%.3e\n", mkl_logdet, ref_logdet, fabs(mkl_logdet - ref_logdet));

	/* full solve Q x = b */
	double *x = Calloc(n, double);
	double *xref = Calloc(n, double);
	for (int i = 0; i < n; i++) {
		x[i] = b[i];
		xref[i] = b[i];
	}
	GMRFLib_pardiso_solve_LLT(store, x, x, 1);
	dpotrs_("L", &n, &one, L, &n, xref, &n, &info, 1);
	printf("[solve_LLT] max|x - Q^{-1}b| = %.3e\n", maxabs_diff(x, xref, n));

	/* L solve : L y = b */
	double *yL = Calloc(n, double);
	double *yLref = Calloc(n, double);
	for (int i = 0; i < n; i++) {
		yL[i] = b[i];
		yLref[i] = b[i];
	}
	GMRFLib_pardiso_solve_L(store, yL, yL, 1);
	dtrsv_("L", "N", "N", &n, L, &n, yLref, &one, 1, 1, 1);
	printf("[solve_L]   max|y - L^{-1}b|   = %.3e\n", maxabs_diff(yL, yLref, n));

	/* L^T solve : L^T y = b */
	double *yT = Calloc(n, double);
	double *yTref = Calloc(n, double);
	for (int i = 0; i < n; i++) {
		yT[i] = b[i];
		yTref[i] = b[i];
	}
	GMRFLib_pardiso_solve_LT(store, yT, yT, 1);
	dtrsv_("L", "T", "N", &n, L, &n, yTref, &one, 1, 1, 1);
	printf("[solve_LT]  max|y - L^{-T}b|   = %.3e\n", maxabs_diff(yT, yTref, n));

	/* selected inverse */
	GMRFLib_pardiso_Qinv(store);
	GMRFLib_csr_tp *Qinv = store->pstore[GMRFLib_PSTORE_TNUM_REF]->Qinv;
	double diag_err = 0.0, off_err = 0.0;
	int noff = 0;
	if (Qinv) {
		for (int i = 0; i < Qinv->s->n; i++) {
			for (int k = Qinv->s->ia[i]; k < Qinv->s->ia[i + 1]; k++) {
				int j = Qinv->s->ja[k];
				double got = Qinv->a[k];
				double ref = Qi_ref[i + j * n];
				double e = fabs(got - ref);
				if (i == j) {
					if (e > diag_err) {
						diag_err = e;
					}
				} else {
					noff++;
					if (e > off_err) {
						off_err = e;
					}
				}
			}
		}
		printf("[Qinv]      diag max err = %.3e   offdiag max err = %.3e   (n_off=%d)\n", diag_err, off_err, noff);
	} else {
		printf("[Qinv]      NULL\n");
	}

	GMRFLib_pardiso_free(&store);
	printf("VALIDATE DONE\n");
	return 0;
}
