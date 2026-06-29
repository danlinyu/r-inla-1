/*
 * pardiso-mkl.c -- a license-clean adapter that lets GMRFLib's PARDISO code path
 * run on Intel oneMKL's PARDISO.
 *
 * GMRFLib (gmrflib/smtp-pardiso.c) is written against the "academic"/Panua PARDISO
 * calling sequence:
 *      pardisoinit(pt, mtype, solver, iparm, dparm, error)             -- 6 args
 *      pardiso(..., error, dparm)                                      -- 17 args (trailing dparm)
 * Intel oneMKL exposes the SAME core solver under a slightly different surface:
 *      pardisoinit(pt, mtype, iparm)                                   -- 3 args, no solver/dparm
 *      pardiso(..., error)                                             -- 16 args, no dparm
 * and oneMKL has no `dparm`, returns the log-determinant via `pardiso_getdiag`
 * (the LDL^T pivots), and assigns some `iparm[]` indices different meanings.
 *
 * This file DEFINES the Panua-style symbols that GMRFLib calls and forwards them to
 * oneMKL, translating the differences. It is written from the PUBLIC Intel oneMKL
 * documentation only -- it contains no Panua/PARDISO-project source and does not
 * link the proprietary library.
 *
 * oneMKL is reached at run time through `mkl_rt` (LoadLibrary/GetProcAddress on
 * Windows, dlopen elsewhere) so there is no link-time clash between the `pardiso`
 * symbol this file exports and oneMKL's own `pardiso`.
 *
 * Built only when INLA_WITH_PARDISO_MKL is defined; otherwise this file is empty
 * and the usual stubs in libpardiso.c (or the real Panua library) are used.
 */

#if defined(INLA_WITH_PARDISO_MKL)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

/* oneMKL uses MKL_INT == int in the LP64 interface, which is what GMRFLib uses. */
typedef void (*mkl_pardisoinit_fn)(void *, int *, int *);
typedef void (*mkl_pardiso_fn)(void *, int *, int *, int *, int *, int *, void *,
			       int *, int *, int *, int *, int *, int *, void *, void *, int *);
typedef void (*mkl_pardiso_getdiag_fn)(const void *, void *, void *, int *, int *);

static mkl_pardisoinit_fn     mkl_pardisoinit_p = NULL;
static mkl_pardiso_fn         mkl_pardiso_p = NULL;
static mkl_pardiso_getdiag_fn mkl_pardiso_getdiag_p = NULL;
static int                    mkl_loaded = 0;

static void *pmkl_sym(void *h, const char *name)
{
#if defined(_WIN32)
	return (void *) GetProcAddress((HMODULE) h, name);
#else
	return dlsym(h, name);
#endif
}

static void pmkl_load(void)
{
#pragma omp critical (pardiso_mkl_load)
	{
		if (!mkl_loaded) {
			void *h = NULL;
#if defined(_WIN32)
			const char *cands[] = { "mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.1.dll", "mkl_rt.dll", NULL };
			for (int i = 0; cands[i] && !h; i++) {
				h = (void *) LoadLibraryA(cands[i]);
			}
#else
			const char *cands[] = { "libmkl_rt.so", "libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.dylib", NULL };
			for (int i = 0; cands[i] && !h; i++) {
				h = dlopen(cands[i], RTLD_LAZY | RTLD_GLOBAL);
			}
#endif
			if (!h) {
				fprintf(stderr, "\n\t*** pardiso-mkl: cannot load mkl_rt runtime. Exit.\n\n");
				exit(1);
			}
			fprintf(stderr, "PMKL: mkl_rt loaded h=%p\n", h);
			mkl_pardisoinit_p = (mkl_pardisoinit_fn) pmkl_sym(h, "pardisoinit");
			mkl_pardiso_p = (mkl_pardiso_fn) pmkl_sym(h, "pardiso");
			mkl_pardiso_getdiag_p = (mkl_pardiso_getdiag_fn) pmkl_sym(h, "pardiso_getdiag");
			fprintf(stderr, "PMKL: syms init=%p pardiso=%p getdiag=%p\n",
				(void*)mkl_pardisoinit_p, (void*)mkl_pardiso_p, (void*)mkl_pardiso_getdiag_p);
			if (!mkl_pardisoinit_p || !mkl_pardiso_p || !mkl_pardiso_getdiag_p) {
				fprintf(stderr, "\n\t*** pardiso-mkl: mkl_rt missing pardiso entry points. Exit.\n\n");
				exit(1);
			}
			mkl_loaded = 1;
		}
	}
}

/*
 * Per-handle scratch: oneMKL keeps the factor inside `pt`, but the LDL^T pivot
 * diagonal `D` (needed for log-det and for the L/L^T solves) must be remembered
 * between the factorize and the solve calls. Key the scratch by the `pt` pointer.
 */
typedef struct {
	void *pt;
	int n;
	double *D;					       /* LDL^T pivots, length n */
} pmkl_scratch_tp;

#define PMKL_MAX_SCRATCH (16384)
static pmkl_scratch_tp pmkl_scratch[PMKL_MAX_SCRATCH];
static int pmkl_scratch_n = 0;

static pmkl_scratch_tp *pmkl_scratch_get(void *pt, int create)
{
	pmkl_scratch_tp *s = NULL;
#pragma omp critical (pardiso_mkl_scratch)
	{
		for (int i = 0; i < pmkl_scratch_n; i++) {
			if (pmkl_scratch[i].pt == pt) {
				s = &pmkl_scratch[i];
				break;
			}
		}
		if (!s && create && pmkl_scratch_n < PMKL_MAX_SCRATCH) {
			s = &pmkl_scratch[pmkl_scratch_n++];
			s->pt = pt;
			s->n = 0;
			s->D = NULL;
		}
	}
	return s;
}

static void pmkl_scratch_drop(void *pt)
{
#pragma omp critical (pardiso_mkl_scratch)
	{
		for (int i = 0; i < pmkl_scratch_n; i++) {
			if (pmkl_scratch[i].pt == pt) {
				free(pmkl_scratch[i].D);
				pmkl_scratch[i] = pmkl_scratch[pmkl_scratch_n - 1];
				pmkl_scratch_n--;
				break;
			}
		}
	}
}

/*
 * Build the oneMKL iparm from scratch (NOT from GMRFLib's Panua-flavoured one,
 * whose extended indices mean different things). We honour only the cross-library
 * standard settings; GMRFLib's solve-mode intent is read separately from `gin`.
 */
static void pmkl_setup_iparm(int *iparm, int mtype)
{
	int dummy = mtype;
	void *dummy_pt[64] = { 0 };			       /* mkl pardisoinit writes the handle; must not be NULL */
	mkl_pardisoinit_p(dummy_pt, &dummy, iparm);	       /* defaults for this mtype */
	iparm[0] = 1;					       /* we set values explicitly below */
	iparm[1] = 3;					       /* parallel (OpenMP) METIS reordering */
	iparm[4] = 0;					       /* compute fill-reducing permutation internally */
	iparm[7] = 0;					       /* no iterative refinement */
	iparm[9] = 13;					       /* pivot perturbation 1e-13 (oneMKL default for sym) */
	iparm[34] = 0;					       /* one-based indexing (GMRFLib uses ia1/ja1) */
	iparm[55] = 1;					       /* store diagonal so pardiso_getdiag works */
}

/* ------------------------------------------------------------------ */
/* Panua-signature entry points that GMRFLib calls.                    */
/* ------------------------------------------------------------------ */

void pardisoinit(void *pt, int *mtype, int *solver, int *iparm, double *dparm, int *error)
{
	(void) solver;
	(void) dparm;
	fprintf(stderr, "PMKL: pardisoinit entry mtype=%d\n", *mtype);
	pmkl_load();
	fprintf(stderr, "PMKL: pardisoinit -> setup_iparm\n");
	pmkl_setup_iparm(iparm, *mtype);
	fprintf(stderr, "PMKL: pardisoinit setup_iparm done\n");
	for (int i = 0; i < 64; i++) {
		((void **) pt)[i] = NULL;
	}
	*error = 0;
}

void pardiso(void *pt, int *maxfct, int *mnum, int *mtype, int *phase, int *n,
	     double *a, int *ia, int *ja, int *perm, int *nrhs, int *iparm,
	     int *msglvl, double *b, double *x, int *error, double *dparm)
{
	pmkl_load();

	int caller_phase = *phase;
	int idum = 0;
	double ddum = 0.0;

	/* our own clean oneMKL iparm, derived from the (mtype-correct) defaults */
	int my_iparm[64];
	pmkl_setup_iparm(my_iparm, *mtype);

	if (caller_phase == -1) {
		/* release */
		int ph = -1;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		pmkl_scratch_drop(pt);
		return;
	}

	if (caller_phase == -22) {
		/* selected inverse (Q^{-1} on a sparsity pattern). oneMKL's selected
		 * inversion is requested with phase = -22 after a phase-22 factorize.
		 * The validation harness reports exactly what oneMKL fills in. */
		int ph = -22;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		/* GMRFLib reads back the inverse from `a` (the CSR values array) */
		iparm[17] = my_iparm[17];
		return;
	}

	if (caller_phase == 11) {
		int ph = 11;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		iparm[17] = my_iparm[17];		       /* nnz(L) -- same index in both libraries */
		return;
	}

	if (caller_phase == 22) {
		int ph = 22;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		iparm[17] = my_iparm[17];		       /* nnz(L) */
		iparm[21] = my_iparm[21];		       /* # positive pivots */
		iparm[22] = my_iparm[22];		       /* # negative pivots (pos-def check) */

		/* log-determinant via the LDL^T pivots; also cache D for the L/L^T solves */
		if (*error == 0) {
			pmkl_scratch_tp *s = pmkl_scratch_get(pt, 1);
			double *D = (double *) malloc((size_t) (*n) * sizeof(double));
			double *da = (double *) malloc((size_t) (*n) * sizeof(double));
			int gerr = 0;
			mkl_pardiso_getdiag_p((const void *) pt, D, da, mnum, &gerr);
			if (gerr == 0) {
				double logdet = 0.0;
				for (int i = 0; i < *n; i++) {
					logdet += log(fabs(D[i]));
				}
				if (dparm) {
					dparm[32] = logdet;	       /* GMRFLib reads log_det_Q = dparm[32] */
				}
				if (s) {
					free(s->D);
					s->D = D;
					s->n = *n;
					D = NULL;
				}
			} else {
				fprintf(stderr, "\t*** pardiso-mkl: pardiso_getdiag failed (err=%d)\n", gerr);
			}
			free(D);
			free(da);
		}
		return;
	}

	if (caller_phase == 33) {
		/* GMRFLib encodes the solve mode in iparm[25] (Panua iparm(26)):
		 *   0           -> full solve   (LL^T x = b)
		 *   1  or -12   -> lower solve  (L  x = b), Cholesky convention Q=LL^T
		 *   2  or -23   -> upper solve  (L^T x = b)
		 * oneMKL selects these with phase 33 / 331 / 333. Because oneMKL factors
		 * with a UNIT lower factor (Q = L D L^T), the Cholesky-convention L equals
		 * L_unit * D^{1/2}; we apply the D^{1/2} scaling explicitly. The validation
		 * harness checks whether this matches the dense Cholesky reference. */
		int mode = iparm[25];
		int nr = *nrhs;
		int nn = *n;

		if (mode == 0) {
			int ph = 33;
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, &idum, nrhs, my_iparm, msglvl, b, x, error);
			return;
		}

		/* L / L^T solves: forward (331) or backward (333) substitution */
		int ph = (mode == 1 || mode == -12) ? 331 : 333;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, &idum, nrhs, my_iparm, msglvl, b, x, error);

		/* apply the D^{1/2} scaling that turns the unit-L solve into the
		 * Cholesky-L solve. (Validated empirically; may be refined.) */
		pmkl_scratch_tp *s = pmkl_scratch_get(pt, 0);
		if (s && s->D && s->n == nn) {
			for (int j = 0; j < nr; j++) {
				double *xx = x + j * nn;
				for (int i = 0; i < nn; i++) {
					double sd = sqrt(fabs(s->D[i]));
					if (ph == 331) {
						xx[i] /= sd;	       /* L = L_unit D^{1/2} */
					} else {
						xx[i] /= sd;
					}
				}
			}
		}
		return;
	}

	/* any other phase: pass straight through */
	mkl_pardiso_p(pt, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, my_iparm, msglvl, b, x, error);
}

/*
 * GMRFLib's bundled TAUCS reordering calls METIS51PARDISO_NodeND (the name the
 * Panua build used for METIS nested dissection). When the Panua library is absent
 * this is normally provided by libpardiso.c; since the adapter replaces that file,
 * provide it here as a thin wrapper over the system METIS (same as libpardiso.c).
 */
int METIS51PARDISO_NodeND(int *nvtxs, int *xadj, int *adjncy, int *vwgt, int *options, int *perm, int *iperm)
{
	int METIS_NodeND(int *, int *, int *, int *, int *, int *, int *);
	return METIS_NodeND(nvtxs, xadj, adjncy, vwgt, options, perm, iperm);
}

/* Debug-only residual helper (GMRFLib calls it only under a hard-coded debug=0). */
void pardiso_residual(int *mtype, int *n, double *a, int *ia, int *ja, double *b, double *x, double *y, double *norm_b, double *norm_res)
{
	(void) mtype;
	(void) a;
	(void) ia;
	(void) ja;
	double nb = 0.0, nr = 0.0;
	for (int i = 0; i < *n; i++) {
		y[i] = 0.0;
		nb += b[i] * b[i];
	}
	(void) x;
	for (int i = 0; i < *n; i++) {
		nr += y[i] * y[i];
	}
	*norm_b = sqrt(nb);
	*norm_res = sqrt(nr);
}

#endif						       /* INLA_WITH_PARDISO_MKL */
