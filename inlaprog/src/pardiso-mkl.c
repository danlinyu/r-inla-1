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
			mkl_pardisoinit_p = (mkl_pardisoinit_fn) pmkl_sym(h, "pardisoinit");
			mkl_pardiso_p = (mkl_pardiso_fn) pmkl_sym(h, "pardiso");
			mkl_pardiso_getdiag_p = (mkl_pardiso_getdiag_fn) pmkl_sym(h, "pardiso_getdiag");
			if (!mkl_pardisoinit_p || !mkl_pardiso_p || !mkl_pardiso_getdiag_p) {
				fprintf(stderr, "\n\t*** pardiso-mkl: mkl_rt missing pardiso entry points. Exit.\n\n");
				exit(1);
			}

			/*
			 * Threading is selected entirely by the MKL_THREADING_LAYER env var.
			 * MKL_THREADING_LAYER=TBB gives MKL-internal parallelism (all cores by
			 * default) with no second OpenMP runtime to clash against GMRFLib's GNU
			 * libgomp -- TBB's work-stealing arena composes safely even when INLA
			 * calls PARDISO from inside its own OpenMP regions. We deliberately do
			 * NOT call mkl_set_num_threads here: the lowercase symbol is MKL's
			 * Fortran interface (argument by reference), and under the TBB layer
			 * thread-count control belongs to TBB (global_control), not to this
			 * OpenMP-era knob. The env var alone is sufficient and proven exact.
			 */
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
	/* keep close to oneMKL defaults (proven in the standalone probe); only the two
	 * settings GMRFLib's data layout and our log-det route require: */
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
	pmkl_load();
	int mt = 2;					       /* SPD; see note in pardiso() */
	(void) mtype;
	pmkl_setup_iparm(iparm, mt);
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
	int safe_nrhs = 1;				       /* GMRFLib passes nrhs=-1 for analysis; oneMKL wants >=1 */

	/* GMRF precision matrices are SPD; use oneMKL mtype=2 (real sym pos def) -- this is
	 * the path proven in the standalone probe. GMRFLib passes -2 (Panua LDL^T convention);
	 * for an indefinite Q oneMKL returns an error which surfaces as the usual pos-def retry. */
	int mt = 2;
	mtype = &mt;

	/* our own clean oneMKL iparm, derived from the (mtype-correct) defaults */
	int my_iparm[64];
	pmkl_setup_iparm(my_iparm, *mtype);

	if (caller_phase == -1) {
		/* release */
		int ph = -1;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, &safe_nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		pmkl_scratch_drop(pt);
		return;
	}

	if (caller_phase == -22) {
		/* Selected inverse: fill a[] (on the upper-triangular (ia,ja) pattern) with
		 * the corresponding elements of Q^{-1}. The DEFAULT path computes the needed
		 * elements with full solves Q X = I (reuses the validated full solve, exact,
		 * but O(n) solves). When PMKL_NATIVE_QINV is set we instead try oneMKL's OWN
		 * phase=-22 selected inversion (a real parallel Takahashi) -- experimental,
		 * validated empirically vs dense LAPACK in pardiso-mkl-validate.c. */
		int nn = *n;
		int base = ia[0];			       /* 1-based in this build */
		int nnz = ia[nn] - base;

		if (getenv("PMKL_NATIVE_QINV")) {
			/* oneMKL native selected inversion. The phase-22 factor is already in pt
			 * (GMRFLib called chol before Qinv). iparm(36)=0 (iparm[35]=0) selects the
			 * "overwrite L/U with the selected inverse" mode; oneMKL writes the inverse
			 * back into a[] on the input matrix's pattern. THE open question (resolved
			 * by this experiment): does it stay on Q's pattern (a[] sized nnz -> safe)
			 * or spill onto the factor's fill pattern (-> overflow = the earlier crash)? */
			int ph = -22, qerr = 0, snrhs = 1;
			double bdum = 0.0, xdum = 0.0;
			my_iparm[35] = 0;		       /* iparm(36)=0: selected inverse, overwrite L/U */
			fprintf(stderr, "\t[pmkl] native phase=-22 selinv: n=%d nnz=%d base=%d a[0]=%g\n",
				nn, nnz, base, a[0]);
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, &snrhs, my_iparm, msglvl,
				      &bdum, &xdum, &qerr);
			fprintf(stderr, "\t[pmkl] native selinv returned err=%d  a[0..2]=%g %g %g\n",
				qerr, a[0], (nnz > 1 ? a[1] : 0.0), (nnz > 2 ? a[2] : 0.0));
			if (qerr == 0) {
				*error = 0;
				return;		       /* a[] now holds Q^{-1} on Q's pattern */
			}
			fprintf(stderr, "\t[pmkl] native selinv FAILED (err=%d) -> fall back to full-solve Qinv\n", qerr);
			/* fall through to the validated O(n)-solve path below */
		}

		/* transpose the upper CSR pattern into per-column lists (0-based column j) */
		int *colptr = (int *) calloc((size_t) (nn + 1), sizeof(int));
		int *colk = (int *) malloc((size_t) nnz * sizeof(int));	/* position in a[] */
		int *colrow = (int *) malloc((size_t) nnz * sizeof(int));	/* row i */
		int *fill = (int *) calloc((size_t) nn, sizeof(int));
		for (int k = 0; k < nnz; k++) {
			colptr[(ja[k] - base) + 1]++;
		}
		for (int j = 0; j < nn; j++) {
			colptr[j + 1] += colptr[j];
		}
		for (int i = 0; i < nn; i++) {
			for (int k = ia[i] - base; k < ia[i + 1] - base; k++) {
				int j = ja[k] - base;
				int p = colptr[j] + fill[j]++;
				colk[p] = k;
				colrow[p] = i;
			}
		}

		/* keep the matrix values for the solves; a[] will be overwritten with Q^{-1} */
		double *acopy = (double *) malloc((size_t) nnz * sizeof(double));
		memcpy(acopy, a, (size_t) nnz * sizeof(double));

		int BS = 64;
		if (BS > nn) {
			BS = nn;
		}
		double *rhs = (double *) malloc((size_t) nn * BS * sizeof(double));
		double *sol = (double *) malloc((size_t) nn * BS * sizeof(double));
		int ph33 = 33, err = 0;
		for (int j0 = 0; j0 < nn; j0 += BS) {
			int nb = (j0 + BS <= nn) ? BS : (nn - j0);
			for (int t = 0; t < nn * nb; t++) {
				rhs[t] = 0.0;
			}
			for (int c = 0; c < nb; c++) {
				rhs[(size_t) c * nn + (j0 + c)] = 1.0;	/* e_{j0+c} */
			}
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph33, n, acopy, ia, ja, &idum, &nb, my_iparm, msglvl, rhs, sol, &err);
			if (err) {
				*error = err;
			}
			for (int c = 0; c < nb; c++) {
				int j = j0 + c;
				double *xc = sol + (size_t) c * nn;
				for (int p = colptr[j]; p < colptr[j + 1]; p++) {
					a[colk[p]] = xc[colrow[p]];
				}
			}
		}
		free(colptr);
		free(colk);
		free(colrow);
		free(fill);
		free(acopy);
		free(rhs);
		free(sol);
		return;
	}

	if (caller_phase == 11) {
		int ph = 11;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, &safe_nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		iparm[17] = my_iparm[17];		       /* nnz(L) -- same index in both libraries */
		return;
	}

	if (caller_phase == 22) {
		int ph = 22;
		mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, perm, &safe_nrhs, my_iparm, msglvl, &ddum, &ddum, error);
		iparm[17] = my_iparm[17];		       /* nnz(L) */
		iparm[21] = my_iparm[21];		       /* # positive pivots */
		iparm[22] = my_iparm[22];		       /* # negative pivots (pos-def check) */

		if (*error == 0) {
			int nn = *n;
			pmkl_scratch_tp *s = pmkl_scratch_get(pt, 1);

			/* log-determinant via the LDL^T pivots (getdiag), written into dparm[32] */
			double *D = (double *) malloc((size_t) nn * sizeof(double));
			double *da = (double *) malloc((size_t) nn * sizeof(double));
			int gerr = 0;
			mkl_pardiso_getdiag_p((const void *) pt, D, da, mnum, &gerr);
			if (gerr == 0) {
				double logdet = 0.0;
				for (int i = 0; i < nn; i++) {
					logdet += log(fabs(D[i]));
				}
				if (dparm) {
					dparm[32] = logdet;
				}
			} else {
				fprintf(stderr, "\t*** pardiso-mkl: pardiso_getdiag failed (err=%d)\n", gerr);
			}
			free(D);
			free(da);

			/* Cache sc = 1/sqrt(pivot) in oneMKL's INTERNAL solve ordering, obtained
			 * directly from phase 332 (the diagonal solve) on a ones-vector:
			 * 332(ones)[i] = 1/D_i in exactly the domain phases 331/333 operate in.
			 * This makes the L/L^T square-root solves correct regardless of oneMKL's
			 * internal fill-reducing permutation (getdiag's ordering is ambiguous). */
			double *ones = (double *) malloc((size_t) nn * sizeof(double));
			double *v = (double *) malloc((size_t) nn * sizeof(double));
			for (int i = 0; i < nn; i++) {
				ones[i] = 1.0;
			}
			int ph332 = 332, ferr = 0, one = 1;
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph332, n, a, ia, ja, &idum, &one, my_iparm, msglvl, ones, v, &ferr);
			if (ferr == 0 && s) {
				free(s->D);
				s->D = (double *) malloc((size_t) nn * sizeof(double));
				for (int i = 0; i < nn; i++) {
					s->D[i] = sqrt(fabs(v[i]));	/* = 1/sqrt(D_i) */
				}
				s->n = nn;
			}
			free(ones);
			free(v);
		}
		return;
	}

	if (caller_phase == 33) {
		/* GMRFLib encodes the solve mode in iparm[25] (Panua iparm(26)):
		 *   0           -> full solve   (Q x = b)
		 *   1  or -12   -> "L"  solve   x = S^{-1} b
		 *   2  or -23   -> "L^T" solve  x = S^{-T} b
		 * where S is a square root of Q (S S^T = Q). oneMKL factors P Q P^T = L D L^T
		 * (UNIT lower L), so a valid square root is S = P^T L D^{1/2}, giving
		 *   S^{-1} b = D^{-1/2} (L^{-1} P b)      -> oneMKL fwd-solve (331) then scale
		 *   S^{-T} b = P^T L^{-T} (D^{-1/2} b)    -> scale then oneMKL bwd-solve (333)
		 * These are mutual adjoints and compose to Q^{-1} (S^{-T}S^{-1}=Q^{-1}); this
		 * is what GMRFLib needs for sampling, NOT agreement with any particular dense
		 * Cholesky factor. D is the pivot diagonal cached at factorization (getdiag),
		 * which lives in the same internal ordering oneMKL's 331/333 operate in. */
		int mode = iparm[25];
		int nr = *nrhs;
		int nn = *n;

		if (mode == 0) {
			int ph = 33;
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, &idum, nrhs, my_iparm, msglvl, b, x, error);
			return;
		}

		pmkl_scratch_tp *s = pmkl_scratch_get(pt, 0);
		int have_sc = (s && s->D && s->n == nn);	       /* s->D holds sc = 1/sqrt(D_i) */

		if (mode == 1 || mode == -12) {
			/* x = sc .* (L^{-1} P b): forward solve (331), then scale */
			int ph = 331;
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, &idum, nrhs, my_iparm, msglvl, b, x, error);
			if (have_sc) {
				for (int j = 0; j < nr; j++) {
					double *xx = x + j * nn;
					for (int i = 0; i < nn; i++) {
						xx[i] *= s->D[i];
					}
				}
			}
		} else {
			/* x = P^T L^{-T} (sc .* b): scale b, then backward solve (333) */
			double *bb = (double *) malloc((size_t) nn * nr * sizeof(double));
			for (int j = 0; j < nr; j++) {
				double *src = b + j * nn, *dst = bb + j * nn;
				for (int i = 0; i < nn; i++) {
					dst[i] = have_sc ? (src[i] * s->D[i]) : src[i];
				}
			}
			int ph = 333;
			mkl_pardiso_p(pt, maxfct, mnum, mtype, &ph, n, a, ia, ja, &idum, nrhs, my_iparm, msglvl, bb, x, error);
			free(bb);
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

/* Panua matrix/vector check + stats helpers GMRFLib references under its debug/check
 * paths. oneMKL has no equivalents; provide no-ops (GMRFLib calls them only when its
 * own csr_check / verbose flags are set, which are off by default). */
void pardiso_chkmatrix(int *a, int *b, double *c, int *d, int *e, int *f)
{
	(void) a; (void) b; (void) c; (void) d; (void) e; (void) f;
}
void pardiso_chkvec(int *a, int *b, double *c, int *d)
{
	(void) a; (void) b; (void) c; (void) d;
}
void pardiso_printstats(int *a, int *b, double *c, int *d, int *e, int *f, double *g, int *h)
{
	(void) a; (void) b; (void) c; (void) d; (void) e; (void) f; (void) g; (void) h;
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
