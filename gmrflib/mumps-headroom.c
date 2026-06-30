/*
 * mumps-headroom.c -- apples-to-apples headroom benchmark for the MUMPS
 * selected-inversion track. Builds ONE besag-like SPD GMRF precision Q on a
 * 110x110 lattice (n = 12100) and computes the factorization + the FULL selected
 * inverse on Q's pattern two ways, in the same process, no INLA/R overhead:
 *
 *   (A) GMRFLib taucs path: GMRFLib_init_problem (Cholesky) + GMRFLib_Qinv
 *       (the existing serial Takahashi recursion);
 *   (B) MUMPS: factorize (job=4) + ICNTL(30) selected inversion over Q's
 *       lower-triangular pattern (job=3), OpenMP-parallel libseq variant.
 *
 * Reports the factor time and the selected-inverse time for each, plus a
 * correctness cross-check of a few diagonal entries. The question: does MUMPS's
 * parallel selected inversion beat taucs's Takahashi at this scale?
 *
 * Exploratory tool on the mumps-selinv branch; not part of any PR.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "dmumps_c.h"
#include "GMRFLib/GMRFLib.h"

#define JOB_INIT -1
#define JOB_END  -2
#define USE_COMM_WORLD -987654
#define ICNTL(I)  icntl[(I)-1]
#define INFOG(I)  infog[(I)-1]

/* inlaprog symbols libGMRFLib references; stubbed (we link only GMRFLib). */
int inla_ncpu(void) { return 1; }
void *inla_stiles_get_setup(void *m) { (void) m; return NULL; }

/* GMRFLib's unconditionally-compiled smtp-pardiso.o and the vendored taucs ordering
 * reference these symbols at link time. With the taucs path they come from inlaprog's
 * libpardiso.c, which we do not link. METIS51PARDISO_NodeND IS used by taucs's
 * nested-dissection ordering -> wrap the system METIS. The pardiso entry points are
 * never called with smtp=TAUCS -> no-op stubs just to satisfy the linker. */
int METIS51PARDISO_NodeND(int *nvtxs, int *xadj, int *adjncy, int *vwgt, int *options, int *perm, int *iperm)
{
	int METIS_NodeND(int *, int *, int *, int *, int *, int *, int *);
	return METIS_NodeND(nvtxs, xadj, adjncy, vwgt, options, perm, iperm);
}
void pardisoinit(void *a, int *b, int *c, int *d, double *e, int *f)
{ (void) a; (void) b; (void) c; (void) d; (void) e; (void) f; }
void pardiso(void *a, int *b, int *c, int *d, int *e, int *f, double *g, int *h, int *i, int *j,
	     int *k, int *l, int *m, double *nn, double *o, int *p, double *q)
{ (void) a; (void) b; (void) c; (void) d; (void) e; (void) f; (void) g; (void) h; (void) i; (void) j;
  (void) k; (void) l; (void) m; (void) nn; (void) o; (void) p; (void) q; }
void pardiso_residual(int *mtype, int *nn, double *a, int *ia, int *ja, double *b, double *x, double *y,
		      double *norm_b, double *norm_res)
{ (void) mtype; (void) nn; (void) a; (void) ia; (void) ja; (void) b; (void) x; (void) y; (void) norm_b; (void) norm_res; }
void pardiso_chkmatrix(int *a, int *s, double *d, int *f, int *g, int *h)
{ (void) a; (void) s; (void) d; (void) f; (void) g; (void) h; }
void pardiso_chkvec(int *a, int *s, double *d, int *f) { (void) a; (void) s; (void) d; (void) f; }
void pardiso_printstats(int *a, int *s, double *d, int *f, int *g, int *h, double *j, int *k)
{ (void) a; (void) s; (void) d; (void) f; (void) g; (void) h; (void) j; (void) k; }

/* SPD besag-like precision on the lattice graph: Q = (D + I) - W,
 * diagonally dominant -> SPD. The graph is passed as Qfunc_args. */
static double Qfunc(int UNUSED(thread_id), int i, int j, double *UNUSED(values), void *arg)
{
	GMRFLib_graph_tp *g = (GMRFLib_graph_tp *) arg;
	if (j < 0) {
		return NAN;
	}
	if (i == j) {
		return (double) g->nnbs[i] + 1.0;
	}
	return -1.0;
}

int main(int argc, char **argv)
{
	int ierr, myid;
	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	setvbuf(stdout, NULL, _IONBF, 0);

	/* replicate the engine startup init (as in pardiso-mkl-validate.c) */
	GMRFLib_numa_init();
	GMRFLib_openmp = Calloc(1, GMRFLib_openmp_tp);
	GMRFLib_openmp->max_threads = omp_get_max_threads();
	GMRFLib_openmp->max_threads2 = 2 * (omp_get_max_threads() + 1);
	GMRFLib_openmp->blas_num_threads_force = 0;
	GMRFLib_openmp->max_threads_nested = Calloc(3, int);
	GMRFLib_openmp->max_threads_nested[0] = GMRFLib_openmp->max_threads;
	GMRFLib_openmp->max_threads_nested[1] = 1;
	GMRFLib_openmp->max_threads_nested[2] = 1;
	GMRFLib_openmp->adaptive = 0;
	GMRFLib_openmp->schedule = omp_sched_guided;
	GMRFLib_openmp->chunk_size = 0;
	GMRFLib_openmp->likelihood_nt = 0;
	GMRFLib_set_error_handler(NULL);
	GMRFLib_init_constr_store();
	GMRFLib_init_constr_store_logdet();
	GMRFLib_graph_init_store();
	GMRFLib_remap_init_store();
	GMRFLib_csr_init_store();
	GMRFLib_openmp->strategy = GMRFLib_OPENMP_STRATEGY_DEFAULT;
	GMRFLib_openmp_implement_strategy(GMRFLib_OPENMP_PLACES_OPTIMIZE, NULL, NULL);
	GMRFLib_smtp = GMRFLib_SMTP_TAUCS;

	int side = 110, n = side * side;		       /* 12100 nodes, bandwidth ~110 */
	GMRFLib_graph_tp *g = NULL;
	GMRFLib_graph_mk_lattice(&g, side, side, 1, 1, 0);
	printf("lattice %dx%d, n=%d, threads=%d\n", side, side, n, omp_get_max_threads());

	/* ---------- (A) GMRFLib taucs: factor + Takahashi Qinv ---------- */
	double t0 = omp_get_wtime();
	GMRFLib_problem_tp *problem = NULL;
	GMRFLib_init_problem(0, &problem, NULL, NULL, NULL, NULL, g, Qfunc, (void *) g, NULL, NULL, NULL);
	double t1 = omp_get_wtime();
	GMRFLib_Qinv(problem);
	double t2 = omp_get_wtime();
	printf("[TAUCS]  factor = %.3f s   Qinv(Takahashi) = %.3f s   total = %.3f s\n",
	       t1 - t0, t2 - t1, t2 - t0);

	/* ---------- (B) MUMPS: factor + ICNTL(30) selected inverse ----------
	 * Assemble Q's LOWER triangle (sym=1). Also build the per-column request
	 * pattern (CSC) for the selected inverse over exactly that pattern. */
	MUMPS_INT8 nnz = 0;
	for (int i = 0; i < n; i++) {
		nnz++;					       /* diagonal */
		for (int k = 0; k < g->nnbs[i]; k++) {
			if (g->nbs[i][k] < i) {
				nnz++;			       /* lower-triangle neighbour (row i > col j) */
			}
		}
	}
	MUMPS_INT *irn = malloc((size_t) nnz * sizeof(MUMPS_INT));
	MUMPS_INT *jcn = malloc((size_t) nnz * sizeof(MUMPS_INT));
	double    *a   = malloc((size_t) nnz * sizeof(double));
	/* request pattern by column: column c (1-based) holds row c (diag) and rows
	 * r>c that are neighbours of c -> CSC irhs_ptr/irhs_sparse. */
	MUMPS_INT *colcnt = calloc((size_t) (n + 2), sizeof(MUMPS_INT));
	for (int i = 0; i < n; i++) {
		colcnt[i + 1]++;			       /* diagonal in column i+1 */
		for (int k = 0; k < g->nnbs[i]; k++) {
			int jj = g->nbs[i][k];
			if (jj > i) {
				colcnt[i + 1]++;	       /* entry (jj+1, i+1) in column i+1 */
			}
		}
	}
	MUMPS_INT nz_rhs = 0;
	for (int c = 1; c <= n; c++) {
		nz_rhs += colcnt[c];
	}
	MUMPS_INT *irhs_ptr = malloc((size_t) (n + 1) * sizeof(MUMPS_INT));
	MUMPS_INT *irhs_sparse = malloc((size_t) nz_rhs * sizeof(MUMPS_INT));
	double    *rhs_sparse = malloc((size_t) nz_rhs * sizeof(double));
	irhs_ptr[0] = 1;
	for (int c = 1; c <= n; c++) {
		irhs_ptr[c] = irhs_ptr[c - 1] + colcnt[c];
	}
	/* fill assembled lower triangle AND the request pattern (same set) */
	MUMPS_INT *fillpos = malloc((size_t) (n + 1) * sizeof(MUMPS_INT));
	for (int c = 1; c <= n; c++) {
		fillpos[c] = irhs_ptr[c - 1];		       /* next write slot for column c */
	}
	MUMPS_INT e = 0;
	for (int i = 0; i < n; i++) {
		/* diagonal (i+1,i+1) */
		irn[e] = i + 1; jcn[e] = i + 1; a[e] = (double) g->nnbs[i] + 1.0; e++;
		int c = i + 1;
		irhs_sparse[fillpos[c] - 1] = i + 1; fillpos[c]++;   /* diag row */
		for (int k = 0; k < g->nnbs[i]; k++) {
			int jj = g->nbs[i][k];
			if (jj < i) {			       /* lower triangle entry (i+1, jj+1) */
				irn[e] = i + 1; jcn[e] = jj + 1; a[e] = -1.0; e++;
			}
			if (jj > i) {			       /* request (jj+1) in column i+1 */
				irhs_sparse[fillpos[c] - 1] = jj + 1; fillpos[c]++;
			}
		}
	}
	printf("MUMPS assembled nnz(lower)=%lld   selinv request nz_rhs=%d\n", (long long) nnz, (int) nz_rhs);

	DMUMPS_STRUC_C id;
	id.comm_fortran = USE_COMM_WORLD;
	id.par = 1; id.sym = 1; id.job = JOB_INIT;
	dmumps_c(&id);
	id.n = n; id.nnz = nnz; id.irn = irn; id.jcn = jcn; id.a = a;
	id.ICNTL(1) = 6; id.ICNTL(2) = 0; id.ICNTL(3) = 0; id.ICNTL(4) = 1;

	double m0 = omp_get_wtime();
	id.job = 4;					       /* analyse + factorize */
	dmumps_c(&id);
	double m1 = omp_get_wtime();
	if (id.INFOG(1) < 0) {
		printf("MUMPS factorize FAILED INFOG(1)=%d INFOG(2)=%d\n", id.INFOG(1), id.INFOG(2));
		id.job = JOB_END; dmumps_c(&id); MPI_Finalize(); return 10;
	}
	id.ICNTL(30) = 1;
	id.nrhs = n; id.lrhs = n;
	id.nz_rhs = nz_rhs;
	id.irhs_ptr = irhs_ptr;
	id.irhs_sparse = irhs_sparse;
	id.rhs_sparse = rhs_sparse;
	id.job = 3;					       /* selected inverse */
	dmumps_c(&id);
	double m2 = omp_get_wtime();
	printf("[MUMPS]  factor = %.3f s   selinv(ICNTL30) = %.3f s   total = %.3f s   INFOG(1)=%d\n",
	       m1 - m0, m2 - m1, m2 - m0, id.INFOG(1));

	/* ---------- correctness cross-check: a few diagonal entries ---------- */
	if (id.INFOG(1) >= 0) {
		double maxerr = 0.0;
		int checked = 0;
		for (int c = 1; c <= n; c += (n / 7 + 1)) {
			/* first entry of column c in rhs_sparse is the diagonal A^-1(c,c) */
			double mumps_diag = rhs_sparse[irhs_ptr[c - 1] - 1];
			double *tp = GMRFLib_Qinv_get(problem, c - 1, c - 1);
			double taucs_diag = tp ? *tp : NAN;
			double er = fabs(mumps_diag - taucs_diag);
			if (er > maxerr) maxerr = er;
			printf("  check Qinv(%d,%d): mumps=%.8f taucs=%.8f err=%.2e\n",
			       c - 1, c - 1, mumps_diag, taucs_diag, er);
			checked++;
		}
		printf("CROSS-CHECK (%d diag entries) max err = %.2e\n", checked, maxerr);
	}

	id.job = JOB_END;
	dmumps_c(&id);
	MPI_Finalize();
	printf("HEADROOM DONE\n");
	return 0;
}
