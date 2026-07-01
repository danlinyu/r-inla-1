# Building the R-INLA `inla` engine natively on Windows

`build-windows.sh` builds the `inla` engine from source on Windows — **no WSL, no GitHub
Actions**, no cross-compilation. It is the local, one-command distillation of the CI
workflow `.github/workflows/windows-native-enhancements.yml`, and produces the same
**statically-linked** engine: the numeric stack (OpenBLAS, GSL, METIS, muParser, zlib,
ltdl, gfortran/quadmath) and the GCC runtime (libgomp, libstdc++, libgcc, libwinpthread)
are baked into `inla.exe`. Only `R.dll` (your R install) and `Rmathfwd.dll` (the Rmath
forwarder into `R.dll`) stay dynamic — R cannot be statically linked.

Result: a drop-in `bundle/` containing **`inla.exe` + `Rmathfwd.dll`** (plus
`libcrypto-3-x64.dll` unless you opt into static crypto — see below).

## Prerequisites

1. **Rtools45** — install from CRAN (<https://cran.r-project.org/bin/windows/Rtools/>).
   Rtools45 ships the **MSYS2 UCRT64** MinGW-w64 toolchain this build uses. Start menu →
   **"Rtools45 UCRT64 shell"** (or run `C:\rtools45\usr\bin\bash.exe` with `MSYSTEM=UCRT64`).
   - Any standalone MSYS2 with the `UCRT64` environment also works; Rtools45 is just the
     supported, self-contained option.
2. **R 4.5** (a version INLA ships a Windows binary for) with its `bin` on `PATH` inside the
   UCRT64 shell, so `Rscript` resolves. Test: `Rscript -e 'R.version.string'`.
   The engine links against this R's `R.dll`, so build R and run R must match.
3. A clone of this repository. The script must be run from inside it (it locates the repo
   root relative to its own path).

The script installs everything else it needs (compilers, OpenBLAS, GSL, METIS, muParser,
zlib, OpenSSL, SuiteSparse, libtool) via `pacman`.

## Usage

From the **Rtools45 UCRT64** shell:

```bash
cd /path/to/r-inla
./build-configs/windows-native/build-windows.sh
```

That single command runs the full pipeline: pacman deps → SIMDE → GMRFLib (+ taucs) →
R import lib + `Rmathfwd.dll` → cgeneric headers → static `inlaprog` → `bundle/`.
It ends by printing the `bundle/` contents, an `ldd` dependency check, and the R install
command. Build time is a few minutes on a typical machine.

### Optional: also statically link libcrypto

By default `libcrypto` stays **dynamic** (OpenSSL static linking is brittle — it drags in
`ws2_32`/`crypt32`/`bcrypt`). To attempt a fully static crypto and reach the 2-file ideal
(`inla.exe` + `Rmathfwd.dll` only):

```bash
STATIC_CRYPTO=1 ./build-configs/windows-native/build-windows.sh
```

If that link fails, drop the flag — dynamic crypto is the safe, supported default.

## Installing the engine into R

The build prints the exact command. In R:

```r
d <- system.file("bin/windows/64bit", package = "INLA")
file.copy(list.files("C:/path/to/r-inla/bundle", full.names = TRUE), d, overwrite = TRUE)
```

`inla.call.builtin()` then resolves to this `inla.exe` automatically — it becomes the
**default** engine, no `inla.setOption(inla.call = ...)` override needed. (You can still
override explicitly with `inla.setOption(inla.call = "C:/path/to/bundle/inla.exe")`.)

## What the script does, step by step

| Step | Action |
|------|--------|
| 1 | `pacman -S --needed` the UCRT64 toolchain + numeric libs |
| 2 | Vendor header-only **SIMDE** v0.8.2 (no MSYS2 package exists) |
| 3 | Build + install **GMRFLib**; stage the vendored **taucs** static lib |
| 4 | `gendef`/`dlltool` an import lib for `R.dll`; build the **`Rmathfwd.dll`** forwarder (unprefixed Rmath names → `R.Rf_*`) |
| 5 | Generate the **cgeneric** headers (`external-packages/update-cgeneric`) |
| 6 | Build **`inlaprog`** with `-static` (+ `-Wl,-Bdynamic` islands for R/Rmath/crypto) |
| 7 | Assemble **`bundle/`** via the transitive DLL closure and print the install step |

## Verifying against the official engine

The CI workflow additionally runs a 6-model validation (gaussian/poisson/binomial ×
fixed/iid/rw1/besag/SPDE-2D, eb + ccd) comparing this engine to the official INLA engine
(hyperparameters compared on the **median**, the stable metric under static-BLAS FP
reordering). To reproduce locally, install the in-repo `rinla/` package with `bundle/`
embedded and fit a model with the default engine; see the `VALIDATION` step in
`.github/workflows/windows-native-enhancements.yml` for the exact script.

## Troubleshooting

- **`Rscript: command not found`** — R's `bin` is not on `PATH` in the UCRT64 shell.
- **`cannot find -lfoo`** — a UCRT64 package updated and no longer ships a static `.a`.
  Bracket that one lib in `-Wl,-Bdynamic -lfoo -Wl,-Bstatic` inside the `inlaprog` link
  step (it then ships as a DLL in `bundle/`).
- **Engine runs but R can't find it** — confirm `R.dll` is reachable (a working R on PATH)
  and that `Rmathfwd.dll` sits next to `inla.exe` in `bin/windows/64bit`.
