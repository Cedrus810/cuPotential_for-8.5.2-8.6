// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// usage.cu — End-to-end C++ demo of the prolate splittings and compact convolution.
//
// Two independent pipelines exercising the public scalar LTO-IR APIs
// (cupo/math/splitting.h, cupo/math/spreading.h) and runtime grid planner against an
// analytical reference blob for a solvated system (water or apo protein + ions;
// 1-2/1-3 pairs excluded, 1-4 and beyond at full strength):
//
//   1. Coulomb (prolate P=1):
//        E_total = E_short + E_recip + E_self - E_excl_LR
//      with cupo_p1_short_range_{value,grad}_f64, the exact
//      structure-factor sum using cupo_p1_fourier_kernel_value_f64,
//      cupo_p1_self_value_f64, and the long-range pair subtraction
//      for excluded pairs.
//
//   2. Dispersion (prolate P=6, C6-only):
//      E_disp = E_R6_short + E_R6_recip + E_R6_self - E_R6_excl_LR — the
//      attractive 1/r⁶ PME-split like Coulomb (cupo_p6_* kernels, sign flip),
//      with geometric combining C6_ij = √C6_i·√C6_j (√C6 = 2√ε·σ³).  The
//      reference is C6-only, so this is what the energy/force/virial PASS/FAIL
//      compares.
//
// By default the reciprocal sum is the exact structure-factor sum above.
// Pass --particle-mesh to instead run a full double-precision host PME mesh
// reciprocal driven by the cupo/math/spreading.h C2OSpline<3> spreading API
// (cupo_spread_1d_f64 → 3-D DFT → discrete deconvolution kernel →
// inverse DFT → cupo_interpolate_1d_f64).  The mesh size is auto-planned a-priori
// from the box + Prolate plan by cupo/kspace.h's cupo_plan_grid (calibrated to
// 5e-4 combined relative L2 force error), or overridden with --grid N. Both
// pipelines (Coulomb and dispersion) honour the flag. See the error-criterion
// and numerical-band choices below; this sample reports independent errors.
// This sample links libcupo.so for planning and compact convolution. --compact uses
// cupo_plan_grid with CUPO_GRID_COMPACT and cupo_convolution_execute on the GPU.
// Spreading, force interpolation, energy and virial reduction stay on the host.
//
// Both pipelines also compute the virial / pressure tensor W = −∂E/∂ε = Σ r⊗F
// (Voigt: xx,yy,zz,xy,xz,yz) analytically — real-space pair + reciprocal
// (cupo_p{1,6}_fourier_kernel_grad_f64; exact or mesh) + exclusion — and compare
// it to the blob's analytical jax-autodiff Ewald reference (Coulomb; C6-only
// dispersion). Energy and virial accuracy are checked separately; meeting a
// force-error target does not guarantee the same accuracy for either quantity.
//
// The exact and --particle-mesh modes require no CUDA execution. Only --compact
// allocates device grids and runs kernels; it uses FP32 input/output and FFTs.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "cupo/math/splitting.h"
#include "cupo/math/spreading.h"
#include "cupo/splitting_plan.h"
#include "cupo/kspace.h"
#include "cupo/version.h"
#include <cupo/convolution.h>
#include <cuda_runtime_api.h>

static constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// Choose either relative L2 force-error metric (dF = F - F_ref):
//   Combined:    ||dF_C + dF_D|| / ||F_C_ref + F_D_ref||.
//   Independent: ||dF_C|| / ||F_C_ref|| and ||dF_D|| / ||F_D_ref||.
// Norms cover all atoms/components; F_C and F_D are physical Coulomb and
// attractive C6 forces, including prefactors/signs (no C12 or bonded forces).
//
// Numerical bands for --compact: z_cut = 2*pi*|f_cut|*rc, f in cycles/length.
//   Criterion     Cutoff             P1 z_cut   P6 z_cut
//   Independent   6 or 9 Å           10.75      12.764031957789484 (default)
//   Combined      6 Å (0.6 nm)       10.75      11.5
//   Combined      9 Å (0.9 nm)       10.75       7.0
// The combined choices met 5e-4 on sampled charged water/protein systems with
// the current compact real grids. Independent values retain the default bands;
// validate each channel against its required tolerance. Set the constants below
// for the desired cutoff/criterion; this sample checks independent errors.
static constexpr double kP1CompactBandwidth = 10.75;
static constexpr double kP6CompactBandwidth = 12.764031957789484;

static bool outside_fourier_band(double k2, double cutoff, double bandwidth)
{
    // cupo/math/splitting.h takes frequencies in cycles/length; bandwidth is angular k * rc.
    return kTwoPi * cutoff * std::sqrt(k2) > bandwidth;
}

static int compact_support_radius(double length, double cutoff, int grid_length,
                                  double bandwidth)
{
    // Last axis mode satisfying 2*pi*|f|*rc <= z_cut, clamped to Nyquist.
    // Use the same predicate and frequency arithmetic as the radial mask;
    // testing scalar values for zero would retain the full oscillatory tail.
    const double inv_length = 1.0 / length;
    int radius = 0;
    for (int m = 1; m <= grid_length / 2; ++m) {
        const double k = m * inv_length;
        if (outside_fourier_band(k*k, cutoff, bandwidth)) break;
        radius = m;
    }
    return radius;
}

// Finite integer shell for the illustrative direct reciprocal sum. The C2
// Fourier complement has nonzero tails; this is a truncation choice, not exact
// support. Increase this bound to check convergence for another system/target.
static inline int recip_m_per_dim(const double box[3], double rc)
{
    const double L_max = std::max({box[0], box[1], box[2]});
    return int(std::ceil(4.0 * L_max / rc)) + 2;
}

// ── Binary-blob layout (must match gen_reference_bin.py) ────────────────────

struct Reference {
    uint32_t n_atoms;
    uint32_t n_excl;
    double box[3];
    double cutoff;
    double coulomb_constant;        // electrostatic prefactor 1/(4πε₀)  [kJ·nm/(mol·e²)]
    std::vector<double>  coords;    // 3·N
    std::vector<double>  charges;   //   N
    std::vector<double>  sigma;     //   N
    std::vector<double>  epsilon;   //   N
    std::vector<int32_t> excl;      // 2·M
    double ref_E_coul;
    std::vector<double>  ref_F_coul;
    double ref_E_disp;              // C6-only dispersion energy (geometric)
    std::vector<double>  ref_F_disp; // C6-only dispersion forces
    double ref_W_coul[6];           // analytical virial (Voigt: xx,yy,zz,xy,xz,yz)
    double ref_W_disp[6];           // C6-only dispersion virial (Voigt)
};

static Reference load_reference(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "ERROR: cannot open '%s'\n", path.c_str());
        std::exit(1);
    }
    Reference r;
    in.read(reinterpret_cast<char*>(&r.n_atoms), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&r.n_excl),  sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(r.box),     sizeof(double) * 3);
    in.read(reinterpret_cast<char*>(&r.cutoff), sizeof(double));
    in.read(reinterpret_cast<char*>(&r.coulomb_constant), sizeof(double));
    r.coords    .resize(3 * r.n_atoms);
    r.charges   .resize(    r.n_atoms);
    r.sigma     .resize(    r.n_atoms);
    r.epsilon   .resize(    r.n_atoms);
    r.excl      .resize(2 * r.n_excl);
    r.ref_F_coul.resize(3 * r.n_atoms);
    r.ref_F_disp.resize(3 * r.n_atoms);
    in.read(reinterpret_cast<char*>(r.coords  .data()), sizeof(double) * r.coords  .size());
    in.read(reinterpret_cast<char*>(r.charges .data()), sizeof(double) * r.charges .size());
    in.read(reinterpret_cast<char*>(r.sigma   .data()), sizeof(double) * r.sigma   .size());
    in.read(reinterpret_cast<char*>(r.epsilon .data()), sizeof(double) * r.epsilon .size());
    in.read(reinterpret_cast<char*>(r.excl    .data()), sizeof(int32_t)* r.excl    .size());
    in.read(reinterpret_cast<char*>(&r.ref_E_coul), sizeof(double));
    in.read(reinterpret_cast<char*>(r.ref_F_coul.data()), sizeof(double) * r.ref_F_coul.size());
    in.read(reinterpret_cast<char*>(&r.ref_E_disp), sizeof(double));
    in.read(reinterpret_cast<char*>(r.ref_F_disp.data()), sizeof(double) * r.ref_F_disp.size());
    in.read(reinterpret_cast<char*>(r.ref_W_coul), sizeof(double) * 6);
    in.read(reinterpret_cast<char*>(r.ref_W_disp), sizeof(double) * 6);
    if (!in) {
        fprintf(stderr, "ERROR: short read from '%s'\n", path.c_str());
        std::exit(1);
    }
    return r;
}

// Pack a sorted (i, j) pair into a 64-bit key for the exclusion set.
static inline uint64_t pack_pair(int i, int j)
{
    if (i > j) std::swap(i, j);
    return (uint64_t(uint32_t(i)) << 32) | uint64_t(uint32_t(j));
}

// Minimum-image displacement for an orthorhombic box.
static inline void min_image(double& dx, double& dy, double& dz, const double L[3])
{
    dx -= std::round(dx / L[0]) * L[0];
    dy -= std::round(dy / L[1]) * L[1];
    dz -= std::round(dz / L[2]) * L[2];
}

// Per-test summary; returned by each section.
struct TestResult {
    double E_total;
    double E_ref;
    double rel_dE;
    double rel_dF;
    double rel_dW;
    bool   pass;
};

// Relative virial error: ‖W − W_ref‖ / ‖W_ref‖ over the 6 Voigt components
// (off-diagonals weighted ×2 so it is the true Frobenius norm of the 3×3 tensor).
static double virial_rel_err(const double W[6], const double W_ref[6])
{
    const double wgt[6] = {1, 1, 1, 2, 2, 2};
    double num = 0.0, den = 0.0;
    for (int c = 0; c < 6; ++c) {
        const double d = W[c] - W_ref[c];
        num += wgt[c] * d * d;
        den += wgt[c] * W_ref[c] * W_ref[c];
    }
    return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// Print the 6 Voigt virial components (computed vs reference) + the rel error.
static void print_virial(const char* label, const double W[6],
                         const double W_ref[6], double rel_dW)
{
    static const char* names[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
    printf("  %s   computed            reference          Δ\n", label);
    for (int c = 0; c < 6; ++c)
        printf("    W_%s = %+15.6e   %+15.6e   %+8.2e\n",
               names[c], W[c], W_ref[c], W[c] - W_ref[c]);
    printf("    virial rel error (Frobenius) = %.3e\n", rel_dW);
}

// Force RMS error vs reference.  Reports the per-atom RMS force-error
// magnitude `dF_rms = sqrt((1/N)·Σ_i |F_i − F_ref_i|²)` and the RMS
// reference-force magnitude `F_rms = sqrt((1/N)·Σ_i |F_ref_i|²)`.  Their
// ratio is the relative L2 / RMS force error (eq. 16), `dF_rms / F_rms`
// (the 1/N factors cancel), used for the PASS/FAIL check.
static void print_force_diff(const std::vector<double>& F,
                             const std::vector<double>& F_ref,
                             int N, double* F_rms_out, double* dF_rms_out)
{
    double sum_dF2 = 0.0, sum_F2 = 0.0;
    for (int i = 0; i < N; ++i) {
        const double dfx = F[3*i  ] - F_ref[3*i  ];
        const double dfy = F[3*i+1] - F_ref[3*i+1];
        const double dfz = F[3*i+2] - F_ref[3*i+2];
        sum_dF2 += dfx*dfx + dfy*dfy + dfz*dfz;
        sum_F2  += F_ref[3*i  ] * F_ref[3*i  ] +
                   F_ref[3*i+1] * F_ref[3*i+1] +
                   F_ref[3*i+2] * F_ref[3*i+2];
    }
    *F_rms_out  = std::sqrt(sum_F2  / N);
    *dF_rms_out = std::sqrt(sum_dF2 / N);
}

// ════════════════════════════════════════════════════════════════════════════
//  Particle-mesh reciprocal (C²-order-3 spread + discrete deconvolution)
// ════════════════════════════════════════════════════════════════════════════
//
// Full double-precision PME reciprocal pipeline driven by the cupo/math/spreading.h C2OSpline<3>
// spreading API — the `--particle-mesh` alternative to the exact O(N·m³)
// structure-factor sum.  Five steps (eq. 8 of Liang et al., arXiv:2505.09727;
// step 3's kernel is their eq. 9):
//
//   (1) spread `charges` onto a Gx·Gy·Gz grid            cupo_spread_1d_f64
//   (2) forward 3-D DFT  ρ → ρ̂                           use cuFFT R2C/C2C
//   (3) multiply ρ̂ by the pre-built deconvolution kernel
//         eterm(m) = pref · K(k²) / |φ̂_3d(m)|²
//       K from cupo_p{1,6}_fourier_kernel_value_f64;
//       |φ̂| from cupo_discrete_fourier_kernel_1d_f64
//   (4) inverse 3-D DFT → potential grid                 use cuFFT C2C/C2R
//   (5) gather forces                                    cupo_interpolate_1d_f64
//
// E_recip = ½ Σ_m eterm(m)·|ρ̂(m)|²;  F_i += −charge_i·(G/L)·Σ ∂W·potential
// (the spread-adjoint), with the order-3 C2 spreading kernel and the discrete
// deconvolution.
//
// The DFT is a naive separable O(3·G⁴) transform: ample for the demo's
// auto-planned grid (water_10k → G≈26).  For large-grid sweeps swap in a real
// FFT (FFTW / pocketfft); the rest of the pipeline is unchanged.

// In-place 1-D DFT of one strided line.  exp(sign·2πi·m·n/G) = W[(m·n) mod G],
// with W[k] = exp(sign·2πi·k/G) precomputed (sign = −1 forward, +1 inverse;
// both unnormalised, matching numpy's fft/ifft·N convention).
static inline void dft_line(std::complex<double>* base, int G, long stride,
                            const std::complex<double>* W,
                            std::complex<double>* tmp)
{
    for (int n = 0; n < G; ++n) tmp[n] = base[static_cast<long>(n) * stride];
    for (int m = 0; m < G; ++m) {
        std::complex<double> acc(0.0, 0.0);
        int mn = 0;                                   // (m·n) mod G, incremental
        for (int n = 0; n < G; ++n) {
            acc += tmp[n] * W[mn];
            mn += m; if (mn >= G) mn -= G;
        }
        base[static_cast<long>(m) * stride] = acc;
    }
}

// Batched 1-D DFT along one axis of the Gx·Gy·Gz complex grid.
static void transform_axis(std::vector<std::complex<double>>& g,
                           int Gx, int Gy, int Gz, int axis, int sign)
{
    const int  Gs[3]      = {Gx, Gy, Gz};
    const long strides[3] = {static_cast<long>(Gy) * Gz, Gz, 1L};
    const int  G      = Gs[axis];
    const long stride = strides[axis];
    std::vector<std::complex<double>> W(G);
    for (int k = 0; k < G; ++k) {
        const double a = sign * kTwoPi * static_cast<double>(k) / G;
        W[k] = std::complex<double>(std::cos(a), std::sin(a));
    }
    const int  d1 = (axis + 1) % 3, d2 = (axis + 2) % 3;
    const int  n1 = Gs[d1],   n2 = Gs[d2];
    const long s1 = strides[d1], s2 = strides[d2];
    #pragma omp parallel
    {
        std::vector<std::complex<double>> tmp(G);
        #pragma omp for collapse(2) schedule(static)
        for (int a1 = 0; a1 < n1; ++a1)
            for (int a2 = 0; a2 < n2; ++a2)
                dft_line(&g[static_cast<long>(a1) * s1 + static_cast<long>(a2) * s2],
                         G, stride, W.data(), tmp.data());
    }
}

static void dft3d(std::vector<std::complex<double>>& g,
                  int Gx, int Gy, int Gz, int sign)
{
    transform_axis(g, Gx, Gy, Gz, 2, sign);
    transform_axis(g, Gx, Gy, Gz, 1, sign);
    transform_axis(g, Gx, Gy, Gz, 0, sign);
}

// Pre-build the discrete 3-D deconvolution kernels: the energy/force influence
// eterm(m) = pref·K(k²)/|φ̂_3d|², and its k²-derivative companion
// eterm'(m) = pref·K'(k²)/|φ̂_3d|² used by the reciprocal virial.  Both are
// position-independent (box/grid/window only), so a real MD loop builds them
// once and reuses them every step.  Everything is f64: |φ̂| is the window's
// sampled-stencil DFT magnitude, K(k²)/K'(k²) the splitting Green's function
// and its dK/dk² (cupo_p{1,6}_fourier_kernel_{value,grad}).
//
// `include_dc` is the k=0 policy, set explicitly by the caller rather than
// inferred from K(0): Coulomb passes false (the singular 1/k² Green's function
// has no DC term in a neutral cell), dispersion passes true (v̂(0) is finite and
// is the dominant contribution).  The DC bin is linear index 0.
// X/Y use FFT order at lengths Kx/Ky; Z retains the first Kz bins of the Gz FFT.
// Use Kz <= Gz/2+1 for R2C storage, or Kx=Gx, Ky=Gy, Kz=Gz for full C2C storage.
// A positive bandwidth explicitly clips both tables at angular |k|*rc > z_cut.
// Zero retains the full scalar transform. The gradient describes a fixed mode
// set; a sharp cutoff is discontinuous when box changes move modes across it.
template <double (*FK_VALUE)(double, double), double (*FK_GRAD)(double, double)>
static void build_deconvolution_kernel(
    const Reference& ref,
    double eterm_pref, bool include_dc, int Gx, int Gy, int Gz, int Kx, int Ky, int Kz,
    std::vector<double>& kernel, std::vector<double>& kernel_grad,
    double bandwidth = 0.0)
{
    if (Kx <= 0 || Kx > Gx || Ky <= 0 || Ky > Gy || Kz <= 0 || Kz > Gz)
        throw std::invalid_argument("Influence dimensions require 0 < Kx <= Gx, 0 < Ky <= Gy, 0 < Kz <= Gz");
    const double rc = ref.cutoff;
    const double inv_Lx = 1.0 / ref.box[0];
    const double inv_Ly = 1.0 / ref.box[1];
    const double inv_Lz = 1.0 / ref.box[2];

    // Per-axis |φ̂(k)| via the discrete Fourier kernel (periodic in k mod G, so
    // the raw 0..G-1 index is fine — see cupo/math/spreading.h).
    std::vector<double> phx(Kx), phy(Ky), phz(Kz);
    for (int m = 0; m < Kx; ++m)
        phx[m] = cupo_discrete_fourier_kernel_1d_f64(m < (Kx+1)/2 ? m : m-Kx, Gx);
    for (int m = 0; m < Ky; ++m)
        phy[m] = cupo_discrete_fourier_kernel_1d_f64(m < (Ky+1)/2 ? m : m-Ky, Gy);
    for (int m = 0; m < Kz; ++m) phz[m] = cupo_discrete_fourier_kernel_1d_f64(m, Gz);

    const size_t total = static_cast<size_t>(Kx) * Ky * Kz;
    kernel.resize(total);
    kernel_grad.resize(total);
    #pragma omp parallel for schedule(static)
    for (int mx = 0; mx < Kx; ++mx) {
        const int    sx  = (mx < (Kx + 1) / 2) ? mx : mx - Kx;   // signed freq
        const double kx  = sx * inv_Lx;
        const double ax2 = phx[mx] * phx[mx];
        for (int my = 0; my < Ky; ++my) {
            const int    sy   = (my < (Ky + 1) / 2) ? my : my - Ky;
            const double ky   = sy * inv_Ly;
            const double axy2 = ax2 * phy[my] * phy[my];
            for (int mz = 0; mz < Kz; ++mz) {
                const int    sz = (mz < (Gz + 1) / 2) ? mz : mz - Gz;
                const double kz = sz * inv_Lz;
                const double k2 = kx * kx + ky * ky + kz * kz;
                const double phi2  = axy2 * phz[mz] * phz[mz];     // |φ̂_3d|²
                const double decon = (phi2 > 0.0) ? (1.0 / phi2) : 0.0;
                const size_t idx = (static_cast<size_t>(mx) * Ky + my) * Kz + mz;
                // The packed rectangular support encloses a sphere. Its
                // corners must be masked explicitly now that FK stays full.
                if (bandwidth > 0.0 && outside_fourier_band(k2, rc, bandwidth)) {
                    kernel[idx] = kernel_grad[idx] = 0.0;
                    continue;
                }
                kernel[idx]      = eterm_pref * FK_VALUE(k2, rc) * decon;
                kernel_grad[idx] = eterm_pref * FK_GRAD(k2, rc)  * decon;
            }
        }
    }

    // k=0 policy, applied explicitly (never inferred from K(0)).  The DC bin is
    // linear index 0: zero it for Coulomb, keep the computed value for dispersion.
    if (!include_dc) {
        kernel[0]      = 0.0;
        kernel_grad[0] = 0.0;
    }
}

// Fractional in-cell offset + integer base grid index for atom coordinate `x`
// on a length-L box mapped to G points (wrapped into [0, G)).
static inline void mesh_base(double x, double L, int G, int& base, double& theta)
{
    double f = x / L; f -= std::floor(f);          // fractional, [0, 1)
    const double t = f * G;
    base  = static_cast<int>(std::floor(t));
    theta = t - base;
}

static void check_cuda(cudaError_t status)
{
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void check_cupo(cupo_status result)
{
    if (result != CUPO_SUCCESS)
        throw std::runtime_error("cupo call failed: cupo_status " +
            std::to_string(static_cast<int>(result)));
}

template<class T> struct DeviceGrid {
    T* data = nullptr;
    explicit DeviceGrid(size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceGrid() { cudaFree(data); }
    DeviceGrid(const DeviceGrid&) = delete;
    DeviceGrid& operator=(const DeviceGrid&) = delete;
};

struct CompactExecution {
    cudaStream_t stream = nullptr;
    cupo_convolution_plan plan = nullptr;
    CompactExecution() { check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~CompactExecution() {
        cudaStreamSynchronize(stream);
        cupo_convolution_plan_destroy(plan);
        cudaStreamDestroy(stream);
    }
    CompactExecution(const CompactExecution&) = delete;
    CompactExecution& operator=(const CompactExecution&) = delete;
};

// Replace the forward DFT, influence multiply and inverse DFT with the compact
// primitive. The caller retains spreading, interpolation and observables.
template <double (*FK_VALUE)(double, double), double (*FK_GRAD)(double, double)>
static double compact_mesh_convolution(
    const Reference& ref, double eterm_pref, bool include_dc,
    int Gx, int Gy, int Gz, std::vector<std::complex<double>>& grid, double W[6],
    double bandwidth)
{
    const int sx = compact_support_radius(ref.box[0], ref.cutoff, Gx, bandwidth);
    const int sy = compact_support_radius(ref.box[1], ref.cutoff, Gy, bandwidth);
    const int sz = compact_support_radius(ref.box[2], ref.cutoff, Gz, bandwidth);

    // Retain signed X/Y and nonnegative Z. Full even axes include Nyquist once.
    const int K[3] = {std::min(Gx, 2*sx+1), std::min(Gy, 2*sy+1), sz+1};
    const size_t spectrum_elements = size_t(K[0]) * K[1] * K[2];
    printf("  compact   : real-space grid=%d×%d×%d, reciprocal grid=%d×%d×%d\n",
           Gx, Gy, Gz, K[0], K[1], K[2]);
    printf("              support radii=(%d,%d,%d), angular |k|*rc <= %.12g, FP32 grids/FFTs\n",
           sx, sy, sz, bandwidth);

    // Build only the retained influence bins, in DC/positive/negative FFT order.
    // Clip at the caller level; the public scalar Fourier functions stay full.
    std::vector<double> kernel, kernel_grad;
    build_deconvolution_kernel<FK_VALUE, FK_GRAD>(
        ref, eterm_pref, include_dc, Gx, Gy, Gz, K[0], K[1], K[2], kernel, kernel_grad,
        bandwidth);
    std::vector<float> input(grid.size()), potential(grid.size());
    std::vector<float> influence(kernel.begin(), kernel.end());
    std::vector<float2> forward(spectrum_elements);
    for (size_t i = 0; i < grid.size(); ++i) input[i] = float(grid[i].real());

    DeviceGrid<float> d_input(input.size()), d_potential(potential.size()), d_h(influence.size());
    DeviceGrid<float2> d_forward(forward.size());
    CompactExecution execution; // Destroy/synchronize before freeing the device buffers.
    // H and exported S both use the exact packed shape K, with Z fastest.
    check_cupo(cupo_convolution_plan_create(Gx, Gy, Gz, sx, sy, sz,
                                         {CUPO_GRID_FP32, 0}, {CUPO_GRID_FP32, 0}, &execution.plan));
    check_cuda(cudaMemcpyAsync(d_input.data, input.data(), input.size()*sizeof(float),
                               cudaMemcpyHostToDevice, execution.stream));
    check_cuda(cudaMemcpyAsync(d_h.data, influence.data(), influence.size()*sizeof(float),
                               cudaMemcpyHostToDevice, execution.stream));

    // H = (physical coupling / box volume) * K / |phi_hat|². Both transforms
    // are unnormalized, as in the host spectral sum: there is NO extra 1/(GxGyGz).
    // Request potential for interpolation and raw S=DFT(input) for E/W below.
    check_cupo(cupo_convolution_execute(execution.plan, d_potential.data, d_forward.data,
                                         d_input.data, d_h.data, execution.stream));
    check_cuda(cudaMemcpyAsync(potential.data(), d_potential.data, potential.size()*sizeof(float),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaMemcpyAsync(forward.data(), d_forward.data, forward.size()*sizeof(float2),
                               cudaMemcpyDeviceToHost, execution.stream));
    check_cuda(cudaStreamSynchronize(execution.stream));
    // In an MD loop, keep the plan, buffers and H alive and reuse them each step.
    for (size_t i = 0; i < grid.size(); ++i) grid[i] = {potential[i], 0.0};

    // Energy/virial remain application code. Interior Z bins represent two
    // conjugate modes; DC and even-grid Z Nyquist represent one. Enumerate both
    // partners explicitly so off-diagonal virials also handle X/Y Nyquist.
    const auto signed_mode = [](int i, int n) { return i < (n+1)/2 ? i : i-n; };
    const auto conjugate_mode = [](int m, int n) { return (n%2 == 0 && m == -n/2) ? m : -m; };
    std::fill(W, W+6, 0.0);
    double energy = 0.0;
    for (int x = 0; x < K[0]; ++x) for (int y = 0; y < K[1]; ++y) for (int z = 0; z < K[2]; ++z) {
        const size_t i = (size_t(x)*K[1]+y)*K[2]+z;
        const double rho2 = double(forward[i].x)*forward[i].x + double(forward[i].y)*forward[i].y;
        const bool nyquist_z = Gz%2 == 0 && z == Gz/2;
        const int modes[3] = {signed_mode(x, K[0]), signed_mode(y, K[1]), nyquist_z ? -z : z};
        const int count = z == 0 || nyquist_z ? 1 : 2;
        for (int partner = 0; partner < count; ++partner) {
            const double kx = (partner ? conjugate_mode(modes[0], Gx) : modes[0]) / ref.box[0];
            const double ky = (partner ? conjugate_mode(modes[1], Gy) : modes[1]) / ref.box[1];
            const double kz = (partner ? -modes[2] : modes[2]) / ref.box[2];
            const double c = 0.5*rho2, ke = influence[i], kg = kernel_grad[i];
            energy += c*ke;
            W[0] += c*(ke + 2*kx*kx*kg); W[1] += c*(ke + 2*ky*ky*kg); W[2] += c*(ke + 2*kz*kz*kg);
            W[3] += c*2*kx*ky*kg; W[4] += c*2*kx*kz*kg; W[5] += c*2*ky*kz*kg;
        }
    }
    return energy;
}
// Run the particle-mesh reciprocal: returns E_recip, accumulates forces into F,
// and writes the reciprocal virial (Voigt) into Wrecip.
template <double (*FK_VALUE)(double, double), double (*FK_GRAD)(double, double)>
static double mesh_reciprocal(const Reference& ref,
                             const std::vector<double>& charges,
                             double eterm_pref, bool include_dc,
                             int Gx, int Gy, int Gz,
                             std::vector<double>& F, double Wrecip[6], bool use_compact = false,
                             double compact_bandwidth = 0.0)
{
    const int N = static_cast<int>(ref.n_atoms);
    const double Lx = ref.box[0], Ly = ref.box[1], Lz = ref.box[2];
    const double inv_Lx = 1.0 / Lx, inv_Ly = 1.0 / Ly, inv_Lz = 1.0 / Lz;
    constexpr int off0 = -1, P = 4;                 // C2OSpline<3>: 4-point, offset −1

    // (1) spread charges onto the grid (single-threaded: N·64 deposits).
    std::vector<std::complex<double>> grid(static_cast<size_t>(Gx) * Gy * Gz,
                                           std::complex<double>(0.0, 0.0));
    for (int i = 0; i < N; ++i) {
        const double q = charges[i];
        if (q == 0.0) continue;
        int bx, by, bz; double thx, thy, thz;
        mesh_base(ref.coords[3*i  ], Lx, Gx, bx, thx);
        mesh_base(ref.coords[3*i+1], Ly, Gy, by, thy);
        mesh_base(ref.coords[3*i+2], Lz, Gz, bz, thz);
        double wx[P], wy[P], wz[P];
        cupo_spread_1d_f64(thx, wx);
        cupo_spread_1d_f64(thy, wy);
        cupo_spread_1d_f64(thz, wz);
        for (int kx = 0; kx < P; ++kx) {
            const int ix = ((bx + off0 + kx) % Gx + Gx) % Gx;
            for (int ky = 0; ky < P; ++ky) {
                const int iy = ((by + off0 + ky) % Gy + Gy) % Gy;
                const double wxy = wx[kx] * wy[ky] * q;
                for (int kz = 0; kz < P; ++kz) {
                    const int iz = ((bz + off0 + kz) % Gz + Gz) % Gz;
                    grid[(static_cast<size_t>(ix) * Gy + iy) * Gz + iz] += wxy * wz[kz];
                }
            }
        }
    }

    double E = 0.0;
    if (use_compact) {
        E = compact_mesh_convolution<FK_VALUE, FK_GRAD>(
            ref, eterm_pref, include_dc, Gx, Gy, Gz, grid, Wrecip, compact_bandwidth);
    } else {
        std::vector<double> kernel, kernel_grad;
        build_deconvolution_kernel<FK_VALUE, FK_GRAD>(
            ref, eterm_pref, include_dc, Gx, Gy, Gz, Gx, Gy, Gz, kernel, kernel_grad);

        // (2) forward DFT - ideally run with cuFFT
        // (C2C): Gx, Gy, Gz -> Gx, Gy, Gz
        // (R2C): Gx, Gy, Gz -> Gx, Gy, Gz/2
        dft3d(grid, Gx, Gy, Gz, -1);
        double Wxx = 0, Wyy = 0, Wzz = 0, Wxy = 0, Wxz = 0, Wyz = 0;

        // (3) apply kernel + accumulate energy + virial.
        //   E   = ½ Σ_m eterm·|ρ̂|²
        //   W_ab= ½ Σ_m |ρ̂|²·[δ_ab·eterm + 2 k_a k_b·eterm']   (Voigt)
        #pragma omp parallel for reduction(+:E,Wxx,Wyy,Wzz,Wxy,Wxz,Wyz) schedule(static)
        for (int mx = 0; mx < Gx; ++mx) {
            const int    sx = (mx < (Gx + 1) / 2) ? mx : mx - Gx;
            const double kx = sx * inv_Lx;
            for (int my = 0; my < Gy; ++my) {
                const int    sy = (my < (Gy + 1) / 2) ? my : my - Gy;
                const double ky = sy * inv_Ly;
                for (int mz = 0; mz < Gz; ++mz) {
                    const int    sz = (mz < (Gz + 1) / 2) ? mz : mz - Gz;
                    const double kz = sz * inv_Lz;
                    const size_t idx = (static_cast<size_t>(mx) * Gy + my) * Gz + mz;
                    const double rho2 = std::norm(grid[idx]);
                    const double ke = kernel[idx], kg = kernel_grad[idx];
                    E += ke * rho2;
                    const double c = 0.5 * rho2;
                    Wxx += c * (ke + 2.0 * kx * kx * kg);
                    Wyy += c * (ke + 2.0 * ky * ky * kg);
                    Wzz += c * (ke + 2.0 * kz * kz * kg);
                    Wxy += c * (2.0 * kx * ky * kg);
                    Wxz += c * (2.0 * kx * kz * kg);
                    Wyz += c * (2.0 * ky * kz * kg);
                    grid[idx] *= ke;                    // → potential for force gather
                }
            }
        }
        E *= 0.5;
        Wrecip[0] = Wxx; Wrecip[1] = Wyy; Wrecip[2] = Wzz;
        Wrecip[3] = Wxy; Wrecip[4] = Wxz; Wrecip[5] = Wyz;

        // (4) inverse DFT → potential grid (real part).
        dft3d(grid, Gx, Gy, Gz, +1);
    }

    // (5) gather forces: F_i_α += −charge_i·(Gα/Lα)·Σ ∂_αW·potential.
    const double cGx = static_cast<double>(Gx) / Lx;
    const double cGy = static_cast<double>(Gy) / Ly;
    const double cGz = static_cast<double>(Gz) / Lz;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const double q = charges[i];
        if (q == 0.0) continue;
        int bx, by, bz; double thx, thy, thz;
        mesh_base(ref.coords[3*i  ], Lx, Gx, bx, thx);
        mesh_base(ref.coords[3*i+1], Ly, Gy, by, thy);
        mesh_base(ref.coords[3*i+2], Lz, Gz, bz, thz);
        double wx[P], wy[P], wz[P], dwx[P], dwy[P], dwz[P];
        cupo_spread_1d_f64(thx, wx);  cupo_interpolate_1d_f64(thx, dwx);
        cupo_spread_1d_f64(thy, wy);  cupo_interpolate_1d_f64(thy, dwy);
        cupo_spread_1d_f64(thz, wz);  cupo_interpolate_1d_f64(thz, dwz);
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int kx = 0; kx < P; ++kx) {
            const int ix = ((bx + off0 + kx) % Gx + Gx) % Gx;
            for (int ky = 0; ky < P; ++ky) {
                const int iy = ((by + off0 + ky) % Gy + Gy) % Gy;
                for (int kz = 0; kz < P; ++kz) {
                    const int iz = ((bz + off0 + kz) % Gz + Gz) % Gz;
                    const double pot =
                        grid[(static_cast<size_t>(ix) * Gy + iy) * Gz + iz].real();
                    ax += dwx[kx] * wy[ky]  * wz[kz]  * pot;
                    ay += wx[kx]  * dwy[ky] * wz[kz]  * pot;
                    az += wx[kx]  * wy[ky]  * dwz[kz] * pot;
                }
            }
        }
        F[3*i  ] += -q * cGx * ax;
        F[3*i+1] += -q * cGy * ay;
        F[3*i+2] += -q * cGz * az;
    }
    return E;
}

// ════════════════════════════════════════════════════════════════════════════
//  Shared stage helpers (used by both run_coulomb and run_dispersion)
// ════════════════════════════════════════════════════════════════════════════

// Exact O(N·m³) structure-factor reciprocal sum — the use_mesh==false twin of
// mesh_reciprocal above, with the same contract: returns E_recip, accumulates
// forces into F, and WRITES the reciprocal virial (Voigt) into Wrecip.
//
// Physics inputs only: per-atom weights w (charges or √B), the splitting Green's
// function value/grad (cupo_p{1,6}_fourier_kernel_*), the reciprocal prefactor
// recip_pref = coupling/V (= coulomb_constant/V for Coulomb, −1/V for dispersion
// — the same quantity mesh_reciprocal takes as eterm_pref), and the k=0 policy
// include_dc (Coulomb skips the DC bin; dispersion keeps the finite, dominant
// v̂(0)).  The ½ structure-factor double-counting factor is applied internally,
// exactly as mesh_reciprocal halves its energy.
template <double (*FK_VALUE)(double, double), double (*FK_GRAD)(double, double)>
static double exact_reciprocal(const Reference& ref,
                               const std::vector<double>& w,
                               double recip_pref, bool include_dc,
                               std::vector<double>& F, double Wrecip[6])
{
    const int    N  = static_cast<int>(ref.n_atoms);
    const double rc = ref.cutoff;
    const int M_PER_DIM = recip_m_per_dim(ref.box, rc);
    const double inv_Lx = 1.0 / ref.box[0];
    const double inv_Ly = 1.0 / ref.box[1];
    const double inv_Lz = 1.0 / ref.box[2];
    double E_recip = 0.0;
    double Wxx = 0, Wyy = 0, Wzz = 0, Wxy = 0, Wxz = 0, Wyz = 0;
    for (int mz = -M_PER_DIM; mz <= M_PER_DIM; ++mz)
    for (int my = -M_PER_DIM; my <= M_PER_DIM; ++my)
    for (int mx = -M_PER_DIM; mx <= M_PER_DIM; ++mx) {
        const bool is_dc = (mx == 0 && my == 0 && mz == 0);
        // k=0 policy, explicit (never inferred from the kernel returning 0):
        // drop the DC bin entirely unless include_dc.
        if (!include_dc && is_dc) continue;
        const double kx = mx * inv_Lx;
        const double ky = my * inv_Ly;
        const double kz = mz * inv_Lz;
        const double k2 = kx*kx + ky*ky + kz*kz;
        const double ker = FK_VALUE(k2, rc);

        double Csum = 0.0, Dsum = 0.0;
        #pragma omp parallel for reduction(+:Csum, Dsum) schedule(static)
        for (int j = 0; j < N; ++j) {
            const double arg = kTwoPi * (kx * ref.coords[3*j    ]
                                       + ky * ref.coords[3*j + 1]
                                       + kz * ref.coords[3*j + 2]);
            Csum += w[j] * std::cos(arg);
            Dsum += w[j] * std::sin(arg);
        }
        const double S2 = Csum * Csum + Dsum * Dsum;
        // E += ½ · recip_pref · ker · |S|²
        const double e_k = 0.5 * recip_pref * ker * S2;
        E_recip += e_k;

        // virial: W_αβ = Σ_k [δ_αβ·e_k + recip_pref·K'(k²)·|S|²·k_α k_β]
        const double dker = FK_GRAD(k2, rc);
        const double w_coef = recip_pref * dker * S2;
        Wxx += e_k + w_coef*kx*kx; Wyy += e_k + w_coef*ky*ky;
        Wzz += e_k + w_coef*kz*kz;
        Wxy += w_coef*kx*ky; Wxz += w_coef*kx*kz; Wyz += w_coef*ky*kz;

        // F_i_α = 2π · recip_pref · w_i · ker · k_α · (C sin − D cos).  Atoms
        // with w_j = 0 contribute term = 0, so no per-atom special-case is needed.
        const double f_coef = kTwoPi * recip_pref * ker;
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; ++j) {
            const double arg = kTwoPi * (kx * ref.coords[3*j    ]
                                       + ky * ref.coords[3*j + 1]
                                       + kz * ref.coords[3*j + 2]);
            const double term = w[j] *
                (Csum * std::sin(arg) - Dsum * std::cos(arg));
            F[3*j    ] += f_coef * kx * term;
            F[3*j + 1] += f_coef * ky * term;
            F[3*j + 2] += f_coef * kz * term;
        }
    }
    Wrecip[0] = Wxx; Wrecip[1] = Wyy; Wrecip[2] = Wzz;
    Wrecip[3] = Wxy; Wrecip[4] = Wxz; Wrecip[5] = Wyz;
    return E_recip;
}

// Short-range O(N²) min-image pair sum, shared by both pipelines.  The per-pair
// math is identical once the sign is folded into a signed coupling:
//   coef = coef_scale · w_i · w_j   (coef_scale = coulomb_constant for Coulomb;
//                                    −1 for the attractive 1/r⁶ dispersion)
//   E += coef·value(r²);  fr = −2·coef·grad(r²);
//   F_i += fr·dr;  F_j −= fr·dr;  W += fr·(dr⊗dr).
// The value/grad kernels are template parameters so they inline (no function-
// pointer indirection in this hot loop).  Zero-weight pairs (coef = 0, e.g.
// dispersion pairs involving an ε = 0 hydrogen) contribute nothing and are skipped.
template <double (*SR_VALUE)(double, double), double (*SR_GRAD)(double, double)>
static double pairwise_short_range(const Reference& ref,
                                   const std::vector<double>& w,
                                   double coef_scale,
                                   const std::unordered_set<uint64_t>& excl_set,
                                   std::vector<double>& F, double W[6])
{
    const int    N   = static_cast<int>(ref.n_atoms);
    const double rc  = ref.cutoff;
    const double rc2 = rc * rc;
    double E_short = 0.0;
    double Wxx = 0, Wyy = 0, Wzz = 0, Wxy = 0, Wxz = 0, Wyz = 0;
    #pragma omp parallel for schedule(dynamic, 64) \
        reduction(+:E_short,Wxx,Wyy,Wzz,Wxy,Wxz,Wyz)
    for (int i = 0; i < N; ++i) {
        const double xi = ref.coords[3*i  ];
        const double yi = ref.coords[3*i+1];
        const double zi = ref.coords[3*i+2];
        const double wi  = w[i];
        double Fix = 0.0, Fiy = 0.0, Fiz = 0.0;
        for (int j = i + 1; j < N; ++j) {
            if (excl_set.count(pack_pair(i, j))) continue;
            const double coef = coef_scale * wi * w[j];   // signed pair coupling
            if (coef == 0.0) continue;                    // zero-weight pair
            double dx = xi - ref.coords[3*j  ];
            double dy = yi - ref.coords[3*j+1];
            double dz = zi - ref.coords[3*j+2];
            min_image(dx, dy, dz, ref.box);
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= rc2) continue;
            const double v = SR_VALUE(r2, rc);
            const double g = SR_GRAD(r2, rc);
            E_short += coef * v;
            // grad is dS/d(r²); force_over_r = −2·coef·grad.
            const double fr = -coef * 2.0 * g;
            Fix += fr * dx;  Fiy += fr * dy;  Fiz += fr * dz;
            Wxx += fr*dx*dx; Wyy += fr*dy*dy; Wzz += fr*dz*dz;
            Wxy += fr*dx*dy; Wxz += fr*dx*dz; Wyz += fr*dy*dz;
            #pragma omp atomic
            F[3*j  ] -= fr * dx;
            #pragma omp atomic
            F[3*j+1] -= fr * dy;
            #pragma omp atomic
            F[3*j+2] -= fr * dz;
        }
        #pragma omp atomic
        F[3*i  ] += Fix;
        #pragma omp atomic
        F[3*i+1] += Fiy;
        #pragma omp atomic
        F[3*i+2] += Fiz;
    }
    W[0]+=Wxx; W[1]+=Wyy; W[2]+=Wzz; W[3]+=Wxy; W[4]+=Wxz; W[5]+=Wyz;
    return E_short;
}

// Self-correction energy: self_pref · self_value(rc) · Σ_i w_i².  Strain-
// independent, so it contributes nothing to the virial.  (Coulomb: self_pref =
// coulomb_constant and cupo_p1_self_value is a negative correction; dispersion:
// self_pref = +1 and cupo_p6_self_value is the positive bare coefficient.)
template <double (*SELF_VALUE)(double)>
static double self_correction(const std::vector<double>& w,
                              double self_pref, double rc)
{
    const int N = static_cast<int>(w.size());
    double w2_sum = 0.0;
    for (int i = 0; i < N; ++i) w2_sum += w[i] * w[i];
    return self_pref * SELF_VALUE(rc) * w2_sum;
}

// Net-charge plasma correction for k=0:  coulomb · 0.5 · G0 · (Σqᵢ)² / V.
// This is only for electrostatics (1/r) - dispersion retains the full k=0 band.
static double net_charge_correction(const std::vector<double>& q,
                                    const double box[3], double coulomb, double rc)
{
    double sum_q = 0.0;
    for (double qi : q) sum_q += qi;
    const double volume = box[0] * box[1] * box[2];
    return coulomb * 0.5 * cupo_p1_fourier_kernel_dc_f64(rc) * sum_q * sum_q / volume;
}

// Excluded-pair long-range subtraction, shared by both pipelines.  Each excluded
// (1-2 / 1-3) pair had its full short+long interaction removed by the exclusion
// list, so here we add back the LONG-range part with the same signed coupling as
// the short-range loop (coef = coef_scale·w_i·w_j) but applied with the OPPOSITE
// force/virial sign, since the total energy carries −E_excl.  Returns the
// un-negated E_excl; the caller subtracts it.
//
// Unlike the short-range kernels (which are singular at r=0), the long-range
// kernels are the smooth complement with nonzero Fourier tails. They are finite
// at small r, which matters because excluded pairs can be very close (bonded).
// So zero-weight pairs contribute coef·v = 0; no per-pair short-circuit is needed.
template <double (*LR_VALUE)(double, double), double (*LR_GRAD)(double, double)>
static double pairwise_excluded_lr(const Reference& ref,
                                   const std::vector<double>& w,
                                   double coef_scale,
                                   std::vector<double>& F, double W[6])
{
    const int    M  = static_cast<int>(ref.n_excl);
    const double rc = ref.cutoff;
    double E_excl = 0.0;
    for (int k = 0; k < M; ++k) {
        const int i = ref.excl[2*k    ];
        const int j = ref.excl[2*k + 1];
        double dx = ref.coords[3*i  ] - ref.coords[3*j  ];
        double dy = ref.coords[3*i+1] - ref.coords[3*j+1];
        double dz = ref.coords[3*i+2] - ref.coords[3*j+2];
        min_image(dx, dy, dz, ref.box);
        const double r2 = dx*dx + dy*dy + dz*dz;
        const double coef = coef_scale * w[i] * w[j];
        const double lr_v = LR_VALUE(r2, rc);
        const double lr_g = LR_GRAD(r2, rc);
        E_excl += coef * lr_v;
        // grad is dS/d(r²); force_over_r = −2·coef·grad, applied with the opposite
        // sign of the short-range loop (E_total carries −E_excl).
        const double fr = -coef * 2.0 * lr_g;
        F[3*i  ] -= fr * dx;  F[3*i+1] -= fr * dy;  F[3*i+2] -= fr * dz;
        F[3*j  ] += fr * dx;  F[3*j+1] += fr * dy;  F[3*j+2] += fr * dz;
        const double w_fr = -fr;
        W[0] += w_fr*dx*dx; W[1] += w_fr*dy*dy; W[2] += w_fr*dz*dz;
        W[3] += w_fr*dx*dy; W[4] += w_fr*dx*dz; W[5] += w_fr*dy*dz;
    }
    return E_excl;
}

// ════════════════════════════════════════════════════════════════════════════
//  Coulomb (prolate P=1)
// ════════════════════════════════════════════════════════════════════════════

static TestResult run_coulomb(const Reference& ref,
                              const std::unordered_set<uint64_t>& excl_set,
                              double kETol, double kFTol, double kWTol,
                              bool use_mesh, const int G_mesh[3], bool use_compact)
{
    const int    N  = static_cast<int>(ref.n_atoms);
    const double rc = ref.cutoff;
    const double V  = ref.box[0] * ref.box[1] * ref.box[2];
    const double coulomb_constant = ref.coulomb_constant;

    printf("\n══════════ Coulomb (prolate P=1) ══════════\n");
    printf("  Ref E = %.6f kJ/mol\n", ref.ref_E_coul);

    std::vector<double> F(3 * N, 0.0);
    // Virial tensor W = −∂E/∂ε = Σ r⊗F (Voigt: xx,yy,zz,xy,xz,yz), accumulated
    // alongside the energy/force from each contribution (short + recip + excl;
    // self is strain-independent ⇒ 0).
    double W[6] = {0, 0, 0, 0, 0, 0};

    // ── (1) Short-range: O(N²) min-image pair loop ──────────────────────────
    auto t0 = std::chrono::steady_clock::now();
    const double E_short =
        pairwise_short_range<cupo_p1_short_range_value_f64,
                             cupo_p1_short_range_grad_f64>(
            ref, ref.charges, coulomb_constant, excl_set, F, W);
    double t_short = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  E_short    = %15.6f kJ/mol      [%6.2fs]\n", E_short, t_short);

    // ── (2) Reciprocal: particle-mesh OR exact structure-factor sum ─────────
    double E_recip = 0.0;
    double Wr[6];
    if (use_mesh) {
        printf("  recip      : particle-mesh G = %d×%d×%d (C²-order-3 spread, "
               "discrete decon)\n", G_mesh[0], G_mesh[1], G_mesh[2]);
        t0 = std::chrono::steady_clock::now();
        E_recip = mesh_reciprocal<cupo_p1_fourier_kernel_value_f64,
                                  cupo_p1_fourier_kernel_grad_f64>(
                      ref, ref.charges, coulomb_constant / V, /*include_dc=*/false,
                      G_mesh[0], G_mesh[1], G_mesh[2], F, Wr, use_compact,
                      kP1CompactBandwidth);
    } else {
        const int M_PER_DIM = recip_m_per_dim(ref.box, rc);
        printf("  recip grid : m ∈ [-%d,%d]³\n", M_PER_DIM, M_PER_DIM);
        t0 = std::chrono::steady_clock::now();
        E_recip = exact_reciprocal<cupo_p1_fourier_kernel_value_f64,
                                   cupo_p1_fourier_kernel_grad_f64>(
                      ref, ref.charges, coulomb_constant / V, /*include_dc=*/false,
                      F, Wr);
    }
    for (int c = 0; c < 6; ++c) W[c] += Wr[c];
    double t_recip = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  E_recip    = %15.6f kJ/mol      [%6.2fs]\n", E_recip, t_recip);

    // ── (3) Self-correction ─────────────────────────────────────────────────
    const double E_self =
        self_correction<cupo_p1_self_value_f64>(ref.charges, coulomb_constant, rc);
    printf("  E_self     = %15.6f kJ/mol\n", E_self);

    // ── (4) Excluded long-range subtraction ────────────────────────────────
    const double E_excl =
        pairwise_excluded_lr<cupo_p1_long_range_value_f64,
                             cupo_p1_long_range_grad_f64>(
            ref, ref.charges, coulomb_constant, F, W);
    printf("  E_excl_LR  = %15.6f kJ/mol\n", E_excl);

    // ── (5) Net-charge background ──────────────────────────────────────────
    const double E_plasma =
        net_charge_correction(ref.charges, ref.box, coulomb_constant, rc);
    printf("  E_plasma   = %15.6f kJ/mol   (0 unless the cell is charged)\n", E_plasma);

    // ── Compare ─────────────────────────────────────────────────────────────
    const double E_total = E_short + E_recip + E_self - E_excl + E_plasma;
    double F_rms = 0.0, dF_rms = 0.0;
    print_force_diff(F, ref.ref_F_coul, N, &F_rms, &dF_rms);
    const double dE = E_total - ref.ref_E_coul;
    const double rel_dE = std::abs(dE) / std::abs(ref.ref_E_coul);
    const double rel_dF = dF_rms / F_rms;
    const double rel_dW = virial_rel_err(W, ref.ref_W_coul);
    printf("  E_total    = %15.6f      ΔE = %+8.3e kJ/mol  (rel: %.3e)\n",
           E_total, dE, rel_dE);
    printf("  F RMS error = %14.3e      (F_rms = %.3e, rel RMS: %.3e)\n",
           dF_rms, F_rms, rel_dF);
    print_virial("W (Voigt)", W, ref.ref_W_coul, rel_dW);
    const bool ok = (rel_dE < kETol) && (rel_dF < kFTol) && (rel_dW < kWTol);
    printf("  %s  (tol: E rel < %.0e, F rel < %.0e, W rel < %.0e)\n",
           ok ? "PASS" : "FAIL", kETol, kFTol, kWTol);
    return {E_total, ref.ref_E_coul, rel_dE, rel_dF, rel_dW, ok};
}

// ════════════════════════════════════════════════════════════════════════════
//  Dispersion PME (prolate P=6, attractive 1/r⁶, C6-only)
// ════════════════════════════════════════════════════════════════════════════

static TestResult run_dispersion(const Reference& ref,
                                 const std::unordered_set<uint64_t>& excl_set,
                                 double kETol, double kFTol, double kWTol,
                                 bool use_mesh, const int G_mesh[3], bool use_compact)
{
    const int    N  = static_cast<int>(ref.n_atoms);
    const double rc = ref.cutoff;
    const double V  = ref.box[0] * ref.box[1] * ref.box[2];

    printf("\n══════════ Dispersion (prolate P=6, C6-only) ══════════\n");
    printf("  Ref E_disp = %.6f kJ/mol  (C6-only, geometric)\n", ref.ref_E_disp);

    // Per-atom dispersion "charge" √C₆ with geometric combining (C6_ij = √C₆_i·√C₆_j).
    // In water TIP3P-FB only oxygens have ε > 0; hydrogens get √C₆ = 0 automatically.
    //   √C₆_i = 2 √ε_i · σ_i³   (so √C₆_i·√C₆_j = 4 √(ε_i ε_j) σ_i³ σ_j³ = C6_ij)
    std::vector<double> sqrtB(N);
    for (int i = 0; i < N; ++i) {
        const double s3 = ref.sigma[i] * ref.sigma[i] * ref.sigma[i];
        sqrtB[i] = 2.0 * std::sqrt(std::max(ref.epsilon[i], 0.0)) * s3;
    }

    std::vector<double> F(3 * N, 0.0);
    // C6-only dispersion virial (Voigt) — the 1/r⁶ (p6) short + recip + excl parts.
    double W[6] = {0, 0, 0, 0, 0, 0};

    // ── (1) Real-space pass: prolate-P=6 short-range 1/r⁶ (attractive).  Shared
    //        with run_coulomb via pairwise_short_range with the signed coupling
    //        coef = −√C₆ᵢ·√C₆ⱼ (the attractive-dispersion sign).
    auto t0 = std::chrono::steady_clock::now();
    const double E_R6_short =
        pairwise_short_range<cupo_p6_short_range_value_f64,
                             cupo_p6_short_range_grad_f64>(
            ref, sqrtB, -1.0, excl_set, F, W);
    double t_real = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  E_R6_short = %15.6f kJ/mol      [%6.2fs]\n", E_R6_short, t_real);

    // ── (2) Reciprocal: particle-mesh OR exact structure-factor sum (cupo_p6) ─
    double E_R6_recip = 0.0;
    double Wr[6];
    if (use_mesh) {
        // Dispersion mesh: spread √B "charges", deconvolve with the C²-order-3
        // discrete kernel, and apply the P=6 Green's function.  eterm prefactor
        // is −1/V (the attractive-dispersion sign; the LJ-C₆ amplitude already
        // lives in √B_i, so there is no Coulomb constant here).
        printf("  recip      : particle-mesh G = %d×%d×%d (C²-order-3 spread, "
               "discrete decon)\n", G_mesh[0], G_mesh[1], G_mesh[2]);
        t0 = std::chrono::steady_clock::now();
        E_R6_recip = mesh_reciprocal<cupo_p6_fourier_kernel_value_f64,
                                     cupo_p6_fourier_kernel_grad_f64>(
                         ref, sqrtB, -1.0 / V, /*include_dc=*/true,
                         G_mesh[0], G_mesh[1], G_mesh[2], F, Wr, use_compact,
                         kP6CompactBandwidth);
    } else {
        // Exact path: recip_pref = −1/V (the attractive-dispersion sign; no
        // Coulomb constant, since √B_i already carries the dispersion amplitude),
        // include_dc=true so the dominant finite v̂(0) ≈ −5965 kJ/mol is kept.
        const int M_PER_DIM = recip_m_per_dim(ref.box, rc);
        printf("  recip grid : m ∈ [-%d,%d]³\n", M_PER_DIM, M_PER_DIM);
        t0 = std::chrono::steady_clock::now();
        E_R6_recip = exact_reciprocal<cupo_p6_fourier_kernel_value_f64,
                                      cupo_p6_fourier_kernel_grad_f64>(
                         ref, sqrtB, -1.0 / V, /*include_dc=*/true, F, Wr);
    }
    for (int c = 0; c < 6; ++c) W[c] += Wr[c];
    double t_recip = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("  E_R6_recip = %15.6f kJ/mol      [%6.2fs]\n", E_R6_recip, t_recip);

    // ── (3) Self-correction for the 1/r⁶ part ──────────────────────────────
    // self_pref = +1 (no Coulomb constant — √B carries the LJ-C₆ amplitude);
    // cupo_p6_self_value returns the positive bare coefficient (= +a₆/(2·rc⁶)).
    const double E_R6_self =
        self_correction<cupo_p6_self_value_f64>(sqrtB, 1.0, rc);
    printf("  E_R6_self  = %15.6f kJ/mol\n", E_R6_self);

    // ── (4) Excluded long-range subtraction for the 1/r⁶ part ──────────────
    // For water TIP3P-FB this is identically 0 (every excluded pair involves at
    // least one hydrogen, and ε_H = 0 → √B = 0), but the loop is run anyway so
    // the code generalises.  coef_scale = −1 (the attractive-dispersion sign).
    const double E_R6_excl =
        pairwise_excluded_lr<cupo_p6_long_range_value_f64,
                             cupo_p6_long_range_grad_f64>(
            ref, sqrtB, -1.0, F, W);
    printf("  E_R6_excl  = %15.6f kJ/mol\n", E_R6_excl);

    // ── Compare ─────────────────────────────────────────────────────────────
    // The reference is the C6-only dispersion (geometric), so the PASS/FAIL
    // comparison uses the cupo C6 dispersion total E_R6 (= p6 short + recip +
    // self − excl) and its forces/virial.
    const double E_R6 = E_R6_short + E_R6_recip + E_R6_self - E_R6_excl;
    double F_rms = 0.0, dF_rms = 0.0;
    print_force_diff(F, ref.ref_F_disp, N, &F_rms, &dF_rms);
    const double dE = E_R6 - ref.ref_E_disp;
    const double rel_dE = std::abs(dE) / std::abs(ref.ref_E_disp);
    const double rel_dF = dF_rms / F_rms;
    const double rel_dW = virial_rel_err(W, ref.ref_W_disp);
    printf("  E_R6 (C6)  = %15.6f      ΔE = %+8.3e kJ/mol  (rel: %.3e)\n",
           E_R6, dE, rel_dE);
    printf("  F RMS error = %14.3e      (F_rms = %.3e, rel RMS: %.3e)\n",
           dF_rms, F_rms, rel_dF);
    print_virial("W_disp (C6, Voigt)", W, ref.ref_W_disp, rel_dW);
    const bool ok = (rel_dE < kETol) && (rel_dF < kFTol) && (rel_dW < kWTol);
    printf("  %s  (tol: E rel < %.0e, F rel < %.0e, W rel < %.0e)\n",
           ok ? "PASS" : "FAIL", kETol, kFTol, kWTol);
    return {E_R6, ref.ref_E_disp, rel_dE, rel_dF, rel_dW, ok};
}

// ════════════════════════════════════════════════════════════════════════════
//  Driver
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) try
{
    if (cupo_abi_version() != CUPO_ABI_VERSION)
        throw std::runtime_error("cupo ABI mismatch: headers require " +
            std::to_string(CUPO_ABI_VERSION) + ", runtime provides " +
            std::to_string(cupo_abi_version()));

    // CLI: [ref.bin] [--particle-mesh|--compact] [--grid N] [--cutoff NM].
    //   --particle-mesh  compute the reciprocal sum on a PME mesh (C²-order-3
    //                    spread + discrete deconvolution) instead of the exact
    //                    structure-factor sum.
    //   --compact        use the GPU compact cuFFTDx convolution; implies mesh.
    //                    Plan with cupo_plan_grid (CUPO_GRID_COMPACT) and truncate
    //                    at the numerical bands above (P1 z_cut=10.75,
    //                    P6 z_cut~12.764 by default).
    //   --grid N         mesh size per dimension.  Default (N≤0): auto-planned
    //                    a-priori from the box + Prolate plan by cupo_plan_grid
    //                    (cupo/kspace.h) at the supported force target. Pass --grid N to
    //                    override. Direct structure-factor path: unused.
    //   --cutoff NM      splitting cutoff (nm), overriding the blob's.  The
    //                    reference totals are cutoff-independent, so the same
    //                    tolerances hold across the recommended ~0.6–0.9 nm range.
    std::string path = "../../data/water_10k/water_10k_ref.bin";
    bool use_mesh = false;
    bool use_compact = false;
    int  grid_arg = 0;        // >0 ⇒ explicit --grid override; ≤0 ⇒ auto-plan
    double cutoff_arg = 0.0;  // >0 ⇒ override the blob's splitting cutoff (nm)
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--particle-mesh") {
            use_mesh = true;
        } else if (arg == "--compact") {
            use_mesh = use_compact = true;
        } else if (arg == "--grid" && a + 1 < argc) {
            grid_arg = std::atoi(argv[++a]);
        } else if (arg.rfind("--grid=", 0) == 0) {
            grid_arg = std::atoi(arg.c_str() + 7);
        } else if (arg == "--cutoff" && a + 1 < argc) {
            cutoff_arg = std::atof(argv[++a]);
        } else if (arg.rfind("--cutoff=", 0) == 0) {
            cutoff_arg = std::atof(arg.c_str() + 9);
        } else if (!arg.empty() && arg[0] != '-') {
            path = arg;
        } else {
            fprintf(stderr, "WARNING: ignoring unrecognised argument '%s'\n",
                    arg.c_str());
        }
    }
    Reference ref = load_reference(path);

    // Optional splitting-cutoff override.  The reference E/F/W totals are
    // cutoff-independent (the splitting only shifts work between short and long
    // range), so the same tolerances hold across the recommended ~0.6–0.9 nm
    // range.  The cutoff must stay within the minimum image (< box/2).
    if (cutoff_arg > 0.0) {
        const double half = 0.5 * std::min({ref.box[0], ref.box[1], ref.box[2]});
        if (cutoff_arg >= half) {
            fprintf(stderr, "ERROR: --cutoff %.3f nm exceeds box/2 (%.3f nm)\n",
                    cutoff_arg, half);
            return 2;
        }
        ref.cutoff = cutoff_arg;
    }

    // Display name from the blob filename ("mcl1_ref.bin" → "mcl1").
    std::string sysname = path;
    if (auto s = sysname.find_last_of("/\\"); s != std::string::npos)
        sysname = sysname.substr(s + 1);
    if (auto s = sysname.rfind("_ref.bin"); s != std::string::npos)
        sysname = sysname.substr(0, s);

    printf("%s references — %s\n", sysname.c_str(), use_compact
           ? "host spreading/interpolation + GPU compact convolution" : "host-only C++ via the LTO-IR API");
    printf("  N=%u, M=%u, box=(%.4f,%.4f,%.4f) nm, cutoff=%.3f nm\n",
           ref.n_atoms, ref.n_excl,
           ref.box[0], ref.box[1], ref.box[2], ref.cutoff);
    printf("  Coulomb constant = %.6f kJ·nm/(mol·e²)\n", ref.coulomb_constant);

    // Per-axis mesh size: auto-planned from the box + Prolate P1 plan (target
    // 5e-4) unless the user overrode --grid.  cupo_plan_grid sizes each
    // axis independently, so non-cubic boxes are handled correctly.
    int G_mesh[3];
    const bool grid_auto = (grid_arg <= 0);
    if (grid_auto) {
        const auto policy = use_compact ? CUPO_GRID_COMPACT : CUPO_GRID_DENSE;
        cupo_splitting_plan_handle plan = nullptr;
        check_cupo(cupo_splitting_plan_create_prolate(ref.cutoff, ref.coulomb_constant, &plan));
        const cupo_status planned = cupo_plan_grid(plan, ref.box, policy, G_mesh);
        const cupo_status destroyed = cupo_splitting_plan_destroy(plan);
        check_cupo(planned);
        check_cupo(destroyed);
    } else if (grid_arg < 2) {
        fprintf(stderr, "ERROR: --grid must be ≥ 2 (got %d)\n", grid_arg);
        return 2;
    } else {
        // Reject unsupported overrides before allocating/spreading a full grid.
        // This is the public axis contract; plan_create also validates geometry.
        if (use_compact) {
            int remainder = grid_arg;
            for (int p : {2, 3, 5, 7, 11})
                while (remainder % p == 0) remainder /= p;
            if (grid_arg > 512 || remainder != 1)
                throw std::runtime_error("--compact --grid must be 11-smooth in [2,512]");
        }
        G_mesh[0] = G_mesh[1] = G_mesh[2] = grid_arg;
    }

    printf("  reciprocal: %s\n", use_compact ? "compact cuFFTDx (FP32)" : use_mesh
           ? "particle-mesh (C²-order-3 spread + discrete deconvolution)"
           : "exact structure-factor sum");
    if (use_mesh)
        printf("  mesh grid : G = %d×%d×%d  (%s)\n", G_mesh[0], G_mesh[1], G_mesh[2],
               grid_auto ? (use_compact ? "cupo_plan_grid compact, target 5e-4"
                                       : "cupo_plan_grid, target 5e-4") : "--grid override");

    std::unordered_set<uint64_t> excl_set;
    excl_set.reserve(ref.n_excl * 2);
    for (uint32_t k = 0; k < ref.n_excl; ++k) {
        excl_set.insert(pack_pair(ref.excl[2*k], ref.excl[2*k + 1]));
    }

    // The original planner targeted an average 5e-4 force error across systems;
    // these inherited 6e-4 per-channel PASS limits allow case-to-case variation.
    // The current 5e-4 combined-force calibration is a separate criterion.
    auto r_coul = run_coulomb   (ref, excl_set, /*E*/2e-5, /*F*/6e-4, /*W*/1e-3, use_mesh, G_mesh, use_compact);
    auto r_disp = run_dispersion(ref, excl_set, /*E*/7e-5, /*F*/6e-4, /*W*/6e-4, use_mesh, G_mesh, use_compact);

    printf("\n══════════ Summary ══════════\n");
    printf("  Coulomb: rel_ΔE = %.3e   rel_ΔF_rms = %.3e   rel_ΔW = %.3e   %s\n",
           r_coul.rel_dE, r_coul.rel_dF, r_coul.rel_dW, r_coul.pass ? "PASS" : "FAIL");
    printf("  Disp(C6): rel_ΔE = %.3e   rel_ΔF_rms = %.3e   rel_ΔW = %.3e   %s\n",
           r_disp.rel_dE, r_disp.rel_dF, r_disp.rel_dW, r_disp.pass ? "PASS" : "FAIL");
    return (r_coul.pass && r_disp.pass) ? 0 : 1;
} catch (const std::exception& error) {
    fprintf(stderr, "ERROR: %s\n", error.what());
    return 2;
}
