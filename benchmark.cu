/*
 * benchmark.cu
 *
 * Real runtime measurement for all CUDA kernels using CUDA events.
 * Compiled entirely by nvcc so that <<<>>> syntax is handled correctly on
 * all host compilers (including MSVC).
 *
 * Measurement methodology
 * -----------------------
 * 1. Allocate host + device arrays.
 * 2. Warm up each kernel (WARMUP_RUNS launches) to prime the GPU pipeline,
 *    JIT cache, and L2 state.  Warm-up times are discarded.
 * 3. Time BENCH_RUNS launches with cudaEventRecord / cudaEventElapsedTime.
 *    cudaEventSynchronize ensures the GPU has finished before the timer is
 *    read — no CPU-side timing uncertainty.
 * 4. Compute mean and standard deviation over the timed runs.
 * 5. Write results as CSV to output/runtimes.csv.
 *
 * Nothing in this file is estimated or inferred.  Every number comes from
 * real hardware execution.
 */

#include <cuda_runtime.h>
#include "kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#ifdef _WIN32
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const int N            = 16 * 1024 * 1024; // 16 M floats = 64 MB per array
static const int STRIDE       = 32;
static const int BLOCK_SIZE   = 256;
static const int WARMUP_RUNS  = 5;
static const int BENCH_RUNS   = 100;
// Number of arithmetic iterations for the compute-bound divergence kernels.
// High enough that memory latency is negligible vs compute time.
static const int COMPUTE_ITERS = 2000;

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel IDs
// ---------------------------------------------------------------------------
enum KernelID {
    K_COALESCED        = 0,
    K_STRIDED          = 1,
    K_DIVERGENT        = 2,
    K_DIVERGENT_COMPUTE = 3,
    K_COMPUTE_REF      = 4,
};

// ---------------------------------------------------------------------------
// Dispatch — one launch per call.  All <<<>>> here so MSVC never sees them.
// ---------------------------------------------------------------------------
static void launch_once(KernelID kid,
                        const float* d_a, const float* d_b, float* d_c,
                        int n, int stride, int iters,
                        int grid, int grid2, int block)
{
    switch (kid) {
    case K_COALESCED:
        coalesced_add    <<<grid,  block>>>(d_a, d_b, d_c, n);
        break;
    case K_STRIDED:
        strided_add      <<<grid2, block>>>(d_a, d_b, d_c, n, stride);
        break;
    case K_DIVERGENT:
        divergent_add    <<<grid,  block>>>(d_a, d_b, d_c, n);
        break;
    case K_DIVERGENT_COMPUTE:
        divergent_compute<<<grid,  block>>>(d_a, d_c, n, iters);
        break;
    case K_COMPUTE_REF:
        compute_ref      <<<grid,  block>>>(d_a, d_c, n, iters);
        break;
    }
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
static float* time_kernel(KernelID kid,
                           const float* d_a, const float* d_b, float* d_c,
                           int n, int stride, int iters,
                           int grid, int grid2, int block,
                           int warmup, int runs)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i)
        launch_once(kid, d_a, d_b, d_c, n, stride, iters, grid, grid2, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* times = new float[runs];
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_once(kid, d_a, d_b, d_c, n, stride, iters, grid, grid2, block);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return times;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
static void compute_stats(const float* arr, int n,
                           double* mean_out, double* std_out)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += arr[i];
    double mean = sum / n;
    double var  = 0.0;
    for (int i = 0; i < n; ++i) { double d = arr[i] - mean; var += d * d; }
    *mean_out = mean;
    *std_out  = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
}

static void ensure_output_dirs()
{
#ifdef _WIN32
    _mkdir("output");
    _mkdir("output/data");
#else
    mkdir("output",       0755);
    mkdir("output/data",  0755);
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    size_t bytes = (size_t)N * sizeof(float);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    if (!h_a || !h_b) { fprintf(stderr, "malloc failed\n"); return 1; }
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f + (float)i * 1e-7f;  // avoid zeros (div by zero in div path)
        h_b[i] = 1.0f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int n_strided = N / STRIDE;
    int grid      = (N         + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid2     = (n_strided + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device : %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Arrays : %d floats, %.1f MB each\n", N, (double)bytes / (1 << 20));
    printf("Compute kernels: %d iterations per thread\n\n", COMPUTE_ITERS);

    // ---- Run all kernels ----
    printf("Benchmarking coalesced_add ...\n");
    float* t1 = time_kernel(K_COALESCED, d_a, d_b, d_c, N, STRIDE, 0,
                             grid, grid2, BLOCK_SIZE, WARMUP_RUNS, BENCH_RUNS);

    printf("Benchmarking strided_add   (stride=%d) ...\n", STRIDE);
    float* t2 = time_kernel(K_STRIDED, d_a, d_b, d_c, N, STRIDE, 0,
                             grid, grid2, BLOCK_SIZE, WARMUP_RUNS, BENCH_RUNS);

    printf("Benchmarking divergent_add ...\n");
    float* t3 = time_kernel(K_DIVERGENT, d_a, d_b, d_c, N, STRIDE, 0,
                             grid, grid2, BLOCK_SIZE, WARMUP_RUNS, BENCH_RUNS);

    printf("Benchmarking divergent_compute (%d iters, fma vs div) ...\n", COMPUTE_ITERS);
    float* t4 = time_kernel(K_DIVERGENT_COMPUTE, d_a, d_b, d_c, N, STRIDE, COMPUTE_ITERS,
                             grid, grid2, BLOCK_SIZE, WARMUP_RUNS, BENCH_RUNS);

    printf("Benchmarking compute_ref       (%d iters, fma only)   ...\n", COMPUTE_ITERS);
    float* t5 = time_kernel(K_COMPUTE_REF, d_a, d_b, d_c, N, STRIDE, COMPUTE_ITERS,
                             grid, grid2, BLOCK_SIZE, WARMUP_RUNS, BENCH_RUNS);

    // ---- Stats ----
    double m1,s1, m2,s2, m3,s3, m4,s4, m5,s5;
    compute_stats(t1, BENCH_RUNS, &m1, &s1);
    compute_stats(t2, BENCH_RUNS, &m2, &s2);
    compute_stats(t3, BENCH_RUNS, &m3, &s3);
    compute_stats(t4, BENCH_RUNS, &m4, &s4);
    compute_stats(t5, BENCH_RUNS, &m5, &s5);

    // ms -> us
    m1*=1000; s1*=1000; m2*=1000; s2*=1000; m3*=1000; s3*=1000;
    m4*=1000; s4*=1000; m5*=1000; s5*=1000;

    printf("\n--- Results ---\n");
    printf("coalesced_add      : mean=%8.2f us  std=%6.2f us\n", m1, s1);
    printf("strided_add        : mean=%8.2f us  std=%6.2f us\n", m2, s2);
    printf("divergent_add      : mean=%8.2f us  std=%6.2f us\n", m3, s3);
    printf("divergent_compute  : mean=%8.2f us  std=%6.2f us\n", m4, s4);
    printf("compute_ref        : mean=%8.2f us  std=%6.2f us\n", m5, s5);
    printf("\ndivergence overhead: %.2fx  (divergent_compute / compute_ref)\n",
           m4 / m5);

    // ---- CSV ----
    ensure_output_dirs();
    FILE* f = fopen("output/data/runtimes.csv", "w");
    if (!f) { fprintf(stderr, "Cannot open output/data/runtimes.csv\n"); return 1; }
    fprintf(f, "kernel,n_elements,bytes_per_array,warmup_runs,bench_runs,mean_us,std_us\n");
    fprintf(f, "coalesced_add,%d,%zu,%d,%d,%.4f,%.4f\n",
            N,         bytes, WARMUP_RUNS, BENCH_RUNS, m1, s1);
    fprintf(f, "strided_add,%d,%zu,%d,%d,%.4f,%.4f\n",
            n_strided, bytes, WARMUP_RUNS, BENCH_RUNS, m2, s2);
    fprintf(f, "divergent_add,%d,%zu,%d,%d,%.4f,%.4f\n",
            N,         bytes, WARMUP_RUNS, BENCH_RUNS, m3, s3);
    fprintf(f, "divergent_compute,%d,%zu,%d,%d,%.4f,%.4f\n",
            N,         bytes, WARMUP_RUNS, BENCH_RUNS, m4, s4);
    fprintf(f, "compute_ref,%d,%zu,%d,%d,%.4f,%.4f\n",
            N,         bytes, WARMUP_RUNS, BENCH_RUNS, m5, s5);
    fclose(f);
    printf("\nSaved: output/data/runtimes.csv\n");

    delete[] t1; delete[] t2; delete[] t3; delete[] t4; delete[] t5;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b);
    return 0;
}
