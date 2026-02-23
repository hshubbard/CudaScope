/*
 * benchmark.cpp
 *
 * Real runtime measurement for the three CUDA kernels using CUDA events.
 *
 * Measurement methodology
 * -----------------------
 * 1. Allocate host + device arrays.
 * 2. Warm up each kernel (WARMUP_RUNS launches) to prime the GPU pipeline,
 *    JIT cache, and L2 state. Warm-up times are discarded.
 * 3. Time BENCH_RUNS launches with cudaEventRecord / cudaEventElapsedTime.
 *    cudaEventSynchronize ensures the GPU has finished before reading the
 *    timer — no CPU-side timing uncertainty.
 * 4. Compute mean and standard deviation over the timed runs.
 * 5. Write results as CSV to output/runtimes.csv.
 *
 * Nothing in this file is estimated or inferred.  Every number is from
 * real hardware execution.
 */

#include <cuda_runtime.h>
#include "kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#  include <direct.h>   // _mkdir
#else
#  include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static const int   N           = 16 * 1024 * 1024; // 16 M floats ≈ 64 MB/array
static const int   STRIDE      = 32;                // stride for kernel 2
static const int   BLOCK_SIZE  = 256;
static const int   WARMUP_RUNS = 5;
static const int   BENCH_RUNS  = 100;

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d – %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: time a single void(void) lambda over `runs` iterations
// Returns array of per-run elapsed milliseconds (caller must free[]).
// ---------------------------------------------------------------------------
template <typename KernelLauncher>
static float* time_kernel(KernelLauncher launch, int warmup, int runs)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < warmup; ++i)
        launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    float* times = new float[runs];
    for (int i = 0; i < runs; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop)); // milliseconds
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return times;
}

// ---------------------------------------------------------------------------
// Helper: compute mean + stddev from float array
// ---------------------------------------------------------------------------
static void stats(const float* arr, int n, double* mean_out, double* std_out)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += arr[i];
    double mean = sum / n;

    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = arr[i] - mean;
        var += d * d;
    }
    *mean_out = mean;
    *std_out  = (n > 1) ? std::sqrt(var / (n - 1)) : 0.0;
}

// ---------------------------------------------------------------------------
// mkdir -p equivalent (portable, single level)
// ---------------------------------------------------------------------------
static void ensure_output_dir()
{
#ifdef _WIN32
    _mkdir("output");
#else
    mkdir("output", 0755);
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // ---- Allocate host arrays ----
    size_t bytes = (size_t)N * sizeof(float);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    if (!h_a || !h_b) { fprintf(stderr, "Host malloc failed\n"); return 1; }
    for (int i = 0; i < N; ++i) { h_a[i] = (float)i * 0.001f; h_b[i] = 1.0f; }

    // ---- Allocate device arrays ----
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ---- Print device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Array size: %d floats (%.1f MB per array)\n\n",
           N, (double)bytes / (1024*1024));

    // ---- Kernel 1: coalesced_add ----
    printf("Benchmarking coalesced_add ...\n");
    auto launch1 = [&](){ coalesced_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N); };
    float* t1 = time_kernel(launch1, WARMUP_RUNS, BENCH_RUNS);

    // ---- Kernel 2: strided_add ----
    // Strided kernel only writes to N/STRIDE elements to stay within bounds.
    int n_strided = N / STRIDE;
    int grid2 = (n_strided + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Benchmarking strided_add (stride=%d) ...\n", STRIDE);
    auto launch2 = [&](){ strided_add<<<grid2, BLOCK_SIZE>>>(d_a, d_b, d_c, N, STRIDE); };
    float* t2 = time_kernel(launch2, WARMUP_RUNS, BENCH_RUNS);

    // ---- Kernel 3: divergent_add ----
    printf("Benchmarking divergent_add ...\n");
    auto launch3 = [&](){ divergent_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N); };
    float* t3 = time_kernel(launch3, WARMUP_RUNS, BENCH_RUNS);

    // ---- Compute stats ----
    double m1, s1, m2, s2, m3, s3;
    stats(t1, BENCH_RUNS, &m1, &s1);
    stats(t2, BENCH_RUNS, &m2, &s2);
    stats(t3, BENCH_RUNS, &m3, &s3);

    // Convert ms → µs
    auto ms2us = [](double v){ return v * 1000.0; };

    printf("\n--- Results ---\n");
    printf("coalesced_add : mean=%.2f µs  std=%.2f µs\n",
           ms2us(m1), ms2us(s1));
    printf("strided_add   : mean=%.2f µs  std=%.2f µs\n",
           ms2us(m2), ms2us(s2));
    printf("divergent_add : mean=%.2f µs  std=%.2f µs\n",
           ms2us(m3), ms2us(s3));

    // ---- Write CSV ----
    ensure_output_dir();
    FILE* f = fopen("output/runtimes.csv", "w");
    if (!f) { fprintf(stderr, "Cannot open output/runtimes.csv\n"); return 1; }

    fprintf(f, "kernel,n_elements,bytes_per_array,warmup_runs,bench_runs,"
               "mean_us,std_us\n");
    fprintf(f, "coalesced_add,%d,%zu,%d,%d,%.4f,%.4f\n",
            N, bytes, WARMUP_RUNS, BENCH_RUNS, ms2us(m1), ms2us(s1));
    fprintf(f, "strided_add,%d,%zu,%d,%d,%.4f,%.4f\n",
            n_strided, bytes, WARMUP_RUNS, BENCH_RUNS, ms2us(m2), ms2us(s2));
    fprintf(f, "divergent_add,%d,%zu,%d,%d,%.4f,%.4f\n",
            N, bytes, WARMUP_RUNS, BENCH_RUNS, ms2us(m3), ms2us(s3));
    fclose(f);
    printf("\nSaved: output/runtimes.csv\n");

    // ---- Cleanup ----
    delete[] t1; delete[] t2; delete[] t3;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b);

    return 0;
}
