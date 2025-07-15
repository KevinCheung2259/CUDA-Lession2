#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16

void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

// 版本1: 朴素的矩阵乘法 - 只使用全局内存
__global__ void matrix_mul_naive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 版本2: 使用共享内存 - 基础tile版本
__global__ void matrix_mul_shared_basic(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // 循环处理所有tile
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载A和B的tile到共享内存
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 计算当前tile的贡献
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// 版本3: 优化的共享内存版本 - 避免bank冲突
__global__ void matrix_mul_shared_optimized(float* A, float* B, float* C, int N) {
    // 使用填充避免bank冲突
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 预取数据到共享内存
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // 使用局部变量减少共享内存访问
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// 版本4: 进一步优化 - 每个线程计算多个元素
__global__ void matrix_mul_shared_advanced(float* A, float* B, float* C, int N) {
    const int THREAD_TILE_SIZE = 4;  // 每个线程计算4x4个元素
    
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // 每个线程计算多个元素
    float sum[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {0};
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载数据到共享内存
        for (int i = 0; i < THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                int row = by * TILE_SIZE + ty * THREAD_TILE_SIZE + i;
                int col_A = t * TILE_SIZE + tx * THREAD_TILE_SIZE + j;
                
                if (row < N && col_A < N && ty * THREAD_TILE_SIZE + i < TILE_SIZE && 
                    tx * THREAD_TILE_SIZE + j < TILE_SIZE) {
                    As[ty * THREAD_TILE_SIZE + i][tx * THREAD_TILE_SIZE + j] = 
                        A[row * N + col_A];
                }
                
                int row_B = t * TILE_SIZE + ty * THREAD_TILE_SIZE + i;
                int col = bx * TILE_SIZE + tx * THREAD_TILE_SIZE + j;
                
                if (row_B < N && col < N && ty * THREAD_TILE_SIZE + i < TILE_SIZE && 
                    tx * THREAD_TILE_SIZE + j < TILE_SIZE) {
                    Bs[ty * THREAD_TILE_SIZE + i][tx * THREAD_TILE_SIZE + j] = 
                        B[row_B * N + col];
                }
            }
        }
        
        __syncthreads();
        
        // 计算多个元素
        for (int i = 0; i < THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                for (int k = 0; k < TILE_SIZE; k++) {
                    if (ty * THREAD_TILE_SIZE + i < TILE_SIZE && 
                        tx * THREAD_TILE_SIZE + j < TILE_SIZE) {
                        sum[i][j] += As[ty * THREAD_TILE_SIZE + i][k] * 
                                   Bs[k][tx * THREAD_TILE_SIZE + j];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // 写回结果
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            int row = by * TILE_SIZE + ty * THREAD_TILE_SIZE + i;
            int col = bx * TILE_SIZE + tx * THREAD_TILE_SIZE + j;
            
            if (row < N && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

// 验证结果正确性
bool verify_result(float* C_ref, float* C_test, int N) {
    float max_error = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float error = fabs(C_ref[i] - C_test[i]);
        max_error = fmax(max_error, error);
    }
    
    printf("最大误差: %e\n", max_error);
    return max_error < 1e-3;
}

// CPU参考实现
void matrix_mul_cpu(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 1024;  // 矩阵大小
    const int size = N * N * sizeof(float);
    
    printf("矩阵乘法性能对比 (大小: %dx%d)\n", N, N);
    printf("=====================================================\n\n");
    
    // 分配主机内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_ref = (float*)malloc(size);
    float* h_C_test = (float*)malloc(size);
    
    // 初始化矩阵
    srand(42);
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // CPU参考实现
    printf("计算CPU参考结果...\n");
    auto start_cpu = clock();
    matrix_mul_cpu(h_A, h_B, h_C_ref, N);
    auto end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000;
    
    // 分配设备内存
    float* d_A, *d_B, *d_C;
    check_cuda_error(cudaMalloc(&d_A, size), "allocate d_A");
    check_cuda_error(cudaMalloc(&d_B, size), "allocate d_B");
    check_cuda_error(cudaMalloc(&d_C, size), "allocate d_C");
    
    // 复制数据到设备
    check_cuda_error(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "copy A");
    check_cuda_error(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "copy B");
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("\nGPU实现对比:\n");
    printf("-----------------------------------------------------\n");
    
    // 测试朴素版本
    cudaEventRecord(start);
    matrix_mul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);
    
    check_cuda_error(cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost), "copy result");
    bool correct = verify_result(h_C_ref, h_C_test, N);
    
    printf("朴素版本:\n");
    printf("  时间: %.3f ms\n", time_naive);
    printf("  正确性: %s\n", correct ? "✓" : "✗");
    printf("  相比CPU: %.2fx\n\n", cpu_time / time_naive);
    
    // 测试基础共享内存版本
    cudaEventRecord(start);
    matrix_mul_shared_basic<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_shared_basic;
    cudaEventElapsedTime(&time_shared_basic, start, stop);
    
    check_cuda_error(cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost), "copy result");
    correct = verify_result(h_C_ref, h_C_test, N);
    
    printf("基础共享内存版本:\n");
    printf("  时间: %.3f ms\n", time_shared_basic);
    printf("  正确性: %s\n", correct ? "✓" : "✗");
    printf("  相比朴素版本: %.2fx\n", time_naive / time_shared_basic);
    printf("  相比CPU: %.2fx\n\n", cpu_time / time_shared_basic);
    
    // 测试优化共享内存版本
    cudaEventRecord(start);
    matrix_mul_shared_optimized<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_shared_opt;
    cudaEventElapsedTime(&time_shared_opt, start, stop);
    
    check_cuda_error(cudaMemcpy(h_C_test, d_C, size, cudaMemcpyDeviceToHost), "copy result");
    correct = verify_result(h_C_ref, h_C_test, N);
    
    printf("优化共享内存版本:\n");
    printf("  时间: %.3f ms\n", time_shared_opt);
    printf("  正确性: %s\n", correct ? "✓" : "✗");
    printf("  相比基础版本: %.2fx\n", time_shared_basic / time_shared_opt);
    printf("  相比CPU: %.2fx\n\n", cpu_time / time_shared_opt);
    
    // cuBLAS对比 (如果可用)
    cublasHandle_t handle;
    if (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS) {
        const float alpha = 1.0f, beta = 0.0f;
        
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                   &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float time_cublas;
        cudaEventElapsedTime(&time_cublas, start, stop);
        
        printf("cuBLAS (参考):\n");
        printf("  时间: %.3f ms\n", time_cublas);
        printf("  相比优化版本: %.2fx\n", time_shared_opt / time_cublas);
        printf("  相比CPU: %.2fx\n\n", cpu_time / time_cublas);
        
        cublasDestroy(handle);
    }
    
    printf("性能总结:\n");
    printf("-----------------------------------------------------\n");
    printf("算法                    时间(ms)    相比CPU    相比朴素\n");
    printf("CPU                     %.1f       1.0x       -\n", cpu_time);
    printf("GPU朴素                 %.1f       %.1fx      1.0x\n", 
           time_naive, cpu_time/time_naive);
    printf("GPU共享内存(基础)       %.1f       %.1fx      %.1fx\n", 
           time_shared_basic, cpu_time/time_shared_basic, time_naive/time_shared_basic);
    printf("GPU共享内存(优化)       %.1f       %.1fx      %.1fx\n", 
           time_shared_opt, cpu_time/time_shared_opt, time_naive/time_shared_opt);
    
    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_ref);
    free(h_C_test);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}