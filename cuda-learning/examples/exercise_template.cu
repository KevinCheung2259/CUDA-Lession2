// CUDA共享内存练习模板
// 练习：实现一个高效的数组求和kernel

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

// 练习1: 基础版本 - 使用全局内存的原子操作
__global__ void array_sum_atomic(float* input, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: 实现使用atomicAdd的版本
    // 提示: 每个线程处理一个元素，使用atomicAdd累加到result
    
}

// 练习2: 共享内存版本 - 块内归约
__global__ void array_sum_shared(float* input, float* result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: 实现共享内存版本的归约
    // 步骤：
    // 1. 加载数据到共享内存
    // 2. 在共享内存中进行归约
    // 3. 第一个线程将结果写回全局内存
    
    // 提示: 使用二分归约算法
    // for (int s = blockDim.x / 2; s > 0; s >>= 1) { ... }
    
}

// 练习3: 优化版本 - 避免bank冲突和warp分化
__global__ void array_sum_optimized(float* input, float* result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: 实现优化版本
    // 优化要点：
    // 1. 处理边界条件（n不是blockDim.x的倍数）
    // 2. 避免warp分化（使用__shfl_down_sync）
    // 3. 减少共享内存bank冲突
    
    // 提示: 可以结合warp级别原语
    // val += __shfl_down_sync(0xffffffff, val, offset);
    
}

// 练习4: 多级归约 - 处理大数组
__global__ void array_sum_multilevel(float* input, float* partial_results, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int grid_size = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: 实现多级归约
    // 1. 每个线程处理多个元素
    // 2. 块内归约得到部分结果
    // 3. 将部分结果存储到partial_results数组
    
    // 处理多个元素的模式：
    // float sum = 0.0f;
    // for (int i = idx; i < n; i += grid_size) {
    //     sum += input[i];
    // }
    
}

// CPU参考实现
float array_sum_cpu(float* input, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += input[i];
    }
    return sum;
}

// 验证结果
bool verify_result(float expected, float actual, float tolerance = 1e-5f) {
    float error = fabsf(expected - actual) / expected;
    printf("期望结果: %.6f, 实际结果: %.6f, 相对误差: %.2e\n", 
           expected, actual, error);
    return error < tolerance;
}

int main() {
    printf("CUDA数组求和练习\n");
    printf("===============================\n\n");
    
    // 测试参数
    const int N = 1024 * 1024;  // 1M elements
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    printf("数组大小: %d 元素\n", N);
    printf("块大小: %d\n", block_size);
    printf("网格大小: %d\n", grid_size);
    printf("\n");
    
    // 分配主机内存
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_result = (float*)malloc(sizeof(float));
    
    // 初始化数据
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;  // 0-1之间的随机数
    }
    
    // CPU参考结果
    printf("计算CPU参考结果...\n");
    float cpu_result = array_sum_cpu(h_input, N);
    printf("CPU结果: %.6f\n\n", cpu_result);
    
    // 分配设备内存
    float* d_input, *d_result, *d_partial_results;
    check_cuda_error(cudaMalloc(&d_input, N * sizeof(float)), "allocate input");
    check_cuda_error(cudaMalloc(&d_result, sizeof(float)), "allocate result");
    check_cuda_error(cudaMalloc(&d_partial_results, grid_size * sizeof(float)), 
                     "allocate partial results");
    
    // 复制数据到设备
    check_cuda_error(cudaMemcpy(d_input, h_input, N * sizeof(float), 
                                cudaMemcpyHostToDevice), "copy input");
    
    // 创建事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("GPU实现测试:\n");
    printf("-------------------------------\n");
    
    // 测试1: 原子操作版本
    printf("测试1: 原子操作版本\n");
    check_cuda_error(cudaMemset(d_result, 0, sizeof(float)), "reset result");
    
    cudaEventRecord(start);
    array_sum_atomic<<<grid_size, block_size>>>(d_input, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time1;
    cudaEventElapsedTime(&time1, start, stop);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(float), 
                                cudaMemcpyDeviceToHost), "copy result");
    
    printf("时间: %.3f ms\n", time1);
    bool correct1 = verify_result(cpu_result, *h_result);
    printf("正确性: %s\n\n", correct1 ? "✓" : "✗");
    
    // 测试2: 共享内存版本
    printf("测试2: 共享内存版本\n");
    check_cuda_error(cudaMemset(d_result, 0, sizeof(float)), "reset result");
    
    cudaEventRecord(start);
    array_sum_shared<<<grid_size, block_size, block_size * sizeof(float)>>>
        (d_input, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time2;
    cudaEventElapsedTime(&time2, start, stop);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(float), 
                                cudaMemcpyDeviceToHost), "copy result");
    
    printf("时间: %.3f ms\n", time2);
    bool correct2 = verify_result(cpu_result, *h_result);
    printf("正确性: %s\n", correct2 ? "✓" : "✗");
    if (time1 > 0) {
        printf("相比原子版本: %.2fx faster\n\n", time1 / time2);
    }
    
    // 测试3: 优化版本
    printf("测试3: 优化版本\n");
    check_cuda_error(cudaMemset(d_result, 0, sizeof(float)), "reset result");
    
    cudaEventRecord(start);
    array_sum_optimized<<<grid_size, block_size, block_size * sizeof(float)>>>
        (d_input, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time3;
    cudaEventElapsedTime(&time3, start, stop);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(float), 
                                cudaMemcpyDeviceToHost), "copy result");
    
    printf("时间: %.3f ms\n", time3);
    bool correct3 = verify_result(cpu_result, *h_result);
    printf("正确性: %s\n", correct3 ? "✓" : "✗");
    if (time2 > 0) {
        printf("相比共享内存版本: %.2fx faster\n\n", time2 / time3);
    }
    
    // 测试4: 多级归约版本
    printf("测试4: 多级归约版本\n");
    
    cudaEventRecord(start);
    array_sum_multilevel<<<grid_size, block_size, block_size * sizeof(float)>>>
        (d_input, d_partial_results, N);
    
    // 第二级归约：对部分结果求和
    array_sum_shared<<<1, block_size, block_size * sizeof(float)>>>
        (d_partial_results, d_result, grid_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time4;
    cudaEventElapsedTime(&time4, start, stop);
    
    check_cuda_error(cudaMemcpy(h_result, d_result, sizeof(float), 
                                cudaMemcpyDeviceToHost), "copy result");
    
    printf("时间: %.3f ms\n", time4);
    bool correct4 = verify_result(cpu_result, *h_result);
    printf("正确性: %s\n", correct4 ? "✓" : "✗");
    
    printf("\n练习总结:\n");
    printf("===============================\n");
    printf("实现要点:\n");
    printf("• 原子操作: 简单但性能受限\n");
    printf("• 共享内存: 显著提升性能\n");
    printf("• 优化技术: warp原语、避免冲突\n");
    printf("• 多级归约: 处理大规模数据\n");
    printf("\n");
    printf("性能对比:\n");
    if (time1 > 0 && time2 > 0 && time3 > 0) {
        printf("原子操作:     %.3f ms (1.00x)\n", time1);
        printf("共享内存:     %.3f ms (%.2fx)\n", time2, time1/time2);
        printf("优化版本:     %.3f ms (%.2fx)\n", time3, time1/time3);
        printf("多级归约:     %.3f ms (%.2fx)\n", time4, time1/time4);
    }
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_result);
    cudaFree(d_partial_results);
    free(h_input);
    free(h_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}