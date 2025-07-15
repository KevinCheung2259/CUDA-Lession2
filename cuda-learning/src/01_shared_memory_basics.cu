#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 不使用共享内存的向量归约
__global__ void vector_reduce_global(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 每个线程处理多个元素
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += input[i];
    }
    
    // 简单的归约，仅用于对比性能
    atomicAdd(output, sum);
}

// 使用共享内存的向量归约
__global__ void vector_reduce_shared(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 将数据加载到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 在共享内存中进行归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 块内的第一个线程将结果写回全局内存
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// 展示共享内存Bank冲突的例子
__global__ void shared_memory_bank_conflict(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 产生Bank冲突的访问模式 - 同一warp中的线程访问相同bank
        sdata[tid * 2] = input[idx];
        __syncthreads();
        
        // 读取时也会产生bank冲突
        output[idx] = sdata[tid * 2];
    }
}

// 避免Bank冲突的优化版本
__global__ void shared_memory_no_bank_conflict(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // 使用填充避免bank冲突
        int padded_idx = tid + (tid / 32);  // 每32个元素添加一个填充
        sdata[padded_idx] = input[idx];
        __syncthreads();
        
        output[idx] = sdata[padded_idx];
    }
}

void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

int main() {
    const int N = 1024 * 1024;  // 1M elements
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    // 分配主机内存
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(sizeof(float));
    
    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // 所有元素都是1，便于验证结果
    }
    
    // 分配设备内存
    float* d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, N * sizeof(float)), "allocate input");
    check_cuda_error(cudaMalloc(&d_output, sizeof(float)), "allocate output");
    
    // 复制数据到设备
    check_cuda_error(cudaMemcpy(d_input, h_input, N * sizeof(float), 
                                cudaMemcpyHostToDevice), "copy input to device");
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("CUDA共享内存基础示例\n");
    printf("数组大小: %d 元素\n", N);
    printf("预期结果: %.0f\n\n", (float)N);
    
    // 测试1: 不使用共享内存的归约
    check_cuda_error(cudaMemset(d_output, 0, sizeof(float)), "reset output");
    
    cudaEventRecord(start);
    vector_reduce_global<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float time_global;
    cudaEventElapsedTime(&time_global, start, stop);
    
    check_cuda_error(cudaMemcpy(h_output, d_output, sizeof(float), 
                                cudaMemcpyDeviceToHost), "copy result from device");
    
    printf("全局内存归约:\n");
    printf("  结果: %.0f\n", *h_output);
    printf("  时间: %.3f ms\n\n", time_global);
    
    // 测试2: 使用共享内存的归约
    check_cuda_error(cudaMemset(d_output, 0, sizeof(float)), "reset output");
    
    int shared_mem_size = block_size * sizeof(float);
    
    cudaEventRecord(start);
    vector_reduce_shared<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float time_shared;
    cudaEventElapsedTime(&time_shared, start, stop);
    
    check_cuda_error(cudaMemcpy(h_output, d_output, sizeof(float), 
                                cudaMemcpyDeviceToHost), "copy result from device");
    
    printf("共享内存归约:\n");
    printf("  结果: %.0f\n", *h_output);
    printf("  时间: %.3f ms\n", time_shared);
    printf("  加速比: %.2fx\n\n", time_global / time_shared);
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}