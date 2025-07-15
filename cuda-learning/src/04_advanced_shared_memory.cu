#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

// 演示Bank冲突的不同情况
__global__ void demonstrate_bank_conflicts(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();
    
    // 情况1: 无冲突 - 连续访问
    if (tid < blockDim.x / 4) {
        output[idx] = sdata[tid * 4] + sdata[tid * 4 + 1] + 
                     sdata[tid * 4 + 2] + sdata[tid * 4 + 3];
    }
}

__global__ void bank_conflict_example(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = input[idx];
    }
    __syncthreads();
    
    // 情况2: 严重Bank冲突 - 同一warp中的线程访问相同bank
    if (tid < WARP_SIZE) {
        int bank_id = tid % 32;  // 每个bank有32个地址
        // 所有线程访问同一个bank的不同地址，造成32路冲突
        output[idx] = sdata[bank_id * 32];
    }
}

__global__ void avoid_bank_conflicts_padding(float* input, float* output, int n) {
    // 使用填充避免bank冲突
    __shared__ float sdata[TILE_SIZE][TILE_SIZE + 1];  // +1避免bank冲突
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + tx;
    
    if (idx < n && ty == 0) {
        sdata[ty][tx] = input[idx];
    }
    __syncthreads();
    
    // 现在列访问不会有bank冲突
    if (idx < n) {
        output[idx] = sdata[tx % TILE_SIZE][ty];
    }
}

// 双缓冲技术示例 - 异步数据传输和计算
__global__ void double_buffering_example(float* input, float* output, int n, int tile_size) {
    extern __shared__ float sdata[];
    
    // 双缓冲区
    float* buffer_a = sdata;
    float* buffer_b = &sdata[tile_size];
    
    int tid = threadIdx.x;
    int num_tiles = (n + tile_size - 1) / tile_size;
    
    // 预加载第一个tile
    if (tid < tile_size && tid < n) {
        buffer_a[tid] = input[tid];
    }
    __syncthreads();
    
    for (int tile = 0; tile < num_tiles - 1; tile++) {
        // 当前tile的计算使用buffer_a
        float result = 0.0f;
        if (tid < tile_size) {
            result = buffer_a[tid] * 2.0f;  // 简单计算
        }
        
        // 同时异步加载下一个tile到buffer_b
        int next_offset = (tile + 1) * tile_size + tid;
        if (tid < tile_size && next_offset < n) {
            buffer_b[tid] = input[next_offset];
        }
        __syncthreads();
        
        // 写回当前结果
        int current_offset = tile * tile_size + tid;
        if (tid < tile_size && current_offset < n) {
            output[current_offset] = result;
        }
        
        // 交换缓冲区
        float* temp = buffer_a;
        buffer_a = buffer_b;
        buffer_b = temp;
        __syncthreads();
    }
    
    // 处理最后一个tile
    if (tid < tile_size) {
        int final_offset = (num_tiles - 1) * tile_size + tid;
        if (final_offset < n) {
            output[final_offset] = buffer_a[tid] * 2.0f;
        }
    }
}

// 复杂的共享内存模式：矩阵转置
__global__ void matrix_transpose_shared(float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1避免bank冲突
    
    int x_in = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y_in = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    // 加载到共享内存 (合并访问)
    if (x_in < width && y_in < height) {
        tile[threadIdx.y][threadIdx.x] = input[y_in * width + x_in];
    }
    __syncthreads();
    
    // 写回到全局内存 (转置后也是合并访问)
    if (x_out < height && y_out < width) {
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// Warp级别的原语使用示例
__global__ void warp_level_primitives(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp内归约使用shuffle指令
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // 只有warp内的第一个线程写入共享内存
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
    
    // 第一个warp对所有warp的结果进行最终归约
    if (warp_id == 0) {
        float final_val = (lane_id < blockDim.x / WARP_SIZE) ? sdata[lane_id] : 0.0f;
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            final_val += __shfl_down_sync(0xffffffff, final_val, offset);
        }
        
        if (lane_id == 0 && blockIdx.x == 0) {
            output[0] = final_val;  // 简化：只写入第一个块的结果
        }
    }
}

// 动态并行示例：在设备端启动新的kernel
__global__ void child_kernel(float* data, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) {
        data[idx] = sqrtf(data[idx]);
    }
}

__global__ void parent_kernel_with_dynamic_parallelism(float* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 父kernel先做一些预处理
    if (tid < n) {
        data[tid] = data[tid] * data[tid];  // 平方
    }
    __syncthreads();
    
    // 只有第一个线程启动子kernel
    if (tid == 0) {
        int child_blocks = (n + 255) / 256;
        child_kernel<<<child_blocks, 256>>>(data, 0, n);
        cudaDeviceSynchronize();  // 等待子kernel完成
    }
}

// 性能测试函数
float benchmark_kernel(void (*kernel_func)(float*, float*, int), 
                      float* d_input, float* d_output, int n, 
                      int block_size, const char* name) {
    int grid_size = (n + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 预热
    kernel_func<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // 测试
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel_func<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= 100;  // 平均时间
    
    printf("%s: %.3f ms\n", name, time_ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time_ms;
}

int main() {
    printf("CUDA高级共享内存技术示例\n");
    printf("===============================================\n\n");
    
    const int N = 1024 * 1024;
    const int block_size = 256;
    
    // 分配内存
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }
    
    float* d_input, *d_output;
    check_cuda_error(cudaMalloc(&d_input, N * sizeof(float)), "allocate input");
    check_cuda_error(cudaMalloc(&d_output, N * sizeof(float)), "allocate output");
    
    check_cuda_error(cudaMemcpy(d_input, h_input, N * sizeof(float), 
                                cudaMemcpyHostToDevice), "copy input");
    
    printf("1. Bank冲突演示和优化:\n");
    printf("-----------------------------------------------\n");
    
    // 测试bank冲突的影响
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int grid_size = (N + block_size - 1) / block_size;
    
    // 无冲突版本
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        demonstrate_bank_conflicts<<<grid_size, block_size, block_size * sizeof(float)>>>
            (d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_no_conflict;
    cudaEventElapsedTime(&time_no_conflict, start, stop);
    time_no_conflict /= 100;
    
    // 有冲突版本
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bank_conflict_example<<<grid_size, block_size, block_size * sizeof(float)>>>
            (d_input, d_output, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_conflict;
    cudaEventElapsedTime(&time_conflict, start, stop);
    time_conflict /= 100;
    
    printf("无Bank冲突: %.3f ms\n", time_no_conflict);
    printf("有Bank冲突: %.3f ms (%.1fx slower)\n", 
           time_conflict, time_conflict / time_no_conflict);
    
    printf("\n2. 矩阵转置测试 (1024x1024):\n");
    printf("-----------------------------------------------\n");
    
    const int matrix_size = 1024;
    float* d_matrix_input, *d_matrix_output;
    check_cuda_error(cudaMalloc(&d_matrix_input, matrix_size * matrix_size * sizeof(float)), 
                     "allocate matrix input");
    check_cuda_error(cudaMalloc(&d_matrix_output, matrix_size * matrix_size * sizeof(float)), 
                     "allocate matrix output");
    
    // 初始化矩阵
    float* h_matrix = (float*)malloc(matrix_size * matrix_size * sizeof(float));
    for (int i = 0; i < matrix_size * matrix_size; i++) {
        h_matrix[i] = (float)i;
    }
    check_cuda_error(cudaMemcpy(d_matrix_input, h_matrix, 
                                matrix_size * matrix_size * sizeof(float), 
                                cudaMemcpyHostToDevice), "copy matrix");
    
    dim3 block_2d(TILE_SIZE, TILE_SIZE);
    dim3 grid_2d((matrix_size + TILE_SIZE - 1) / TILE_SIZE, 
                 (matrix_size + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matrix_transpose_shared<<<grid_2d, block_2d>>>(d_matrix_input, d_matrix_output, 
                                                        matrix_size, matrix_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_transpose;
    cudaEventElapsedTime(&time_transpose, start, stop);
    time_transpose /= 10;
    
    printf("矩阵转置 (共享内存优化): %.3f ms\n", time_transpose);
    
    // 计算带宽
    float bytes_transferred = 2.0f * matrix_size * matrix_size * sizeof(float);  // 读+写
    float bandwidth = bytes_transferred / (time_transpose / 1000.0f) / (1024*1024*1024);
    printf("有效带宽: %.1f GB/s\n", bandwidth);
    
    printf("\n3. Warp级别原语测试:\n");
    printf("-----------------------------------------------\n");
    
    cudaEventRecord(start);
    warp_level_primitives<<<grid_size, block_size, 
                           (block_size / WARP_SIZE) * sizeof(float)>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_warp;
    cudaEventElapsedTime(&time_warp, start, stop);
    printf("Warp级别归约: %.3f ms\n", time_warp);
    
    printf("\n4. 双缓冲技术测试:\n");
    printf("-----------------------------------------------\n");
    
    const int tile_size = 256;
    cudaEventRecord(start);
    double_buffering_example<<<1, tile_size, 2 * tile_size * sizeof(float)>>>
        (d_input, d_output, tile_size * 4, tile_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_double_buffer;
    cudaEventElapsedTime(&time_double_buffer, start, stop);
    printf("双缓冲示例: %.3f ms\n", time_double_buffer);
    
    printf("\n5. 共享内存最佳实践总结:\n");
    printf("===============================================\n");
    printf("✓ 使用共享内存减少全局内存访问\n");
    printf("✓ 避免Bank冲突:\n");
    printf("  - 使用填充 (+1) 避免列访问冲突\n");
    printf("  - 交错访问模式\n");
    printf("  - 使用__shfl_*函数减少共享内存使用\n");
    printf("✓ 优化内存访问模式:\n");
    printf("  - 合并全局内存访问\n");
    printf("  - 使用双缓冲隐藏延迟\n");
    printf("  - 预取数据到共享内存\n");
    printf("✓ 同步优化:\n");
    printf("  - 最小化__syncthreads()调用\n");
    printf("  - 使用warp级别原语\n");
    printf("  - 考虑异步执行\n");
    
    printf("\n6. 调试和分析建议:\n");
    printf("===============================================\n");
    printf("使用以下工具分析共享内存使用:\n");
    printf("• nvcc --ptxas-options=-v 查看寄存器和共享内存使用\n");
    printf("• ncu --metrics shared_load_throughput,shared_store_throughput\n");
    printf("• ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum\n");
    printf("• 使用CUDA-MEMCHECK检查内存错误\n");
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_matrix_input);
    cudaFree(d_matrix_output);
    free(h_input);
    free(h_output);
    free(h_matrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}