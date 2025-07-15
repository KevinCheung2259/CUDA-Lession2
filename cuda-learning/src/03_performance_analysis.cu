#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>

void check_cuda_error(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

// 不同内存访问模式的kernel，用于性能分析
__global__ void coalesced_access(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 合并访问模式 - 连续的线程访问连续的内存
        output[idx] = input[idx] * 2.0f;
    }
}

__global__ void strided_access(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int strided_idx = idx * stride;
    if (strided_idx < n) {
        // 跨步访问模式 - 线程访问不连续的内存
        output[strided_idx] = input[strided_idx] * 2.0f;
    }
}

__global__ void random_access(float* input, float* output, int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 随机访问模式 - 无规律的内存访问
        int random_idx = indices[idx] % n;
        output[idx] = input[random_idx] * 2.0f;
    }
}

// 计算密集型kernel，用于分析计算吞吐量
__global__ void compute_intensive(float* input, float* output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // 执行大量计算操作
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) + cosf(val) + sqrtf(fabsf(val) + 1.0f);
        }
        output[idx] = val;
    }
}

// 内存密集型kernel，用于分析内存带宽
__global__ void memory_intensive(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 读取多个内存位置
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            int offset = (idx + i * blockDim.x) % n;
            sum += input[offset];
        }
        output[idx] = sum / 8.0f;
    }
}

// 分支分化kernel，用于分析warp效率
__global__ void divergent_branches(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        
        // 根据线程ID的不同执行不同的计算路径
        if (threadIdx.x % 2 == 0) {
            // 偶数线程执行复杂计算
            for (int i = 0; i < 100; i++) {
                val = sinf(val) * cosf(val);
            }
        } else {
            // 奇数线程执行简单计算
            val = val * 2.0f;
        }
        
        output[idx] = val;
    }
}

// 无分支版本，用于对比
__global__ void no_divergent_branches(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        
        // 所有线程执行相同的计算
        for (int i = 0; i < 50; i++) {
            val = sinf(val) * cosf(val);
        }
        val = val * 2.0f;
        
        output[idx] = val;
    }
}

// 获取设备属性和性能指标
void print_device_properties() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    printf("设备信息:\n");
    printf("=========================================\n");
    printf("设备名称: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("多处理器数量: %d\n", prop.multiProcessorCount);
    printf("每个多处理器的最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("每个块的最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("最大块维度: (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("最大网格维度: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("共享内存大小: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("全局内存大小: %.2f GB\n", (float)prop.totalGlobalMem / (1024*1024*1024));
    printf("内存总线宽度: %d bits\n", prop.memoryBusWidth);
    printf("内存时钟频率: %d MHz\n", prop.memoryClockRate / 1000);
    printf("L2缓存大小: %d KB\n", prop.l2CacheSize / 1024);
    printf("warp大小: %d\n", prop.warpSize);
    printf("\n");
    
    // 计算理论峰值性能
    float memory_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("理论内存带宽: %.1f GB/s\n", memory_bandwidth);
    printf("\n");
}

// 分析kernel的占用率
void analyze_occupancy(const void* kernel, int block_size) {
    int min_grid_size, block_size_opt;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_opt, kernel, 0, 0);
    
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);
    
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    float occupancy = (max_active_blocks * block_size / (float)prop.maxThreadsPerMultiProcessor) * 100;
    
    printf("占用率分析:\n");
    printf("当前块大小: %d\n", block_size);
    printf("建议块大小: %d\n", block_size_opt);
    printf("每个SM的活跃块数: %d\n", max_active_blocks);
    printf("理论占用率: %.1f%%\n", occupancy);
    printf("\n");
}

// 内存带宽测试
float measure_bandwidth(float* d_input, float* d_output, int n, int iterations) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        coalesced_access<<<grid_size, block_size>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    // 计算带宽 (读+写) * 数据大小 * 迭代次数 / 时间
    float bandwidth = (2.0f * n * sizeof(float) * iterations) / (time_ms / 1000.0f) / (1024*1024*1024);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return bandwidth;
}

int main() {
    printf("CUDA性能分析示例\n");
    printf("===========================================\n\n");
    
    // 打印设备属性
    print_device_properties();
    
    const int N = 1024 * 1024;  // 1M elements
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    // 分配内存
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    int* h_indices = (int*)malloc(N * sizeof(int));
    
    // 初始化数据
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
        h_indices[i] = rand();
    }
    
    float* d_input, *d_output;
    int* d_indices;
    
    check_cuda_error(cudaMalloc(&d_input, N * sizeof(float)), "allocate d_input");
    check_cuda_error(cudaMalloc(&d_output, N * sizeof(float)), "allocate d_output");
    check_cuda_error(cudaMalloc(&d_indices, N * sizeof(int)), "allocate d_indices");
    
    check_cuda_error(cudaMemcpy(d_input, h_input, N * sizeof(float), 
                                cudaMemcpyHostToDevice), "copy input");
    check_cuda_error(cudaMemcpy(d_indices, h_indices, N * sizeof(int), 
                                cudaMemcpyHostToDevice), "copy indices");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("性能测试结果:\n");
    printf("===========================================\n");
    
    // 1. 内存访问模式分析
    printf("1. 内存访问模式对比:\n");
    printf("-------------------------------------------\n");
    
    // 合并访问
    cudaEventRecord(start);
    coalesced_access<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_coalesced;
    cudaEventElapsedTime(&time_coalesced, start, stop);
    printf("合并访问: %.3f ms\n", time_coalesced);
    
    // 跨步访问
    cudaEventRecord(start);
    strided_access<<<grid_size/4, block_size>>>(d_input, d_output, N, 4);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_strided;
    cudaEventElapsedTime(&time_strided, start, stop);
    printf("跨步访问 (stride=4): %.3f ms (%.1fx slower)\n", 
           time_strided, time_strided / time_coalesced);
    
    // 随机访问
    cudaEventRecord(start);
    random_access<<<grid_size, block_size>>>(d_input, d_output, d_indices, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_random;
    cudaEventElapsedTime(&time_random, start, stop);
    printf("随机访问: %.3f ms (%.1fx slower)\n", 
           time_random, time_random / time_coalesced);
    
    // 2. 计算 vs 内存密集型对比
    printf("\n2. 计算 vs 内存密集型对比:\n");
    printf("-------------------------------------------\n");
    
    // 内存密集型
    cudaEventRecord(start);
    memory_intensive<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_memory;
    cudaEventElapsedTime(&time_memory, start, stop);
    printf("内存密集型: %.3f ms\n", time_memory);
    
    // 计算密集型
    cudaEventRecord(start);
    compute_intensive<<<grid_size, block_size>>>(d_input, d_output, N, 10);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_compute;
    cudaEventElapsedTime(&time_compute, start, stop);
    printf("计算密集型: %.3f ms\n", time_compute);
    
    // 3. 分支分化影响
    printf("\n3. 分支分化影响:\n");
    printf("-------------------------------------------\n");
    
    // 有分支分化
    cudaEventRecord(start);
    divergent_branches<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_divergent;
    cudaEventElapsedTime(&time_divergent, start, stop);
    printf("分支分化版本: %.3f ms\n", time_divergent);
    
    // 无分支分化
    cudaEventRecord(start);
    no_divergent_branches<<<grid_size, block_size>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_no_divergent;
    cudaEventElapsedTime(&time_no_divergent, start, stop);
    printf("统一执行版本: %.3f ms (%.1fx faster)\n", 
           time_no_divergent, time_divergent / time_no_divergent);
    
    // 4. 占用率分析
    printf("\n4. 占用率分析:\n");
    printf("-------------------------------------------\n");
    analyze_occupancy((const void*)coalesced_access, block_size);
    
    // 5. 内存带宽测试
    printf("5. 内存带宽测试:\n");
    printf("-------------------------------------------\n");
    float bandwidth = measure_bandwidth(d_input, d_output, N, 100);
    printf("实际内存带宽: %.1f GB/s\n", bandwidth);
    
    // 6. 性能分析工具使用提示
    printf("\n6. 推荐的性能分析流程:\n");
    printf("===========================================\n");
    printf("步骤1: 编译程序\n");
    printf("  nvcc -O3 -o perf_analysis 03_performance_analysis.cu\n\n");
    
    printf("步骤2: 基础性能分析 (nvprof)\n");
    printf("  nvprof ./perf_analysis\n");
    printf("  nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./perf_analysis\n\n");
    
    printf("步骤3: 详细分析 (Nsight Compute)\n");
    printf("  ncu --set full ./perf_analysis\n");
    printf("  ncu --metrics sm__cycles_elapsed.avg,dram__bytes_read.sum ./perf_analysis\n\n");
    
    printf("步骤4: 可视化分析 (Nsight Systems)\n");
    printf("  nsys profile -t cuda ./perf_analysis\n\n");
    
    printf("关键指标解释:\n");
    printf("- 占用率 (Occupancy): 衡量SM利用率，目标>50%%\n");
    printf("- 内存效率 (Memory Efficiency): 有效内存传输比例\n");
    printf("- 分支效率 (Branch Efficiency): warp内指令一致性\n");
    printf("- 缓存命中率: L1/L2缓存性能\n");
    printf("- 指令吞吐量: 计算单元利用率\n");
    
    // 清理资源
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
    free(h_input);
    free(h_output);
    free(h_indices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}