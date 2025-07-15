#include <stdio.h>
#include <cuda_runtime.h>

// 辅助函数声明
int _ConvertSMVer2Cores(int major, int minor);

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  SM count: %d\n", prop.multiProcessorCount);
        printf("  CUDA cores: %d\n", _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount);
        printf("  Global memory: %.2f GB\n", prop.totalGlobalMem/1024.0/1024.0/1024.0);
        printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock/1024);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
    }
    return 0;
}

// 辅助函数将SM版本转换为核心数
int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    return nGpuArchCoresPerSM[index-1].Cores;
} 