# CUDA共享内存与性能优化实战教程

一个完整的CUDA学习项目，通过4个递进的实战示例，帮助您掌握共享内存使用、kernel性能分析和矩阵乘法优化技术。

![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=flat&logo=ubuntu&logoColor=white)

## 🎯 学习目标

- ✅ 掌握CUDA共享内存的有效使用
- ✅ 学会分析和优化CUDA kernel性能  
- ✅ 实现高效的矩阵乘法算法
- ✅ 理解Bank冲突和避免方法
- ✅ 掌握性能分析工具的使用

## 📂 项目结构

```
cuda-learning/
├── src/                              # 🔥 核心示例代码
│   ├── 01_shared_memory_basics.cu      # 共享内存基础
│   ├── 02_matrix_multiply_evolution.cu # 矩阵乘法演进  
│   ├── 03_performance_analysis.cu      # 性能分析方法
│   └── 04_advanced_shared_memory.cu    # 高级优化技术
├── scripts/compile_and_run.sh          # 🛠️ 一键编译运行
├── Makefile                            # 📦 构建配置
├── docs/CUDA_Study_Guide.md           # 📚 详细学习指南
└── README.md                          # 📖 快速入门
```

## 🚀 快速开始

### 环境要求

- **GPU**: NVIDIA GPU (计算能力 3.0+)
- **CUDA**: 10.0+ 工具包
- **操作系统**: Ubuntu 18.04+ / CentOS 7+

### 安装CUDA

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-cuda-toolkit

# 验证安装
nvcc --version
nvidia-smi
```

### 编译运行

#### 方法1: 使用Makefile (推荐)
```bash
# 编译所有示例
make

# 运行特定示例
make run-example1    # 共享内存基础
make run-example2    # 矩阵乘法演进
make run-example3    # 性能分析
make run-example4    # 高级技术

# 运行所有示例
make examples

# 性能分析
make profile
```

#### 方法2: 使用脚本
```bash
# 使脚本可执行
chmod +x scripts/compile_and_run.sh

# 编译并运行示例1
./scripts/compile_and_run.sh 1

# 查看帮助
./scripts/compile_and_run.sh --help
```

## 📊 示例概览

### 示例1: 共享内存基础
**文件**: `01_shared_memory_basics.cu`

对比全局内存和共享内存的性能差异，展示共享内存在向量归约中的应用。

```bash
make run-example1
```

**学习要点**:
- CUDA内存层次结构
- 共享内存声明和使用
- 同步原语 `__syncthreads()`
- Bank冲突初步认识

### 示例2: 矩阵乘法演进  
**文件**: `02_matrix_multiply_evolution.cu`

从朴素实现到高度优化，展示矩阵乘法的完整优化过程。

```bash
make run-example2
```

**优化历程**:
1. **朴素版本** → 基础并行
2. **共享内存版本** → 数据重用
3. **避免Bank冲突** → 内存访问优化
4. **vs cuBLAS** → 与库函数对比

**性能提升**: 可达到20-100倍加速！

### 示例3: 性能分析方法
**文件**: `03_performance_analysis.cu`

全面的性能分析方法和工具使用指南。

```bash
make run-example3
```

**分析维度**:
- 内存访问模式分析
- 计算 vs 内存密集型对比
- 分支分化影响测量
- 占用率和带宽分析

### 示例4: 高级共享内存技术
**文件**: `04_advanced_shared_memory.cu`

高级优化技术的实战应用。

```bash
make run-example4
```

**技术要点**:
- Bank冲突深入分析
- 双缓冲技术
- Warp级别原语
- 矩阵转置优化

## 🔍 性能分析工具

### 基础分析 (CUDA Events)
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float time_ms;
cudaEventElapsedTime(&time_ms, start, stop);
```

### 专业分析 (Nsight Compute)
```bash
# 完整分析
ncu --set full ./your_program

# 特定指标
ncu --metrics sm__cycles_elapsed.avg,dram__bytes_read.sum ./your_program
```

### 系统分析 (Nsight Systems)
```bash
# 时间线分析
nsys profile -t cuda ./your_program
```

## 📈 典型性能提升

| 算法 | 基准 | 优化后 | 加速比 | 主要技术 |
|------|------|--------|--------|----------|
| 向量归约 | 全局内存 | 共享内存 | 3-5x | 数据重用 |
| 矩阵乘法 | 朴素实现 | Tiling+优化 | 10-20x | 共享内存+Bank优化 |
| 矩阵转置 | 朴素实现 | 共享内存 | 5-10x | 合并访问 |

## 🛠️ 常用命令

```bash
# 查看GPU信息
nvidia-smi

# 编译选项
nvcc -O3 -arch=sm_50 -std=c++11 source.cu -o output

# 性能分析
nvprof ./program
ncu --set basic ./program

# 内存检查
cuda-memcheck ./program

# 查看PTX代码
nvcc --ptx source.cu
```

## 📚 学习路径

1. **阅读详细指南**: [docs/CUDA_Study_Guide.md](docs/CUDA_Study_Guide.md)
2. **运行基础示例**: 从示例1开始，逐步运行所有示例
3. **性能分析实践**: 使用示例3学习分析工具
4. **高级技术探索**: 通过示例4掌握优化技术
5. **实际项目应用**: 将学到的技术应用到实际问题

## 🔥 关键优化技巧

### 共享内存使用
```cuda
// ✅ 避免Bank冲突
__shared__ float data[TILE_SIZE][TILE_SIZE + 1];  // +1填充

// ✅ 合理的tile大小
#define TILE_SIZE 16  // 或32，根据问题调整

// ✅ 同步管理
__syncthreads();  // 在需要的地方同步
```

### 内存访问优化
```cuda
// ✅ 合并访问
int idx = blockIdx.x * blockDim.x + threadIdx.x;
data[idx] = input[idx];  // 连续访问

// ❌ 避免跨步访问
data[idx * stride] = input[idx * stride];  // 性能损失
```

### 性能分析要点
- 📊 **占用率**: 目标 > 50%
- 📈 **内存效率**: 监控缓存命中率
- 🎯 **分支效率**: 减少warp分化
- ⚡ **带宽利用**: 接近理论峰值

## 🆘 常见问题

### Q: 编译错误 "nvcc not found"
```bash
# 安装CUDA工具包
sudo apt install nvidia-cuda-toolkit
# 或从NVIDIA官网下载最新版本
```

### Q: 运行时错误 "no CUDA-capable device"
```bash
# 检查GPU是否被识别
nvidia-smi
# 确保驱动正确安装
```

### Q: 性能没有预期提升
- 检查问题规模是否足够大
- 分析内存访问模式
- 使用性能分析工具找瓶颈
- 参考示例3的分析方法

## 🔗 相关资源

- [NVIDIA CUDA官方文档](https://docs.nvidia.com/cuda/)
- [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight工具文档](https://docs.nvidia.com/nsight-compute/)
- [GPU架构白皮书](https://developer.nvidia.com/gpu-architecture)

## 🤝 贡献

欢迎提交Issue和Pull Request！

- 🐛 报告bug
- 💡 提出改进建议  
- 📖 改进文档
- 🆕 添加新示例

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**🎯 开始你的CUDA性能优化之旅吧！从运行第一个示例开始：**

```bash
make run-example1
```

**📧 有问题？欢迎通过GitHub Issues交流！**