#!/bin/bash

# CUDA环境检查脚本
# 检查CUDA开发环境是否正确配置

echo "CUDA开发环境检查"
echo "======================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_passed=0
check_total=0

# 检查函数
check_command() {
    local cmd=$1
    local name=$2
    local required=$3
    
    ((check_total++))
    
    if command -v "$cmd" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $name: 已安装"
        if [ "$cmd" = "nvcc" ]; then
            echo "  版本: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
        elif [ "$cmd" = "nvidia-smi" ]; then
            echo "  驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)"
        fi
        ((check_passed++))
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}✗${NC} $name: 未找到 (必需)"
        else
            echo -e "${YELLOW}!${NC} $name: 未找到 (可选)"
        fi
        return 1
    fi
}

# 检查GPU
check_gpu() {
    ((check_total++))
    
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓${NC} NVIDIA GPU: 已检测到"
        echo "  GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | while read line; do
            echo "    $line"
        done
        ((check_passed++))
        return 0
    else
        echo -e "${RED}✗${NC} NVIDIA GPU: 未检测到或驱动未安装"
        return 1
    fi
}

# 检查CUDA设备
check_cuda_devices() {
    ((check_total++))
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} CUDA设备: 检测到 $gpu_count 个GPU"
            nvidia-smi -L
            ((check_passed++))
            return 0
        fi
    fi
    
    echo -e "${RED}✗${NC} CUDA设备: 未检测到可用的CUDA设备"
    return 1
}

# 检查编译环境
check_compile_environment() {
    ((check_total++))
    
    # 创建临时测试文件
    local test_file="/tmp/cuda_test.cu"
    cat > "$test_file" << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel() {
    printf("Hello from GPU!\n");
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("CUDA devices: %d\n", device_count);
    
    if (device_count > 0) {
        test_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    
    return 0;
}
EOF

    # 尝试编译
    if nvcc -o /tmp/cuda_test "$test_file" &> /dev/null; then
        echo -e "${GREEN}✓${NC} CUDA编译: 编译测试通过"
        
        # 尝试运行
        if /tmp/cuda_test &> /dev/null; then
            echo -e "${GREEN}✓${NC} CUDA运行: 运行测试通过"
        else
            echo -e "${YELLOW}!${NC} CUDA运行: 编译成功但运行失败"
        fi
        
        ((check_passed++))
        rm -f /tmp/cuda_test "$test_file"
        return 0
    else
        echo -e "${RED}✗${NC} CUDA编译: 编译测试失败"
        rm -f "$test_file"
        return 1
    fi
}

# 主检查流程
echo ""
echo "1. 基础工具检查"
echo "--------------------------------------"
check_command "nvcc" "CUDA编译器 (nvcc)" "true"
check_command "nvidia-smi" "NVIDIA系统管理工具" "true"
check_command "make" "Make构建工具" "false"
check_command "git" "Git版本控制" "false"

echo ""
echo "2. GPU和驱动检查"
echo "--------------------------------------"
check_gpu
check_cuda_devices

echo ""
echo "3. 性能分析工具检查"
echo "--------------------------------------"
check_command "nvprof" "NVPROF分析器" "false"
check_command "ncu" "Nsight Compute" "false"
check_command "nsys" "Nsight Systems" "false"

echo ""
echo "4. 编译和运行测试"
echo "--------------------------------------"
if command -v nvcc &> /dev/null && nvidia-smi &> /dev/null; then
    check_compile_environment
else
    echo -e "${YELLOW}!${NC} 跳过编译测试 (缺少必需工具)"
fi

echo ""
echo "检查总结"
echo "======================================"
echo "通过检查: $check_passed / $check_total"

if [ "$check_passed" -eq "$check_total" ]; then
    echo -e "${GREEN}🎉 环境配置完美！可以开始CUDA开发了！${NC}"
    exit 0
elif [ "$check_passed" -ge $((check_total * 3 / 4)) ]; then
    echo -e "${YELLOW}⚠️  环境基本就绪，但有一些可选工具未安装${NC}"
    exit 0
else
    echo -e "${RED}❌ 环境配置不完整，请安装缺失的组件${NC}"
    echo ""
    echo "安装建议:"
    echo "--------------------------------------"
    
    if ! command -v nvcc &> /dev/null; then
        echo "• 安装CUDA工具包:"
        echo "  Ubuntu: sudo apt install nvidia-cuda-toolkit"
        echo "  或从 https://developer.nvidia.com/cuda-downloads 下载"
    fi
    
    if ! nvidia-smi &> /dev/null; then
        echo "• 安装NVIDIA驱动:"
        echo "  Ubuntu: sudo ubuntu-drivers autoinstall"
        echo "  或 sudo apt install nvidia-driver-<version>"
    fi
    
    if ! command -v make &> /dev/null; then
        echo "• 安装构建工具:"
        echo "  Ubuntu: sudo apt install build-essential"
    fi
    
    echo ""
    echo "安装完成后重新运行此脚本进行验证。"
    exit 1
fi