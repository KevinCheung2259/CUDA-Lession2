#!/bin/bash

# CUDA学习项目编译和运行脚本
# 使用方法: ./compile_and_run.sh [example_number]

set -e  # 遇到错误时停止

PROJECT_DIR="/home/scrapybara/cuda-learning"
SRC_DIR="$PROJECT_DIR/src"
BUILD_DIR="$PROJECT_DIR/build"

# 创建构建目录
mkdir -p "$BUILD_DIR"

# 检查NVCC是否可用
if ! command -v nvcc &> /dev/null; then
    echo "错误: nvcc未找到。请确保CUDA已正确安装并在PATH中。"
    echo "在Ubuntu上安装CUDA:"
    echo "  sudo apt update"
    echo "  sudo apt install nvidia-cuda-toolkit"
    echo "  # 或者从NVIDIA官网下载最新版本"
    exit 1
fi

# 显示CUDA版本信息
echo "CUDA版本信息:"
nvcc --version
echo ""

# 编译选项
NVCC_FLAGS="-O3 -arch=sm_50 -std=c++11"
CUDA_LIBS="-lcublas"

# 编译函数
compile_example() {
    local example_name=$1
    local source_file="$SRC_DIR/${example_name}.cu"
    local output_file="$BUILD_DIR/${example_name}"
    
    if [ ! -f "$source_file" ]; then
        echo "错误: 源文件 $source_file 不存在"
        return 1
    fi
    
    echo "编译 $example_name..."
    if nvcc $NVCC_FLAGS "$source_file" -o "$output_file" $CUDA_LIBS; then
        echo "✓ 编译成功: $output_file"
        return 0
    else
        echo "✗ 编译失败: $example_name"
        return 1
    fi
}

# 运行函数
run_example() {
    local example_name=$1
    local executable="$BUILD_DIR/${example_name}"
    
    if [ ! -f "$executable" ]; then
        echo "错误: 可执行文件 $executable 不存在，请先编译"
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "运行示例: $example_name"
    echo "=========================================="
    
    # 检查GPU是否可用
    if ! nvidia-smi &> /dev/null; then
        echo "警告: 未检测到NVIDIA GPU或驱动未正确安装"
        echo "程序可能无法正常运行"
    fi
    
    # 运行程序
    "$executable"
    local exit_code=$?
    
    echo ""
    echo "程序运行完毕 (退出码: $exit_code)"
    return $exit_code
}

# 性能分析函数
profile_example() {
    local example_name=$1
    local executable="$BUILD_DIR/${example_name}"
    
    if [ ! -f "$executable" ]; then
        echo "错误: 可执行文件 $executable 不存在，请先编译"
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "性能分析: $example_name"
    echo "=========================================="
    
    # 检查分析工具
    if command -v nvprof &> /dev/null; then
        echo "使用nvprof进行基础分析..."
        nvprof "$executable"
    elif command -v ncu &> /dev/null; then
        echo "使用Nsight Compute进行详细分析..."
        ncu --set basic "$executable"
    else
        echo "未找到性能分析工具 (nvprof或ncu)"
        echo "请安装CUDA工具包的完整版本"
        return 1
    fi
}

# 显示帮助信息
show_help() {
    echo "CUDA学习项目编译和运行脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [选项] [示例编号]"
    echo ""
    echo "选项:"
    echo "  -h, --help        显示此帮助信息"
    echo "  -c, --compile     只编译，不运行"
    echo "  -r, --run         只运行（需要先编译）"
    echo "  -p, --profile     性能分析"
    echo "  -a, --all         编译并运行所有示例"
    echo ""
    echo "可用示例:"
    echo "  1  共享内存基础 (01_shared_memory_basics)"
    echo "  2  矩阵乘法演进 (02_matrix_multiply_evolution)"
    echo "  3  性能分析方法 (03_performance_analysis)"
    echo "  4  高级共享内存技术 (04_advanced_shared_memory)"
    echo ""
    echo "示例:"
    echo "  $0 1              编译并运行示例1"
    echo "  $0 -c 2           只编译示例2"
    echo "  $0 -p 3           对示例3进行性能分析"
    echo "  $0 -a             编译并运行所有示例"
}

# 获取示例名称
get_example_name() {
    case $1 in
        1) echo "01_shared_memory_basics" ;;
        2) echo "02_matrix_multiply_evolution" ;;
        3) echo "03_performance_analysis" ;;
        4) echo "04_advanced_shared_memory" ;;
        *) echo "" ;;
    esac
}

# 主函数
main() {
    local compile_only=false
    local run_only=false
    local profile_only=false
    local run_all=false
    local example_num=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--compile)
                compile_only=true
                shift
                ;;
            -r|--run)
                run_only=true
                shift
                ;;
            -p|--profile)
                profile_only=true
                shift
                ;;
            -a|--all)
                run_all=true
                shift
                ;;
            [1-4])
                example_num=$1
                shift
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 运行所有示例
    if [ "$run_all" = true ]; then
        echo "编译并运行所有示例..."
        for i in {1..4}; do
            example_name=$(get_example_name $i)
            if [ -n "$example_name" ]; then
                compile_example "$example_name"
                run_example "$example_name"
                echo ""
            fi
        done
        return 0
    fi
    
    # 检查示例编号
    if [ -z "$example_num" ]; then
        echo "请指定示例编号 (1-4) 或使用 -a 运行所有示例"
        show_help
        exit 1
    fi
    
    example_name=$(get_example_name "$example_num")
    if [ -z "$example_name" ]; then
        echo "无效的示例编号: $example_num"
        show_help
        exit 1
    fi
    
    # 执行相应操作
    if [ "$compile_only" = true ]; then
        compile_example "$example_name"
    elif [ "$run_only" = true ]; then
        run_example "$example_name"
    elif [ "$profile_only" = true ]; then
        profile_example "$example_name"
    else
        # 默认：编译并运行
        if compile_example "$example_name"; then
            run_example "$example_name"
        fi
    fi
}

# 脚本入口
main "$@"