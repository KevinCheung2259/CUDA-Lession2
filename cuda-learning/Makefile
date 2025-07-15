# CUDA学习项目Makefile

# 基本设置
PROJECT_NAME = cuda-learning
SRC_DIR = src
BUILD_DIR = build
SCRIPTS_DIR = scripts

# 编译器和标志
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_50 -std=c++11 -Xcompiler -Wall
CUDA_LIBS = -lcublas
INCLUDES = -I.

# 源文件和目标文件
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
TARGETS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SOURCES))

# 默认目标
.PHONY: all clean help install-deps check-cuda examples profile

all: check-cuda $(TARGETS)

# 检查CUDA环境
check-cuda:
	@echo "检查CUDA环境..."
	@which nvcc > /dev/null || (echo "错误: nvcc未找到，请安装CUDA工具包" && exit 1)
	@echo "CUDA版本信息:"
	@nvcc --version | head -n 4
	@echo ""

# 创建构建目录
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# 编译规则
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "编译 $<..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@ $(CUDA_LIBS)
	@echo "✓ 编译完成: $@"

# 单独编译每个示例
example1: $(BUILD_DIR)/01_shared_memory_basics
example2: $(BUILD_DIR)/02_matrix_multiply_evolution  
example3: $(BUILD_DIR)/03_performance_analysis
example4: $(BUILD_DIR)/04_advanced_shared_memory

# 运行示例
run-example1: example1
	@echo "运行共享内存基础示例..."
	@$(BUILD_DIR)/01_shared_memory_basics

run-example2: example2
	@echo "运行矩阵乘法演进示例..."
	@$(BUILD_DIR)/02_matrix_multiply_evolution

run-example3: example3
	@echo "运行性能分析示例..."
	@$(BUILD_DIR)/03_performance_analysis

run-example4: example4
	@echo "运行高级共享内存示例..."
	@$(BUILD_DIR)/04_advanced_shared_memory

# 运行所有示例
examples: all
	@echo "运行所有示例..."
	@for target in $(TARGETS); do \
		echo ""; \
		echo "=========================================="; \
		echo "运行: $$target"; \
		echo "=========================================="; \
		$$target || echo "示例运行失败: $$target"; \
	done

# 性能分析
profile: all
	@echo "开始性能分析..."
	@if command -v nvprof >/dev/null 2>&1; then \
		echo "使用nvprof分析..."; \
		for target in $(TARGETS); do \
			echo "分析: $$target"; \
			nvprof $$target 2>&1 | head -n 50; \
		done; \
	elif command -v ncu >/dev/null 2>&1; then \
		echo "使用Nsight Compute分析..."; \
		for target in $(TARGETS); do \
			echo "分析: $$target"; \
			ncu --set basic $$target; \
		done; \
	else \
		echo "未找到性能分析工具 (nvprof或ncu)"; \
	fi

# 安装依赖 (仅适用于有sudo权限的系统)
install-deps:
	@echo "安装CUDA开发依赖..."
	@if command -v apt >/dev/null 2>&1; then \
		echo "使用apt安装..."; \
		sudo apt update; \
		sudo apt install -y nvidia-cuda-toolkit nvidia-cuda-dev; \
	elif command -v yum >/dev/null 2>&1; then \
		echo "使用yum安装..."; \
		sudo yum install -y cuda-toolkit cuda-devel; \
	else \
		echo "请手动从NVIDIA官网下载并安装CUDA工具包"; \
		echo "https://developer.nvidia.com/cuda-downloads"; \
	fi

# 清理
clean:
	@echo "清理构建文件..."
	@rm -rf $(BUILD_DIR)
	@echo "清理完成"

# 创建发布版本
dist: clean all
	@echo "创建发布包..."
	@tar -czf $(PROJECT_NAME).tar.gz \
		$(SRC_DIR)/ \
		$(SCRIPTS_DIR)/ \
		$(BUILD_DIR)/ \
		Makefile \
		README.md \
		docs/
	@echo "发布包已创建: $(PROJECT_NAME).tar.gz"

# 显示帮助信息
help:
	@echo "CUDA学习项目构建系统"
	@echo ""
	@echo "可用目标:"
	@echo "  all              编译所有示例"
	@echo "  example1         编译共享内存基础示例"
	@echo "  example2         编译矩阵乘法演进示例"
	@echo "  example3         编译性能分析示例"
	@echo "  example4         编译高级共享内存示例"
	@echo ""
	@echo "  run-example1     运行共享内存基础示例"
	@echo "  run-example2     运行矩阵乘法演进示例"
	@echo "  run-example3     运行性能分析示例"
	@echo "  run-example4     运行高级共享内存示例"
	@echo "  examples         运行所有示例"
	@echo ""
	@echo "  profile          对所有示例进行性能分析"
	@echo "  check-cuda       检查CUDA环境"
	@echo "  install-deps     安装CUDA依赖 (需要sudo)"
	@echo "  clean            清理构建文件"
	@echo "  dist             创建发布包"
	@echo "  help             显示此帮助信息"
	@echo ""
	@echo "示例用法:"
	@echo "  make              # 编译所有示例"
	@echo "  make run-example1 # 编译并运行示例1"
	@echo "  make examples     # 编译并运行所有示例"
	@echo "  make profile      # 进行性能分析"

# 显示项目信息
info:
	@echo "项目信息:"
	@echo "  名称: $(PROJECT_NAME)"
	@echo "  源码目录: $(SRC_DIR)"
	@echo "  构建目录: $(BUILD_DIR)"
	@echo "  编译器: $(NVCC)"
	@echo "  编译标志: $(NVCC_FLAGS)"
	@echo "  CUDA库: $(CUDA_LIBS)"
	@echo ""
	@echo "源文件:"
	@for src in $(SOURCES); do echo "  $$src"; done
	@echo ""
	@echo "目标文件:"
	@for target in $(TARGETS); do echo "  $$target"; done