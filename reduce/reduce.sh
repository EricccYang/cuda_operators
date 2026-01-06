#!/bin/bash

# 定义源文件数组（不带.cu后缀）
source_files=(
    "reduce_v1"
    "reduce_v2" 
    "reduce_v3"
    "reduce_v4"
    "reduce_v5-unroll8"
)

source_files_sm=(
    "reduce_v1_sm"
    "reduce_v2_sm"
    "reduce_v3_sm"
    "reduce_v4_sm"
    "reduce_v5-unroll8_sm"  # 修正了文件名拼写错误
)

# 函数：编译CUDA文件
compile_cuda() {
    local source_file="$1"
    local target_file="$2"
    echo "Compiling $source_file.cu -> $target_file"
    nvcc -o "$target_file" "$source_file.cu"
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed for $source_file.cu"
        exit 1
    fi
}

# 函数：运行ncu分析
run_ncu() {
    local target_file="$1"
    local output_file="$2"
    echo "Running NCU analysis: $target_file -> $output_file.ncu-rep"
    ncu --set basic -o "$output_file" "./$target_file"
    if [ $? -ne 0 ]; then
        echo "Warning: NCU analysis failed for $target_file, but continuing..."
    fi
}

# 主函数
main() {
    local build_only=false
    local analyze_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            "build")
                build_only=true
                ;;
            "analyze")
                analyze_only=true
                ;;
            "all")
                build_only=true
                analyze_only=true
                ;;
            *)
                echo "Unknown argument: $1"
                echo "Usage: $0 [build|analyze|all]"
                exit 1
                ;;
        esac
        shift
    done

    # 如果没有指定参数，默认执行所有操作
    if [ $# -eq 0 ] && [ "$build_only" = false ] && [ "$analyze_only" = false ]; then
        build_only=true
        analyze_only=true
    fi

    # 编译标准版本
    if [ "$build_only" = true ]; then
        echo "=== Compiling standard versions ==="
        for src in "${source_files[@]}"; do
            target=$(basename "$src")
            compile_cuda "$src" "$target"
        done
        
        echo "=== Compiling SM versions ==="
        for src in "${source_files_sm[@]}"; do
            target=$(basename "$src")
            compile_cuda "$src" "$target"
        done
    fi

    # 运行NCU分析
    if [ "$analyze_only" = true ]; then
        echo "=== Running NCU analysis ==="
        
        # 标准版本分析
        for src in "${source_files[@]}"; do
            target=$(basename "$src")
            run_ncu "$target" "$target"
        done
        
        # SM版本分析
        for src in "${source_files_sm[@]}"; do
            target=$(basename "$src")
            run_ncu "$target" "$target"
        done
    fi

    echo "Script completed successfully!"
}

# 检查ncu是否可用
if ! command -v ncu &> /dev/null; then
    echo "Warning: ncu command not found. Analysis will fail."
fi

# 执行主函数，传递所有参数
main "$@"