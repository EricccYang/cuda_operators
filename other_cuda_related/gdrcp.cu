#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <gdrcopy.h>

int main() {
    // 1. 初始化 GDRCopy
    gdr_t g = gdr_open();
    if (!g) { fprintf(stderr, "GDRCopy open failed\n"); exit(1); }

    // 2. 分配 GPU 内存（需标记为 CUDA 可访问）
    int *d_ptr;
    cudaMalloc(&d_ptr, sizeof(int));  // 分配 4 字节

    // 3. 映射 GPU 内存到 CPU 地址空间
    gdr_mh_t mh;
    if (gdr_pin_buffer(g, (unsigned long)d_ptr, sizeof(int), 0, 0, &mh)) {
        fprintf(stderr, "Pin buffer failed\n"); exit(1);
    }
    void *h_ptr;
    if (gdr_map(g, mh, &h_ptr, sizeof(int))) {
        fprintf(stderr, "Map failed\n"); exit(1);
    }
    int *cpu_ptr = (int*)gdr_get_h_ptr(h_ptr);

    // 4. 写入 GPU 数据
    int data = 42;
    cudaMemcpy(d_ptr, &data, sizeof(int), cudaMemcpyHostToDevice);

    // 5. 通过 GDRCopy 零拷贝读取（无需 cudaMemcpy）
    gdr_copy_from_mapping(g, h_ptr, cpu_ptr, sizeof(int));
    printf("Read from GPU: %d\n", *cpu_ptr);  // 输出: 42

    // 6. 清理
    gdr_unmap(g, mh, h_ptr, sizeof(int));
    gdr_unpin_buffer(g, mh);
    gdr_close(g);
    cudaFree(d_ptr);
    return 0;
}