#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// 在 Host 上调用，但是在 Device 上执行
__global__ void kernel(uint *A, uint *B, int row) {
  auto x = threadIdx.x / 4;
  auto y = threadIdx.x % 4;
  A[x * row + y] = x;
  B[x * row + y] = y;
}

// 在 Host 上执行
int main(int argc, char **argv) {
  uint *Xs, *Ys;      // Host 中两个指针，指向内存
  uint *Xs_d, *Ys_d;  // Host 中两个指针，指向 Device 内存

  uint SIZE = 4;  // size

  // 内存中分配 4x4 大小的 uint 空间
  Xs = (uint *)malloc(SIZE * SIZE * sizeof(uint));
  Ys = (uint *)malloc(SIZE * SIZE * sizeof(uint));

  // CUDA 显存中分配 4x4 大小的 uint 空格键
  cudaMalloc((void **)&Xs_d, SIZE * SIZE * sizeof(uint));
  cudaMalloc((void **)&Ys_d, SIZE * SIZE * sizeof(uint));

  // thread 组成 block，block 组成 grid
  /*
    因为是 4x4 矩阵，自然的想法是启动一个大小为 16 个 thread 的 block
    只启动一个1个block，
    16 个 thread 会在GPU上并行执行

   */
  dim3 grid_size(1, 1, 1);
  dim3 block_size(4 * 4);

  // Kernel Launch
  // 在 GPU 上，按照 grid_size，block_size 的配置，启动这个名叫 kernel 的函数，Xs_d, Ys_d, 4 作为输入参数
  // kernel launch 是异步的，CPU并不会阻塞等待调用结束，
  kernel<<<grid_size, block_size>>>(Xs_d, Ys_d, 4);

  // 我们调用cudaMemcpy，将 Device 上的数据 Xs_d，Ys_d 复制到 CPU 上
  // 为什么不会读到GPU 中的脏数据？因为 CUDA Stream 的存在，kernel 和 cudaMemcpy 都是一个CUDAStream，一定是FIFO的
  cudaMemcpy(Xs, Xs_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);
  cudaMemcpy(Ys, Ys_d, SIZE * SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

  // 阻塞CPU执行，等待GPU上的任务执行完毕，再执行下面的动作
  cudaDeviceSynchronize();

  // 因为kernel launch 和 cudamemcpy，CPU能够访问的Xs和Ys已经是GPU的计算结果。直接打印就是计算后的结果
  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      std::cout << "[" << Xs[row * SIZE + col] << "|" << Ys[row * SIZE + col]
                << "] ";
    }
    std::cout << "\n";
  }

  // 释放CPU和GPU上的内存
  cudaFree(Xs_d);
  cudaFree(Ys_d);
  free(Xs);
  free(Ys);
}
