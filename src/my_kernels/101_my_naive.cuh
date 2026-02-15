#pragma once

#include <cuda_runtime.h>

/*
  计算一个矩阵乘法：C = A @ B
  其中 A:MxK  B:KxN  C:MxN
 */

__global__ void my_naive_kernel(int M, int K, int N, float alpha, const float *A,
                                const float *B, float beta, float *C){
    // 我们让每个线程计算矩阵中的一个元素

    // 那么我们一般会让grid_size 和输出的大小对应，block_size 和问题无关，
    // 这里我们选择就是(4,4)
    // 这里的 dim3 grid_size(M/4, N/4, 1)
    // threadIdx: thread 在 block 中的编号
    // blockIdx：block 在 grid 中的编号
    // blockDim：一个block中，每个方向上有多少thread

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < K) {
        float tmp = 0.0;
        for (int i = 0; i < K; i ++){
            // C[row][col] = A[row][0] * B[0][col]
            //             + A[row][1] * B[1][col]
            //             + A[row][2] * B[2][col]
            //             + ......
            //             + A[row][K] * B[K][col]

            // 这里要通过 A[row][i] 换算成一个scalar，具体下标
            int AIndex = row * K + i;
            int BIndex = i * N + col;
            tmp += A[AIndex] * B[BIndex];
        }
        // 简单的赋值，和项目对齐，我们使用 alpha*(A@B) + beta*C
//        CIndex = row * N + col;
//        C[CIndex] = tmp;
        C[row*N+col] = alpha * tmp + beta * C[row*N+col];
    }



}