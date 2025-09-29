#include <cuda_runtime.h>
#include <math.h>

__global__ void customScalarKernel(const float *input, const float *grid,
                                   float *output, int N, int C, int H_in,
                                   int W_in, int H_out, int W_out) {

  int h_out = threadIdx.y + blockIdx.y * blockDim.y;
  int w_out = threadIdx.x + blockIdx.x * blockDim.x;
  if (h_out >= H_out || w_out >= W_out)
    return;

  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      int grid_offset = n * H_out * W_out * 2 + h_out * W_out * 2 + w_out * 2;
      float gx = grid[grid_offset + 0];
      float gy = grid[grid_offset + 1];

      float x = (W_in - 1) * (gx + 1.0f) / 2.0f;
      float y = (H_in - 1) * (gy + 1.0f) / 2.0f;

      int x0 = static_cast<int>(floorf(x));
      int x1 = x0 + 1;
      int y0 = static_cast<int>(floorf(y));
      int y1 = y0 + 1;

      float dx = x - x0;
      float dy = y - y0;

      x0 = max(0, min(x0, W_in - 1));
      x1 = max(0, min(x1, W_in - 1));
      y0 = max(0, min(y0, H_in - 1));
      y1 = max(0, min(y1, H_in - 1));

      const int inp_n_c_offset = n * C * H_in * W_in + c * H_in * W_in;

      float q00 = input[inp_n_c_offset + y0 * W_in + x0];
      float q01 = input[inp_n_c_offset + y0 * W_in + x1];
      float q10 = input[inp_n_c_offset + y1 * W_in + x0];
      float q11 = input[inp_n_c_offset + y1 * W_in + x1];

      float result = q00 * (1.f - dx) * (1.f - dy) + q01 * dx * (1.f - dy) +
                     q10 * (1.f - dx) * dy + q11 * dx * dy;

      const int out_n_c_offset = n * C * H_out * W_out + c * H_out * W_out;
      output[out_n_c_offset + h_out * W_out + w_out] = result;
    }
  }
}

void customScalarImpl(const float *input, const float *grid, float *output,
                      int N, int C, int H_in, int W_in, int H_out,
                      int W_out, cudaStream_t stream) {
  int block_size = 16;
  dim3 block(block_size, block_size);
  dim3 grid_size((W_out + block_size - 1) / block_size, (H_out + block_size - 1) / block_size);

  customScalarKernel<<<grid_size, block, 0, stream>>>(
      input, grid, output, N, C, H_in,W_in, H_out, W_out);
}
