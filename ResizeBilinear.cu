/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ResizeBilinear.hpp"
#include <cuda_fp16.h>
#include <cassert>

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims ResizeBilinearPlugin::getOutputDimensions(int index,
                                                        const nvinfer1::Dims *inputDims,
                                                        int nbInputs) {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(is_CHW(input));
  assert(_ndims == 2);
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  int s = 0;
  for( int d=0; d<input.nbDims; ++d ) {
    output.type[d] = input.type[d];
    if( input.type[d] == nvinfer1::DimensionType::kSPATIAL ) {
      output.d[d] = int(input.d[d] * _scale[s++]);
    } else {
      output.d[d] = input.d[d];
    }
  }
  return output;
}

int ResizeBilinearPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  assert(is_CHW(this->getInputDims(0)));
  assert(is_CHW(_output_dims));
  assert(_ndims == 2);
  return 0;
}


template <typename Data>
__global__
void resize_bilinear_kernel_2d(int nbatch,
                              float2 scale,
                              int2 isize,
                              int2 osize,
                              Data const* idata, int istride, int ibatchstride,
                              Data*       odata, int ostride, int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  for( int batch=z0; batch<nbatch; batch+=gridDim.z ) {
    for( int oy=y0; oy<osize.y; oy+=blockDim.y*gridDim.y ) {
      for( int ox=x0; ox<osize.x; ox+=blockDim.x*gridDim.x ) {
        float ix = float(ox) / scale.x;
        float iy = float(oy) / scale.y;
        int w_low = ((int)ix < isize.x) ? (int)ix : (int)(isize.x - 1);
        int h_low = ((int)iy < isize.y) ? (int)iy : (int)(isize.y - 1);
        int w_high = w_low + 1 < isize.x ? w_low + 1 : (int)(isize.x - 1);
        int h_high = h_low + 1 < isize.y ? h_low + 1 : (int)(isize.y - 1);
        // int w_low = int(ix);
        // int h_low = int(iy);
        // int w_high = w_low + 1;
        // int h_high = h_low + 1;
        float lw = ix - w_low, lh = iy - h_low;
        float hw = 1 - lw,    hh = 1 - lh;
        Data W1 = (Data)(hh * hw), W2 = (Data)(hh * lw), W3 = (Data)(lh * hw), W4 = (Data)(lh * lw);
        Data v1 = idata[batch * ibatchstride + h_low * istride + w_low];
        Data v2 = idata[batch * ibatchstride + h_low * istride + w_high];
        Data v3 = idata[batch * ibatchstride + h_high * istride + w_low];
        Data v4 = idata[batch * ibatchstride + h_high * istride + w_high];
        odata[batch * obatchstride + oy * ostride + ox] = Data(W1 * v1) + Data(W2 * v2) + Data(W3 * v3) + Data(W4 * v4);
    
      }
    }
  }
}

int ResizeBilinearPlugin::enqueue(int batchSize,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int nchan = input_dims.d[0];
  switch( _ndims ) {
  case 2: {
    float2 scale = {_scale[1], _scale[0]};
    int2 isize = {input_dims.d[2], input_dims.d[1]};
    int2 osize = {_output_dims.d[2], _output_dims.d[1]};
    int istride =   input_dims.d[2];
    int ostride = _output_dims.d[2];
    int ibatchstride =   input_dims.d[1] * istride;
    int obatchstride = _output_dims.d[1] * ostride;
    dim3 block(16, 16);
    dim3 grid((osize.x - 1) / block.x + 1,
              (osize.y - 1) / block.y + 1,
              std::min(batchSize * nchan, 65535));
    if (getDataType()==nvinfer1::DataType::kFLOAT) {				
      resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, scale, isize, osize,
         static_cast<float const*>( inputs[0]), istride, ibatchstride,
         static_cast<float*      >(outputs[0]), ostride, obatchstride);
    } else {
      resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, scale, isize, osize,
         static_cast<__half const*>( inputs[0]), istride, ibatchstride,
         static_cast<__half*      >(outputs[0]), ostride, obatchstride);
    }
    return cudaGetLastError() != cudaSuccess;
  }
  default: return -1;
  }
}
