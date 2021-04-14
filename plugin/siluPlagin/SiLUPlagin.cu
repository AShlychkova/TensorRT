#include "SiLUPlugin.h"
#include <cuda_fp16.h>

template <typename T_DATA>
__global__ void kernel(
    int N,
    T_DATA* inputs,
    T_DATA* outputs
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        outputs[index] = inputs[index] / (1. + fexp(-inputs[index]);
    }
    __syncthreads();
}

template <typename T>
int SiLUInference(
    int batchSize,
    int iC,
    int iH,
    int iW,
    T* inputs,
    T* outputs,
    cudaStream_t stream){
    const int nThreads = 512;
    int len = iC * iH * iW;

    int nBlocksCopy = (int)((float)len / nThreads) + 1;

    float stepACh = coordsRange / (float)(iH - 1);
    float stepACw = coordsRange / (float)(iW - 1);

    for(int i=0; i<batchSize; ++i){
        // NOTE: kernelCopy kernel can be replaced with cudaMemcpy function
        kernel<<<nBlocksCopy, nThreads, 0, stream>>>(len, inputs, outputs);
        outputs += len;
    }

    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        return 1;
    }
    return 0;
}

int SiLUPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    switch(iType){
    case DataType::kFLOAT:
        return SiLUInference(batchSize, iC, iH, iW (float*)inputs[0], (float*)outputs[0], stream);
    case DataType::kHALF:
        return SiLUInference(batchSize, iC, iH, iW, (__half*)inputs[0], (__half*)outputs[0], stream);
    }
    return 1;
}
