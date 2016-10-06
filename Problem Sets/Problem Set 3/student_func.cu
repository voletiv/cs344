/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

#include <stdio.h>

__device__ float fAtomicMin(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

__device__ float fAtomicMax(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }
    return __int_as_float(old);
}

__global__
void reduceMin(const float* const d_logLuminance,
                float* d_min,
                const size_t numCols,
                const size_t numRows)
{
    extern __shared__ float sdata[];
    
    const int globalId = blockIdx.x*blockDim.x + threadIdx.x;
    if (globalId >= numRows*numCols)
        return;
    
    const int tid = threadIdx.x;
    
    // copy d_logLuminance data into shared memory
    sdata[tid] = d_logLuminance[globalId];
    __syncthreads();
    
    for (unsigned int s = blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            if (sdata[tid+s] < sdata[tid])
            {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid==0) fAtomicMin(d_min, sdata[tid]);

}

__global__
void reduceMax(const float* const d_logLuminance,
                float* d_max,
                const size_t numCols,
                const size_t numRows)
{
    extern __shared__ float sdata[];
    
    const int globalId = blockIdx.x*blockDim.x + threadIdx.x;
    if (globalId >= numRows*numCols)
        return;
    
    const int tid = threadIdx.x;
    
    // copy d_logLuminance data into shared memory
    sdata[tid] = d_logLuminance[globalId];
    __syncthreads();
    
    for (unsigned int s = blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            if (sdata[tid+s] > sdata[tid])
            {
                sdata[tid] = sdata[tid+s];
            }
        }
        __syncthreads();
    }
    
    if (tid==0) fAtomicMax(d_max, sdata[tid]);

}

__global__
void findBins(const float* d_logLuminance,
                unsigned int* d_bins,
                float* d_min,
                float* d_lumRange,
                const size_t numBins,
                const size_t numCols,
                const size_t numRows)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= numCols*numRows)
        return;
    
    float min = *d_min;
    float lumRange = *d_lumRange;
    int bin = (d_logLuminance[tid] - min) / (lumRange) * numBins;
    if (bin >= numBins) bin = numBins - 1;
    atomicAdd(&d_bins[bin], 1);
}

__global__
void inclusiveScanHist( unsigned int* d_bins,
                        unsigned int* const d_cdf,
                        const size_t numBins)
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (tid >= numBins)
        return;
    
    if (tid>0)
        d_cdf[tid] = d_bins[tid-1];
    else
        d_cdf[tid] = 0;
    __syncthreads();
    
    int toAdd = 0;
    
    for(int i=1; i<=numBins; i<<=1)
    {
        if (tid >= i)
        {
            toAdd = d_cdf[tid-i];
            __syncthreads();
            atomicAdd(&d_cdf[tid], toAdd);
            __syncthreads();
        }
    }
    
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    
    //int numBins = 33;
    
    //const int maxThreadsPerBlock = 1024;
    int blockSi = 1024;
    int gridSi = (numRows*numCols + blockSi -1)/blockSi;
    
    /*
    float min=1000, max=-100;
    float* h_logLum = (float *)malloc(numRows*numCols*sizeof(float));
    checkCudaErrors(cudaMemcpy(h_logLum, d_logLuminance, sizeof(numRows*numCols*sizeof(float)), cudaMemcpyDeviceToHost));
    for (int i=0; i<numRows*numCols; i++) {
        if (h_logLum[i] > max)
            max = h_logLum[i];
        if (h_logLum[i] < min)
            min = h_logLum[i];
    }
    */
    
    float h_min = 1000;
    float* d_min;
    checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_min, &h_min, sizeof(float), cudaMemcpyHostToDevice));
    
    reduceMin<<<gridSi, blockSi, blockSi*sizeof(float)>>>(d_logLuminance, d_min, numCols, numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    min_logLum = h_min;
    
    float h_max = -100;
    float* d_max;
    checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_max, &h_max, sizeof(float), cudaMemcpyHostToDevice));
    
    reduceMax<<<gridSi, blockSi, blockSi*sizeof(float)>>>(d_logLuminance, d_max, numCols, numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    max_logLum = h_max;
    
    float h_lumRange = max_logLum - min_logLum;
    float* d_lumRange;
    checkCudaErrors(cudaMalloc(&d_lumRange, sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_lumRange, &h_lumRange, sizeof(float), cudaMemcpyHostToDevice));
    
    /*
    std::cout << min << " " << min_logLum << std::endl;
    std::cout << max << " " << max_logLum << std::endl;
    std::cout << h_lumRange << std::endl;
    std::cout << " bins=" << numBins << std::endl;
    */
    
    // bin = (lum[i] - lumMin) / lumRange * numBins
    unsigned int* h_bins = (unsigned int *)malloc(numBins*sizeof(int));
    for (int i=0; i<numBins; i++)
        h_bins[i] = 0;
    
    unsigned int* d_bins;
    checkCudaErrors(cudaMalloc(&d_bins, numBins*sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_bins, h_bins, numBins*sizeof(int), cudaMemcpyHostToDevice));
    
    findBins<<<gridSi, blockSi>>>(d_logLuminance, d_bins, d_min, d_lumRange, numBins, numCols, numRows);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /*
    checkCudaErrors(cudaMemcpy(h_bins, d_bins, numBins*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "total=" << numRows*numCols << std::endl << "AFTER:" << std::endl;
    std::cout << "h_bins:" << std::endl;
    for (int i=0; i<numBins; i++)
        std::cout << h_bins[i] << std::endl;
    */
    
    //Scan
    int blockS = blockSi;
    int gridS = (numBins + blockS - 1)/blockS;
    
    /*
    int h_wrongToAdd=0;
    int* d_wrongToAdd;
    checkCudaErrors(cudaMalloc(&d_wrongToAdd, sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_wrongToAdd, &h_wrongToAdd, sizeof(int), cudaMemcpyHostToDevice));
    */
    
    inclusiveScanHist<<<gridS, blockS>>>(d_bins, d_cdf, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    /*
    int *h_cdf = (int *)malloc(numBins*sizeof(int));
    checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, numBins*sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "h_cdf:" << std::endl;
    for (int i=0; i<numBins; i++)
        std::cout << h_cdf[i] << std::endl;
    
    checkCudaErrors(cudaMemcpy(&h_wrongToAdd, d_wrongToAdd, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "Wrong? " << h_wrongToAdd << std::endl;
    */
    
    checkCudaErrors(cudaFree(d_min));
    checkCudaErrors(cudaFree(d_max));
    checkCudaErrors(cudaFree(d_bins));

}