#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 256
#define NAIVE_SWEEP 0
#define EFFICIENT_SWEEP 1

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernUpsweep(int n, int* x, int step) {
            // spaced one apart, then two apart, etc. 
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;

            int inc = 1 << (step + 1);

            #if NAIVE_SWEEP
            if (index % inc == 0) {
                x[index + inc - 1] += x[index + inc / 2 - 1];
            }
            #elif EFFICIENT_SWEEP
            int i = ((index + 1) * inc) - 1;
            x[i] += x[i - inc / 2];
            #endif
        }

        __global__ void kernDownsweep(int n, int* x, int step) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;

            int inc = 1 << (step + 1);

            #if NAIVE_SWEEP 
            if (index % inc == 0) {
                int lc = x[index + inc / 2 - 1]; 
                x[index + inc / 2 - 1] = x[index + inc - 1];
                x[index + inc - 1] += lc;
            }
            #elif EFFICIENT_SWEEP
            int i = ((index + 1) * inc) - 1;
            int lc = x[i - inc / 2];
            x[i - inc / 2] = x[i];
            x[i] += lc;
            #endif
        }

        void performScan(int n, int* x) {
            #if NAIVE_SWEEP
            int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int i = 0; i < ilog2ceil(n); ++i) {
                kernUpsweep << <numBlocks, BLOCK_SIZE >> > (n, x, i);
                checkCUDAError("kernUpsweep failed!");
            }
            cudaMemset(x + (n - 1), 0, sizeof(int));
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                kernDownsweep << <numBlocks, BLOCK_SIZE >> > (n, x, i);
                checkCUDAError("kernDownsweep failed!");
            }
            #elif EFFICIENT_SWEEP
            int numActive = n / 2;
            int numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int i = 0; i < ilog2ceil(n); ++i) {
                kernUpsweep << <numBlocks, BLOCK_SIZE >> > (numActive, x, i); 
                checkCUDAError("kernUpsweep failed!");

                numActive /= 2;
                numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }
            numActive = 1;
            numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMemset(x + (n - 1), 0, sizeof(int));
            for (int i = ilog2ceil(n) - 1; i >= 0; --i) {
                kernDownsweep << <numBlocks, BLOCK_SIZE >> > (numActive, x, i);
                checkCUDAError("kernDownsweep failed!");

                numActive *= 2;
                numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }
            #endif
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool useTimer) {
            // pad input
            int inputSize = 1 << ilog2ceil(n);
            int numBlocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // allocate memory 
            int* dev_input;
            cudaMalloc((void**)&dev_input, inputSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_input failed!");

            // copy input to device
            cudaMemset(dev_input, 0, inputSize * sizeof(int));
            checkCUDAError("cudaMemset dev_input failed!");

            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_input failed!");

            cudaDeviceSynchronize();

            // exclusive work-efficient scan
            if (useTimer) timer().startGpuTimer();
            performScan(inputSize, dev_input);
            if (useTimer) timer().endGpuTimer();

            // output result
            cudaMemcpy(odata, dev_input, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            // free memory
            cudaFree(dev_input);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // pad input
            int inputSize = 1 << ilog2ceil(n);
            int numBlocks = (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // allocate memory 
            int* dev_data;
            cudaMalloc((void**)&dev_data, inputSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            int* dev_boolArr;
            cudaMalloc((void**)&dev_boolArr, inputSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_boolArr failed!");

            int* dev_scanRes;
            cudaMalloc((void**)&dev_scanRes, inputSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_scanRes failed!");

            int* dev_compacted;
            cudaMalloc((void**)&dev_compacted, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_compacted failed!");

            // copy data to device
            cudaMemset(dev_data, 0, inputSize * sizeof(int));
            checkCUDAError("cudaMemset dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_data failed!");

            // initalize boolean array
            cudaMemset(dev_boolArr, 0, inputSize * sizeof(int));
            checkCUDAError("cudaMemset dev_boolArr failed!");

            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // build boolean array
            StreamCompaction::Common::kernMapToBoolean << <numBlocks, BLOCK_SIZE>> > (inputSize, dev_boolArr, dev_data);
            checkCUDAError("kernMapToBoolean failed!");

            // scan 
            #if NAIVE_SWEEP
            cudaMemcpy(dev_scanRes, dev_boolArr, inputSize * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int i = 0; i < ilog2ceil(inputSize); ++i) {
                kernUpsweep << <numBlocks, BLOCK_SIZE >> > (inputSize, dev_scanRes, i);
                checkCUDAError("kernUpsweep failed");
            }
            cudaMemset(dev_scanRes + (inputSize - 1), 0, sizeof(int));
            for (int i = ilog2ceil(inputSize) - 1; i >= 0; --i) {
                kernDownsweep << <numBlocks, BLOCK_SIZE >> > (inputSize, dev_scanRes, i);
                checkCUDAError("kernDownsweep failed");
            }
            #elif EFFICIENT_SWEEP
            int numActive = inputSize / 2;
            numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMemcpy(dev_scanRes, dev_boolArr, inputSize * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int i = 0; i < ilog2ceil(inputSize); ++i) {
                kernUpsweep << <numBlocks, BLOCK_SIZE >> > (numActive, dev_scanRes, i);
                checkCUDAError("kernUpsweep failed!");

                numActive /= 2;
                numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }
            numActive = 1;
            numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMemset(dev_scanRes + (inputSize - 1), 0, sizeof(int));
            for (int i = ilog2ceil(inputSize) - 1; i >= 0; --i) {
                kernDownsweep << <numBlocks, BLOCK_SIZE >> > (numActive, dev_scanRes, i);
                checkCUDAError("kernDownsweep failed!");

                numActive *= 2;
                numBlocks = (numActive + BLOCK_SIZE - 1) / BLOCK_SIZE;
            }
            #endif

            numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // scatter
            StreamCompaction::Common::kernScatter << <numBlocks, BLOCK_SIZE >> > (n, dev_compacted, dev_data, dev_boolArr, dev_scanRes);
            checkCUDAError("kernScatter failed!");
            timer().endGpuTimer();

            // figure out length
            int* host_boolArr = new int[inputSize];
            int* host_scanRes = new int[inputSize];

            cudaMemcpy(host_boolArr, dev_boolArr, inputSize * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy host_boolArr failed!");

            cudaMemcpy(host_scanRes, dev_scanRes, inputSize * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy host_scanRes failed!");

            int compactLength = host_boolArr[n - 1] + host_scanRes[n - 1];

            // output result
            cudaMemcpy(odata, dev_compacted, compactLength * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed!");
         
            // free memory
            delete[] host_boolArr;
            delete[] host_scanRes;
            cudaFree(dev_data);
            cudaFree(dev_boolArr);
            cudaFree(dev_scanRes);
            cudaFree(dev_compacted);

            return compactLength;
        }
    }
}