#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaive(int n, int* odata, int* idata, int step) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index >= n) return;

            int inc = 1 << (step - 1);

            if (index >= inc) {
                odata[index] = idata[index - inc] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // allocate memory
            int* dev_input1;
            cudaMalloc((void**)&dev_input1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_input1 failed!");

            int* dev_input2;
            cudaMalloc((void**)&dev_input2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_input2 failed!");

            // copy input data to device
            cudaMemcpy(dev_input2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy dev_input2 failed!");

            cudaDeviceSynchronize();

            timer().startGpuTimer();
            // peform inclusive scan
            int numPasses = ilog2ceil(n);
            int currentPass = 1;

            while (currentPass <= numPasses) {
                // separate kernel launches
                kernNaive << <numBlocks, BLOCK_SIZE >> > (n, dev_input1, dev_input2, currentPass);
                checkCUDAError("kernNaive failed!");
                std::swap(dev_input1, dev_input2);
                currentPass += 1;
            }
            timer().endGpuTimer();

            // convert inclusive scan to exclusive scan
            int* toExclusive = new int[n];
            cudaMemcpy(toExclusive, dev_input2, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy toExclusive failed!");

            for (int i = n - 1; i > 0; --i) {
                toExclusive[i] = toExclusive[i - 1];
            }
            toExclusive[0] = 0;

            // output result
            cudaMemcpy(odata, toExclusive, n * sizeof(int), cudaMemcpyHostToHost);
            checkCUDAError("cudaMemcpy odata failed!");

            // free memory
            delete[] toExclusive;
            cudaFree(dev_input1);
            cudaFree(dev_input2);
        }
    }
}
