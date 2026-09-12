#include <cstdio>
#include "cpu.h"

#include "common.h"

#include <vector>

/*
* `StreamCompaction::CPU::scan`: compute an exclusive prefix sum. For performance comparison, this is supposed to be a simple `for` loop. But for better understanding before starting moving to GPU, you can simulate the GPU scan in this function first.
* `StreamCompaction::CPU::compactWithoutScan`: stream compaction without using
  the `scan` function.
* `StreamCompaction::CPU::compactWithScan`: stream compaction using the `scan`
  function. Map the input array to an array of 0s and 1s, scan it, and use
  scatter to produce the output. You will need a **CPU** scatter implementation
  for this (see slides or GPU Gems chapter for an explanation).
*/

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata, bool useTimer) {
            if (useTimer) timer().startCpuTimer();
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
            }
            if (useTimer) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int oidx = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[oidx] = idata[i];
                    oidx += 1;
                }
            }
            timer().endCpuTimer();
            return oidx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // temporary arrays 
            int* boolArr = new int[n];
            int* scanRes = new int[n];
            // construct temp array of 0s and 1s
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    boolArr[i] = 1;
                }
                else {
                    boolArr[i] = 0;
                }
            }
            // perform exclusive scan on boolean array
            scan(n, scanRes, boolArr, false);
            // scatter
            int newLength = 0;
            for (int i = 0; i < n; ++i) {
                if (boolArr[i] == 1) {
                    odata[scanRes[i]] = idata[i];
                    newLength += 1;
                }
            }
            // delete temp arrays
            delete[] boolArr;
            delete[] scanRes;
            timer().endCpuTimer();
            return newLength;
        }
    }
}
