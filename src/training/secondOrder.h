#ifndef SECONDORDERH
#define SECONDORDERH

#include "../common/framework.h"
#include "reduce.h"
#include "memoryRoutines.h"
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

int secondOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension) {
    int size = 0;
    if (iHighCompute) { size += sizeof(float) * nDimension; } //not in cache, need to allocate new space
    if (iLowCompute) { size += sizeof(float) * nDimension; }
    return size;
}

// added by Zhu Lei
template<bool iLowCompute, bool iHighCompute, class Kernel>
__global__ void secondOrderMultiPhaseOne(bool *iLowComputeArray, bool *iHighComputeArray, float **devDataArray,
                                         int *devDataPitchInFloatsArray, float **devTransposedDataArray,
                                         int *devTransposedDataPitchInFloatsArray, float **devLabelsArray,
                                         int *nPointsArray, int *nDimensionArray, float *epsilonArray,
                                         float *cEpsilonArray, float **devAlphaArray, float **devFArray,
                                         float *alpha1DiffArray, float *alpha2DiffArray, int *iLowArray,
                                         int *iHighArray, float *parameterAArray, float *parameterBArray,
                                         float *parameterCArray, float **devCacheArray, int *devCachePitchInFloatsArray,
                                         int *iLowCacheIndexArray, int *iHighCacheIndexArray,
                                         int **devLocalIndicesRHArray, float **devLocalFsRHArray) {

    int i = blockIdx.y; // SVM_i
    extern __shared__ float xIHigh[]; //dynamic shared mem within block
    float *xILow;
    __shared__ int tempLocalIndices[BLOCKSIZE]; //static shared mem
    __shared__ float tempLocalFs[BLOCKSIZE];

    if (iHighComputeArray[i])
    {
        // if High is not cached, then Low starting addr should be end of High array
        xILow = &xIHigh[nDimensionArray[i]];
    } else {
        // if High is cached, then Low just takes the first addr of the shared mem
        xILow = xIHigh;
    }

    if (iHighComputeArray[i]) {
        //get the iHigh th point's row
        coopExtractRowVector(devTransposedDataArray[i], devTransposedDataPitchInFloatsArray[i], iHighArray[i],
                             nDimensionArray[i], xIHigh);

    }

    if (iLowComputeArray[i]) {
        coopExtractRowVector(devTransposedDataArray[i], devTransposedDataPitchInFloatsArray[i], iLowArray[i],
                             nDimensionArray[i], xILow);
    }

    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x; // data point index

    float alpha;
    float f;
    float label;
    int reduceFlag, num_of_points = nPointsArray[i];

    if (globalIndex < num_of_points) //coalesced access, possible idle threads
    {
        alpha = devAlphaArray[i][globalIndex];
        f = devFArray[i][globalIndex];
        label = devLabelsArray[i][globalIndex];
    }

    if ((globalIndex < num_of_points) &&
        (((label > 0) && (alpha < cEpsilonArray[i])) ||
         ((label < 0) && (alpha > epsilonArray[i])))) {
        reduceFlag = REDUCE0;
    } else {
        reduceFlag = NOREDUCE;
    }

    __syncthreads();

    if (globalIndex < num_of_points) {
        float highKernel;
        float lowKernel;
        if (iHighComputeArray[i]) {
            highKernel = 0;
        } else {
            highKernel = devCacheArray[i][(devCachePitchInFloatsArray[i] * iHighCacheIndexArray[i]) + globalIndex];
        }
        if (iLowComputeArray[i]) {
            lowKernel = 0;
        } else {
            lowKernel = devCacheArray[i][(devCachePitchInFloatsArray[i] * iLowCacheIndexArray[i]) + globalIndex];
        }

        if (iHighComputeArray[i] && iLowComputeArray[i]) {
            Kernel::dualKernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i],
                               devDataArray[i] + globalIndex + (devDataPitchInFloatsArray[i] * nDimensionArray[i]),
                               xIHigh, 1, xILow, 1, parameterAArray[i], parameterBArray[i], parameterCArray[i],
                               highKernel, lowKernel);
        } else if (iHighComputeArray[i]) {
            highKernel = Kernel::kernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i],
                                        devDataArray[i] + globalIndex +
                                        (devDataPitchInFloatsArray[i] * nDimensionArray[i]), xIHigh, 1,
                                        parameterAArray[i], parameterBArray[i], parameterCArray[i]);
        } else if (iLowComputeArray[i]) {
            lowKernel = Kernel::kernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i],
                                       devDataArray[i] + globalIndex +
                                       (devDataPitchInFloatsArray[i] * nDimensionArray[i]), xILow, 1,
                                       parameterAArray[i], parameterBArray[i], parameterCArray[i]);
        }

        f = f + alpha1DiffArray[i] * highKernel;
        f = f + alpha2DiffArray[i] * lowKernel;

        //if phi(i,low) is not cached
        if (iLowComputeArray[i]) {
            devCacheArray[i][(devCachePitchInFloatsArray[i] * iLowCacheIndexArray[i]) + globalIndex] = lowKernel;
        }
        //if phi(i,high) is not cached
        if (iHighComputeArray[i]) {
            devCacheArray[i][(devCachePitchInFloatsArray[i] * iHighCacheIndexArray[i]) + globalIndex] = highKernel;
        }
        devFArray[i][globalIndex] = f;
    }


    if ((reduceFlag & REDUCE0) == 0) {
        tempLocalFs[threadIdx.x] = FLT_MAX; //Ignore me
    } else {
        tempLocalFs[threadIdx.x] = f;
        tempLocalIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
//    argminReduce(tempLocalFs, tempLocalIndices);
    if (threadIdx.x == 0) {
        float *minidx = thrust::min_element(thrust::device, tempLocalFs, tempLocalFs + BLOCKSIZE);
        int argmin = minidx - tempLocalFs;
        devLocalIndicesRHArray[i][blockIdx.x] = tempLocalIndices[argmin];
        devLocalFsRHArray[i][blockIdx.x] = tempLocalFs[argmin];

//        devLocalIndicesRHArray[i][blockIdx.x] = tempLocalIndices[0];
//        devLocalFsRHArray[i][blockIdx.x] = tempLocalFs[0];
    }
}

template<bool iLowCompute, bool iHighCompute, class Kernel>
__global__ void secondOrderPhaseOne(float *devData, int devDataPitchInFloats, float *devTransposedData,
                                    int devTransposedDataPitchInFloats, float *devLabels, int nPoints, int nDimension,
                                    float epsilon, float cEpsilon, float *devAlpha, float *devF, float alpha1Diff,
                                    float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB,
                                    float parameterC, float *devCache, int devCachePitchInFloats, int iLowCacheIndex,
                                    int iHighCacheIndex, int *devLocalIndicesRH, float *devLocalFsRH) {

    extern __shared__ float xIHigh[];
    float *xILow;
    __shared__ int tempLocalIndices[BLOCKSIZE];
    __shared__ float tempLocalFs[BLOCKSIZE];

    if (iHighCompute) {
        xILow = &xIHigh[nDimension];
    } else {
        xILow = xIHigh;
    }


    if (iHighCompute) {
        coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iHigh, nDimension, xIHigh);
    }

    if (iLowCompute) {
        coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iLow, nDimension, xILow);
    }

    __syncthreads();


    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

    float alpha;
    float f;
    float label;
    int reduceFlag;

    if (globalIndex < nPoints) {
        alpha = devAlpha[globalIndex];
        f = devF[globalIndex];
        label = devLabels[globalIndex];
    }

    if ((globalIndex < nPoints) &&
        (((label > 0) && (alpha < cEpsilon)) ||
         ((label < 0) && (alpha > epsilon)))) {
        reduceFlag = REDUCE0;
    } else {
        reduceFlag = NOREDUCE;
    }

    if (globalIndex < nPoints) {
        float highKernel;
        float lowKernel;
        if (iHighCompute) {
            highKernel = 0;
        } else {
            highKernel = devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];
        }
        if (iLowCompute) {
            lowKernel = 0;
        } else {
            lowKernel = devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex];
        }

        if (iHighCompute && iLowCompute) {
            Kernel::dualKernel(devData + globalIndex, devDataPitchInFloats,
                               devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1, xILow, 1,
                               parameterA, parameterB, parameterC, highKernel, lowKernel);
        } else if (iHighCompute) {
            highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                        devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1,
                                        parameterA, parameterB, parameterC);
        } else if (iLowCompute) {
            lowKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                       devData + globalIndex + (devDataPitchInFloats * nDimension), xILow, 1,
                                       parameterA, parameterB, parameterC);
        }

        f = f + alpha1Diff * highKernel;
        f = f + alpha2Diff * lowKernel;

        if (iLowCompute) {
            devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex] = lowKernel;
        }
        if (iHighCompute) {
            devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
        }
        devF[globalIndex] = f;
    }
    __syncthreads();

    if ((reduceFlag & REDUCE0) == 0) {
        tempLocalFs[threadIdx.x] = FLT_MAX; //Ignore me
    } else {
        tempLocalFs[threadIdx.x] = f;
        tempLocalIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
    argminReduce(tempLocalFs, tempLocalIndices);
    if (threadIdx.x == 0) {
        devLocalIndicesRH[blockIdx.x] = tempLocalIndices[0];
        devLocalFsRH[blockIdx.x] = tempLocalFs[0];
    }
}


int secondOrderPhaseTwoSize() {
    int size = 0;
    return size;
}

// added by Zhu Lei
__global__ void secondOrderMultiPhaseTwo(void **devResultArray, int **devLocalIndicesRHArray, float **devLocalFsRHArray,
                                         const int *inputSizeArray) {
    int idx = blockIdx.x; //SVM number
    int tid = threadIdx.x;

    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempFs[BLOCKSIZE];

    //Load elements
    if (tid < inputSizeArray[idx]) // inputSizeArray = BlockWidthArray = # of points / BLOCKSIZE
    {
        tempIndices[tid] = devLocalIndicesRHArray[idx][tid]; // for each training block's argmin index
        tempFs[tid] = devLocalFsRHArray[idx][tid];           // for each training block's min(f)
    } else {
        tempFs[tid] = FLT_MAX;
    }

    if (inputSizeArray[idx] > BLOCKSIZE)
        // if the # of points is larger than BLOCKSIZE(256), i.e. need more threads to process the current SVM
    {
#pragma unroll
        for (int i = tid + BLOCKSIZE; i < inputSizeArray[idx]; i += blockDim.x) {
            argMin(tempIndices[tid], tempFs[tid], devLocalIndicesRHArray[idx][i],
                   devLocalFsRHArray[idx][i], tempIndices + tid, tempFs + tid);
        }
    }

    __syncthreads();
    argminReduce(tempFs, tempIndices);
    int iHigh = tempIndices[0];
    float bHigh = tempFs[0];

    if (threadIdx.x == 0) {
        *((float *) devResultArray[idx] + 3) = bHigh;
        *((int *) devResultArray[idx] + 7) = iHigh;
    }
}

__global__ void secondOrderPhaseTwo(void *devResult, int *devLocalIndicesRH, float *devLocalFsRH, int inputSize) {
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempFs[BLOCKSIZE];

    //Load elements
    if (threadIdx.x < inputSize) {
        tempIndices[threadIdx.x] = devLocalIndicesRH[threadIdx.x];
        tempFs[threadIdx.x] = devLocalFsRH[threadIdx.x];
    } else {
        tempFs[threadIdx.x] = FLT_MAX;
    }

    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
            argMin(tempIndices[threadIdx.x], tempFs[threadIdx.x], devLocalIndicesRH[i], devLocalFsRH[i],
                   tempIndices + threadIdx.x, tempFs + threadIdx.x);
        }
    }
    __syncthreads();
    argminReduce(tempFs, tempIndices);
    int iHigh = tempIndices[0];
    float bHigh = tempFs[0];

    if (threadIdx.x == 0) {
        *((float *) devResult + 3) = bHigh;
        *((int *) devResult + 7) = iHigh;
    }
}


int secondOrderPhaseThreeSize(bool iHighCompute, int nDimension) {
    int size = 0;
    if (iHighCompute) { size += sizeof(float) * nDimension; }
    return size;
}

// added by Zhu Lei
template<bool iHighCompute, class Kernel>
__global__ void
secondOrderMultiPhaseThree(bool *iHighComputeArray, float **devDataArray, int *devDataPitchInFloatsArray,
                           float **devTransposedDataArray, int *devTransposedDataPitchInFloatsArray,
                           float **devLabelsArray, float **devKernelDiagArray, float *epsilonArray,
                           float *cEpsilonArray, float **devAlphaArray, float **devFArray, float *bHighArray,
                           int *iHighArray, float **devCacheArray, int *devCachePitchInFloatsArray,
                           int *iHighCacheIndexArray, int *nDimensionArray, int *nPointsArray, float *parameterAArray,
                           float *parameterBArray, float *parameterCArray, float **devLocalFsRLArray,
                           int **devLocalIndicesMaxObjArray, float **devLocalObjsMaxObjArray) {
    int i = blockIdx.y;  //num of SVM
    extern __shared__ float xIHigh[];
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempValues[BLOCKSIZE];
    __shared__ float iHighSelfKernel;


    if (iHighComputeArray[i]) {
        //Load xIHigh into shared memory
        coopExtractRowVector(devTransposedDataArray[i], devTransposedDataPitchInFloatsArray[i], iHighArray[i],
                             nDimensionArray[i], xIHigh);
    }

    if (threadIdx.x == 0) {
        iHighSelfKernel = devKernelDiagArray[i][iHighArray[i]];
    }

    __syncthreads();


    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

    float alpha;
    float f;
    float label;
    int reduceFlag;
    float obj;

    if (globalIndex < nPointsArray[i]) {
        alpha = devAlphaArray[i][globalIndex];
        f = devFArray[i][globalIndex];
        label = devLabelsArray[i][globalIndex];

        float highKernel; //phi(I_high, cur point)

        if (iHighComputeArray[i]) {
            highKernel = Kernel::kernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i],
                                        devDataArray[i] + globalIndex +
                                        (devDataPitchInFloatsArray[i] * nDimensionArray[i]), xIHigh, 1,
                                        parameterAArray[i], parameterBArray[i], parameterCArray[i]);
            devCacheArray[i][(devCachePitchInFloatsArray[i] * iHighCacheIndexArray[i]) + globalIndex] = highKernel;
        } else {
            highKernel = devCacheArray[i][(devCachePitchInFloatsArray[i] * iHighCacheIndexArray[i]) + globalIndex];
        }


        float beta = bHighArray[i] - f;

        float kappa = iHighSelfKernel + devKernelDiagArray[i][globalIndex] - 2 * highKernel; //eta

        if (kappa <= 0) {
            kappa = epsilonArray[i];
        }

        obj = beta * beta / kappa;
        if (((label > 0) && (alpha > epsilonArray[i])) ||
            ((label < 0) && (alpha < cEpsilonArray[i]))) {
            if (beta <= epsilonArray[i]) {
                reduceFlag = REDUCE1 | REDUCE0;
            } else {
                reduceFlag = REDUCE0;
            }
        } else {
            reduceFlag = NOREDUCE;
        }
    } else {
        reduceFlag = NOREDUCE;
    }

    if ((reduceFlag & REDUCE0) == 0) {
        tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
    } else {
        tempValues[threadIdx.x] = f;
    }

    __syncthreads();

    maxReduce(tempValues);
    if (threadIdx.x == 0) {
        devLocalFsRLArray[i][blockIdx.x] = tempValues[0];
    }

    if ((reduceFlag & REDUCE1) == 0) {
        tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
        tempIndices[threadIdx.x] = 0;
    } else {
        tempValues[threadIdx.x] = obj;
        tempIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
    argmaxReduce(tempValues, tempIndices);

    if (threadIdx.x == 0) {
        devLocalIndicesMaxObjArray[i][blockIdx.x] = tempIndices[0];
        devLocalObjsMaxObjArray[i][blockIdx.x] = tempValues[0];
    }
}

template<bool iHighCompute, class Kernel>
__global__ void secondOrderPhaseThree(float *devData, int devDataPitchInFloats, float *devTransposedData,
                                      int devTransposedDataPitchInFloats, float *devLabels, float *devKernelDiag,
                                      float epsilon, float cEpsilon, float *devAlpha, float *devF, float bHigh,
                                      int iHigh, float *devCache, int devCachePitchInFloats, int iHighCacheIndex,
                                      int nDimension, int nPoints, float parameterA, float parameterB, float parameterC,
                                      float *devLocalFsRL, int *devLocalIndicesMaxObj, float *devLocalObjsMaxObj) {
    extern __shared__ float xIHigh[];
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempValues[BLOCKSIZE];
    __shared__ float iHighSelfKernel;


    if (iHighCompute) {
        //Load xIHigh into shared memory
        coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iHigh, nDimension, xIHigh);
    }

    if (threadIdx.x == 0) {
        iHighSelfKernel = devKernelDiag[iHigh];
    }

    __syncthreads();


    int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

    float alpha;
    float f;
    float label;
    int reduceFlag;
    float obj;

    if (globalIndex < nPoints) {
        alpha = devAlpha[globalIndex];

        f = devF[globalIndex];
        label = devLabels[globalIndex];

        float highKernel;

        if (iHighCompute) {
            highKernel = 0;
        } else {
            highKernel = devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];
        }

        if (iHighCompute) {
            highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats,
                                        devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1,
                                        parameterA, parameterB, parameterC);
            devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex] = highKernel;
        }


        float beta = bHigh - f;

        float kappa = iHighSelfKernel + devKernelDiag[globalIndex] - 2 * highKernel;

        if (kappa <= 0) {
            kappa = epsilon;
        }

        obj = beta * beta / kappa;
        if (((label > 0) && (alpha > epsilon)) ||
            ((label < 0) && (alpha < cEpsilon))) {
            if (beta <= epsilon) {
                reduceFlag = REDUCE1 | REDUCE0;
            } else {
                reduceFlag = REDUCE0;
            }
        } else {
            reduceFlag = NOREDUCE;
        }
    } else {
        reduceFlag = NOREDUCE;
    }

    if ((reduceFlag & REDUCE0) == 0) {
        tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
    } else {
        tempValues[threadIdx.x] = f;
    }

    __syncthreads();

    maxReduce(tempValues);
    if (threadIdx.x == 0) {
        devLocalFsRL[blockIdx.x] = tempValues[0];
    }

    if ((reduceFlag & REDUCE1) == 0) {
        tempValues[threadIdx.x] = -FLT_MAX; //Ignore me
        tempIndices[threadIdx.x] = 0;
    } else {
        tempValues[threadIdx.x] = obj;
        tempIndices[threadIdx.x] = globalIndex;
    }
    __syncthreads();
    argmaxReduce(tempValues, tempIndices);

    if (threadIdx.x == 0) {
        devLocalIndicesMaxObj[blockIdx.x] = tempIndices[0];
        devLocalObjsMaxObj[blockIdx.x] = tempValues[0];
    }
}


int secondOrderPhaseFourSize() {
    int size = 0;
    return size;
}

// added by Zhu Lei
__global__ void
secondOrderMultiPhaseFour(float **devLabelsArray, float **devKernelDiagArray, float **devFArray, float **devAlphaArray,
                          float *costArray, int *iHighArray, float *bHighArray, void **devResultArray,
                          float **devCacheArray, int *devCachePitchInFloatsArray, int *iHighCacheIndexArray,
                          float **devLocalFsRLArray, int **devLocalIndicesMaxObjArray, float **devLocalObjsMaxObjArray,
                          int *inputSizeArray, int iteration) {
    int idx = blockIdx.x;
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempValues[BLOCKSIZE];


    if (threadIdx.x < inputSizeArray[idx]) {
        tempIndices[threadIdx.x] = devLocalIndicesMaxObjArray[idx][threadIdx.x];
        tempValues[threadIdx.x] = devLocalObjsMaxObjArray[idx][threadIdx.x];
    } else {
        tempValues[threadIdx.x] = -FLT_MAX;
        tempIndices[threadIdx.x] = -1;
    }

    if (inputSizeArray[idx] > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSizeArray[idx]; i += blockDim.x) {
            argMax(tempIndices[threadIdx.x], tempValues[threadIdx.x], devLocalIndicesMaxObjArray[idx][i],
                   devLocalObjsMaxObjArray[idx][i], tempIndices + threadIdx.x, tempValues + threadIdx.x);
        }
    }
    __syncthreads();


    argmaxReduce(tempValues, tempIndices);

    __syncthreads();
    int iLow;
    if (threadIdx.x == 0) {
        iLow = tempIndices[0];
    }

    if (threadIdx.x < inputSizeArray[idx]) {
        tempValues[threadIdx.x] = devLocalFsRLArray[idx][threadIdx.x];
    } else {
        tempValues[threadIdx.x] = -FLT_MAX;
    }

    if (inputSizeArray[idx] > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSizeArray[idx]; i += blockDim.x) {
            maxOperator(tempValues[threadIdx.x], devLocalFsRLArray[idx][i], tempValues + threadIdx.x);
        }
    }
    __syncthreads();
    maxReduce(tempValues);
    __syncthreads();
    float bLow;
    if (threadIdx.x == 0) {
        bLow = tempValues[0];
    }


    if (threadIdx.x == 0) {

        float eta = devKernelDiagArray[idx][iHighArray[idx]] + devKernelDiagArray[idx][iLow];

        eta = eta - 2 * (*(devCacheArray[idx] + (devCachePitchInFloatsArray[idx] * iHighCacheIndexArray[idx]) + iLow));


        float alpha1Old = devAlphaArray[idx][iHighArray[idx]];
        float alpha2Old = devAlphaArray[idx][iLow];
        float alphaDiff = alpha2Old - alpha1Old;
        float lowLabel = devLabelsArray[idx][iLow];
        float sign = devLabelsArray[idx][iHighArray[idx]] * lowLabel;
        float alpha2UpperBound;
        float alpha2LowerBound;
        if (sign < 0) {
            if (alphaDiff < 0) {
                alpha2LowerBound = 0;
                alpha2UpperBound = costArray[idx] + alphaDiff;
            } else {
                alpha2LowerBound = alphaDiff;
                alpha2UpperBound = costArray[idx];
            }
        } else {
            float alphaSum = alpha2Old + alpha1Old;
            if (alphaSum < costArray[idx]) {
                alpha2UpperBound = alphaSum;
                alpha2LowerBound = 0;
            } else {
                alpha2LowerBound = alphaSum - costArray[idx];
                alpha2UpperBound = costArray[idx];
            }
        }
        float alpha2New;
        if (eta > 0) {
            alpha2New = alpha2Old + lowLabel * (devFArray[idx][iHighArray[idx]] - devFArray[idx][iLow]) / eta;
            if (alpha2New < alpha2LowerBound) {
                alpha2New = alpha2LowerBound;
            } else if (alpha2New > alpha2UpperBound) {
                alpha2New = alpha2UpperBound;
            }
        } else {
            float slope = lowLabel * (bHighArray[idx] - bLow);
            float delta = slope * (alpha2UpperBound - alpha2LowerBound);
            if (delta > 0) {
                if (slope > 0) {
                    alpha2New = alpha2UpperBound;
                } else {
                    alpha2New = alpha2LowerBound;
                }
            } else {
                alpha2New = alpha2Old;
            }
        }
        float alpha2Diff = alpha2New - alpha2Old;
        float alpha1Diff = -sign * alpha2Diff;
        float alpha1New = alpha1Old + alpha1Diff;
        devAlphaArray[idx][iLow] = alpha2New;
        devAlphaArray[idx][iHighArray[idx]] = alpha1New;

        *((float *) devResultArray[idx] + 0) = alpha2Old;
        *((float *) devResultArray[idx] + 1) = alpha1Old;
        *((float *) devResultArray[idx] + 2) = bLow;
        *((float *) devResultArray[idx] + 3) = bHighArray[idx];
        *((float *) devResultArray[idx] + 4) = alpha2New;
        *((float *) devResultArray[idx] + 5) = alpha1New;
        *((int *) devResultArray[idx] + 6) = iLow;
        *((int *) devResultArray[idx] + 7) = iHighArray[idx];
    }
}

__global__ void
secondOrderPhaseFour(float *devLabels, float *devKernelDiag, float *devF, float *devAlpha, float cost, int iHigh,
                     float bHigh, void *devResult, float *devCache, int devCachePitchInFloats, int iHighCacheIndex,
                     float *devLocalFsRL, int *devLocalIndicesMaxObj, float *devLocalObjsMaxObj, int inputSize,
                     int iteration) {
    __shared__ int tempIndices[BLOCKSIZE];
    __shared__ float tempValues[BLOCKSIZE];


    if (threadIdx.x < inputSize) {
        tempIndices[threadIdx.x] = devLocalIndicesMaxObj[threadIdx.x];
        tempValues[threadIdx.x] = devLocalObjsMaxObj[threadIdx.x];
    } else {
        tempValues[threadIdx.x] = -FLT_MAX;
        tempIndices[threadIdx.x] = -1;
    }

    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
            argMax(tempIndices[threadIdx.x], tempValues[threadIdx.x], devLocalIndicesMaxObj[i], devLocalObjsMaxObj[i],
                   tempIndices + threadIdx.x, tempValues + threadIdx.x);
        }
    }
    __syncthreads();


    argmaxReduce(tempValues, tempIndices);

    __syncthreads();
    int iLow;
    if (threadIdx.x == 0) {
        iLow = tempIndices[0];
        /* int temp = tempIndices[0]; */
/*     if (temp < 0) { */
/*       iLow = 17; */
/*     } else if (temp > 4176) { */
/*       iLow = 18; */
/*     } else { */
/*       iLow = temp; */
/*     } */

    }

    /* if (iteration > 1721) { */
/*     if (threadIdx.x == 0) { */
/*       //\*((float*)devResult + 0) = devAlpha[iLow]; */
/*       //\*((float*)devResult + 1) = devAlpha[iHigh]; */
/*       //\*((float*)devResult + 2) = bLow; */
/*       //\*((float*)devResult + 3) = bHigh; */

/*       *((int*)devResult + 6) = iLow; */
/*       *((int*)devResult + 7) = iHigh; */
/*     } */
/*     return; */
/*   } */

    if (threadIdx.x < inputSize) {
        tempValues[threadIdx.x] = devLocalFsRL[threadIdx.x];
    } else {
        tempValues[threadIdx.x] = -FLT_MAX;
    }

    if (inputSize > BLOCKSIZE) {
        for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
            maxOperator(tempValues[threadIdx.x], devLocalFsRL[i], tempValues + threadIdx.x);
        }
    }
    __syncthreads();
    maxReduce(tempValues);
    __syncthreads();
    float bLow;
    if (threadIdx.x == 0) {
        bLow = tempValues[0];
    }


    if (threadIdx.x == 0) {

        float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];

        eta = eta - 2 * (*(devCache + (devCachePitchInFloats * iHighCacheIndex) + iLow));


        float alpha1Old = devAlpha[iHigh];
        float alpha2Old = devAlpha[iLow];
        float alphaDiff = alpha2Old - alpha1Old;
        float lowLabel = devLabels[iLow];
        float sign = devLabels[iHigh] * lowLabel;
        float alpha2UpperBound;
        float alpha2LowerBound;
        if (sign < 0) {
            if (alphaDiff < 0) {
                alpha2LowerBound = 0;
                alpha2UpperBound = cost + alphaDiff;
            } else {
                alpha2LowerBound = alphaDiff;
                alpha2UpperBound = cost;
            }
        } else {
            float alphaSum = alpha2Old + alpha1Old;
            if (alphaSum < cost) {
                alpha2UpperBound = alphaSum;
                alpha2LowerBound = 0;
            } else {
                alpha2LowerBound = alphaSum - cost;
                alpha2UpperBound = cost;
            }
        }
        float alpha2New;
        if (eta > 0) {
            alpha2New = alpha2Old + lowLabel * (devF[iHigh] - devF[iLow]) / eta;
            if (alpha2New < alpha2LowerBound) {
                alpha2New = alpha2LowerBound;
            } else if (alpha2New > alpha2UpperBound) {
                alpha2New = alpha2UpperBound;
            }
        } else {
            float slope = lowLabel * (bHigh - bLow);
            float delta = slope * (alpha2UpperBound - alpha2LowerBound);
            if (delta > 0) {
                if (slope > 0) {
                    alpha2New = alpha2UpperBound;
                } else {
                    alpha2New = alpha2LowerBound;
                }
            } else {
                alpha2New = alpha2Old;
            }
        }
        float alpha2Diff = alpha2New - alpha2Old;
        float alpha1Diff = -sign * alpha2Diff;
        float alpha1New = alpha1Old + alpha1Diff;
        devAlpha[iLow] = alpha2New;
        devAlpha[iHigh] = alpha1New;

        *((float *) devResult + 0) = alpha2Old;
        *((float *) devResult + 1) = alpha1Old;
        *((float *) devResult + 2) = bLow;
        *((float *) devResult + 3) = bHigh;
        *((float *) devResult + 4) = alpha2New;
        *((float *) devResult + 5) = alpha1New;
        *((int *) devResult + 6) = iLow;
        *((int *) devResult + 7) = iHigh;
    }
}

// added by Zhu Lei
void
launchMultiSecondOrder(const int num_svm, bool *iLowComputeArray, bool *iHighComputeArray, int kType, int *nPointsArray,
                       int *nDimensionArray, int *devnDimensionArray, dim3 blocksConfig, dim3 threadsConfig,
                       dim3 globalThreadsConfig, float **devDataArray, int *devDataPitchInFloatsArray,
                       float **devTransposedDataArray, int *devTransposedDataPitchInFloatsArray,
                       float **devLabelsArray, float *epsilonArray, float *cEpsilonArray, float **devAlphaArray,
                       float **devFArray, float *sAlpha1DiffArray, float *sAlpha2DiffArray, int *iLowArray,
                       int *iHighArray, float *parameterAArray, float *parameterBArray, float *parameterCArray,
                       Cache **kernelCacheArray, float **devCacheArray, int *devCachePitchInFloatsArray,
                       int *iLowCacheIndexArray, int *iHighCacheIndexArray, int **devLocalIndicesRHArray,
                       float **devLocalFsRHArray, float **devLocalFsRLArray, int **devLocalIndicesMaxObjArray,
                       float **devLocalObjsMaxObjArray, float **devKernelDiagArray, void **devResultArray,
                       float **hostResultArray, float *costArray, int iteration, int *blockWidthArray,
                       bool *deviLowComputeArray, bool *deviHighComputeArray, int *deviHighArray,
                       int *deviHighCacheIndexArray, void **h_devResultArray) {
    int i;
    int phaseOneSize = 0;
    for (i = 0; i < num_svm; i++) {
        int temp = secondOrderPhaseOneSize(iLowComputeArray[i], iHighComputeArray[i], nDimensionArray[i]);
        if (phaseOneSize < temp) {
            phaseOneSize = temp;
        }
    }
    int phaseTwoSize = 0;
    for (i = 0; i < num_svm; i++) {
        int temp = secondOrderPhaseTwoSize();
        if (phaseTwoSize < temp) {
            phaseTwoSize = temp;
        }
    }

    if (iLowComputeArray[0] == true) {
        if (iHighComputeArray[0] == true) {
            //both not in cache
            switch (kType) {
                case LINEAR:
                    secondOrderMultiPhaseOne<true, true, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                   (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case POLYNOMIAL:
                    secondOrderMultiPhaseOne<true, true, Polynomial> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                       (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case GAUSSIAN:
                    secondOrderMultiPhaseOne<true, true, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                     (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case SIGMOID:
                    secondOrderMultiPhaseOne<true, true, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                    (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
            }
        } else if (iHighComputeArray[0] == false) {
            //Low in cache, high not in cache
            switch (kType) {
                case LINEAR:
                    secondOrderMultiPhaseOne<true, false, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                    (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case POLYNOMIAL:
                    secondOrderMultiPhaseOne<true, false, Polynomial> << < blocksConfig, threadsConfig,
                            phaseOneSize >> >
                            (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case GAUSSIAN:
                    secondOrderMultiPhaseOne<true, false, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                      (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case SIGMOID:
                    secondOrderMultiPhaseOne<true, false, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                     (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
            }

        }
    } else if (iLowComputeArray[0] == false) {
        if (iHighComputeArray[0] == true) {
            switch (kType) {
                case LINEAR:
                    secondOrderMultiPhaseOne<false, true, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                    (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case POLYNOMIAL:
                    secondOrderMultiPhaseOne<false, true, Polynomial> << < blocksConfig, threadsConfig,
                            phaseOneSize >> >
                            (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case GAUSSIAN:
                    secondOrderMultiPhaseOne<false, true, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                      (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case SIGMOID:
                    secondOrderMultiPhaseOne<false, true, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                     (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
            }
        } else if (iHighComputeArray[0] == false) {
            //both in cache
            switch (kType) {
                case LINEAR:
                    secondOrderMultiPhaseOne<false, false, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                     (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case POLYNOMIAL:
                    secondOrderMultiPhaseOne<false, false, Polynomial> << < blocksConfig, threadsConfig,
                            phaseOneSize >> >
                            (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case GAUSSIAN:
                    secondOrderMultiPhaseOne<false, false, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                       (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
                case SIGMOID:
                    secondOrderMultiPhaseOne<false, false, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                      (deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, deviHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, deviHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray);
                    break;
            }

        }
    }

    //num_svm block, each with BLOCKSIZE threads
    secondOrderMultiPhaseTwo << < num_svm, globalThreadsConfig, phaseTwoSize >> >
                                                                (devResultArray, devLocalIndicesRHArray, devLocalFsRHArray, blockWidthArray);

    float *bHighArray = (float *) malloc(sizeof(float) * num_svm);
    int phaseThreeSize = 0;

#pragma unroll
    for (i = 0; i < num_svm; i++) {
        checkCudaErrors(cudaMemcpy((void *) (hostResultArray[i]), h_devResultArray[i], 8 * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        bHighArray[i] = *(hostResultArray[i] + 3);
        iHighArray[i] = *((int *) hostResultArray[i] + 7);
        kernelCacheArray[i]->findData(iHighArray[i], iHighCacheIndexArray[i], iHighComputeArray[i]);
        phaseThreeSize = max(phaseThreeSize, secondOrderPhaseThreeSize(iHighComputeArray[i], nDimensionArray[i]));
    }


//    for (i = 0; i < num_svm; i++) {
//        //float eta = *(hostResultArray);
//        bHighArray[i] = *(hostResultArray[i] + 3);
//        iHighArray[i] = *((int *) hostResultArray[i] + 7);
//
//        kernelCacheArray[i]->findData(iHighArray[i], iHighCacheIndexArray[i], iHighComputeArray[i]);
//    }


//    for (i = 0; i < num_svm; i++) {
//        int temp = secondOrderPhaseThreeSize(iHighComputeArray[i], nDimensionArray[i]);
//        if (phaseThreeSize < temp) phaseThreeSize = temp;
//    }

    checkCudaErrors(cudaMemcpy((void *) deviHighCacheIndexArray, (void *) iHighCacheIndexArray, num_svm * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *) deviHighComputeArray, (void *) iHighComputeArray, num_svm * sizeof(bool),
                               cudaMemcpyHostToDevice));


    checkCudaErrors(
            cudaMemcpy((void *) deviHighArray, (void *) iHighArray, num_svm * sizeof(int), cudaMemcpyHostToDevice));
    float *devbHighArray;
    checkCudaErrors(cudaMalloc((void **) (&devbHighArray), num_svm * sizeof(float)));
    checkCudaErrors(
            cudaMemcpy((void *) devbHighArray, (void *) bHighArray, num_svm * sizeof(float), cudaMemcpyHostToDevice));


    if (iHighComputeArray[0] == true) {
        switch (kType) {
            case LINEAR:
                secondOrderMultiPhaseThree<true, Linear> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                           (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
            case POLYNOMIAL:
                secondOrderMultiPhaseThree<true, Polynomial> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                               (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
            case GAUSSIAN:
                secondOrderMultiPhaseThree<true, Gaussian> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                             (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
            case SIGMOID:
                secondOrderMultiPhaseThree<true, Sigmoid> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                            (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
        }
    } else {
        switch (kType) {
            case LINEAR:
                secondOrderMultiPhaseThree<false, Linear> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                            (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
            case POLYNOMIAL:
                secondOrderMultiPhaseThree<false, Polynomial> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                                (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
            case GAUSSIAN:
                secondOrderMultiPhaseThree<false, Gaussian> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                              (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
            case SIGMOID:
                secondOrderMultiPhaseThree<false, Sigmoid> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                             (deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, devbHighArray, deviHighArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devnDimensionArray, nPointsArray, parameterAArray, parameterBArray, parameterCArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray);
                break;
        }
    }
    /* if (iteration == 1722) { */
/*    float* localObjsMaxObj = (float*)malloc(blocksConfig.x * sizeof(float)); */
/*    int* localIndicesMaxObj = (int*)malloc(blocksConfig.x * sizeof(int)); */
/*    cudaMemcpy(localObjsMaxObj, devLocalObjsMaxObjArray, sizeof(float) * blocksConfig.x, cudaMemcpyDeviceToHost); */
/*    cudaMemcpy(localIndicesMaxObj, devLocalIndicesMaxObjArray, sizeof(int) * blocksConfig.x, cudaMemcpyDeviceToHost); */
/*    for(int i = 0; i < blocksConfig.x; i++) { */
/*      printf("(%i: %f)\n", localIndicesMaxObj[i], localObjsMaxObj[i]); */
/*    } */
/*    free(localObjsMaxObj); */
/*    free(localIndicesMaxObj); */
/*    }       */
    secondOrderMultiPhaseFour << < num_svm, globalThreadsConfig, phaseTwoSize >> >
                                                                 (devLabelsArray, devKernelDiagArray, devFArray, devAlphaArray, costArray, deviHighArray, devbHighArray, devResultArray, devCacheArray, devCachePitchInFloatsArray, deviHighCacheIndexArray, devLocalFsRLArray, devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray, blockWidthArray, iteration);
    cudaDeviceSynchronize();
    free(bHighArray);
    cudaFree(devbHighArray);
}

void launchSecondOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, dim3 blocksConfig,
                       dim3 threadsConfig, dim3 globalThreadsConfig, float *devData, int devDataPitchInFloats,
                       float *devTransposedData, int devTransposedDataPitchInFloats, float *devLabels, float epsilon,
                       float cEpsilon, float *devAlpha, float *devF, float sAlpha1Diff, float sAlpha2Diff, int iLow,
                       int iHigh, float parameterA, float parameterB, float parameterC, Cache *kernelCache,
                       float *devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex,
                       int *devLocalIndicesRH, float *devLocalFsRH, float *devLocalFsRL, int *devLocalIndicesMaxObj,
                       float *devLocalObjsMaxObj, float *devKernelDiag, void *devResult, float *hostResult, float cost,
                       int iteration) {

    int phaseOneSize = secondOrderPhaseOneSize(iLowCompute, iHighCompute, nDimension);
    int phaseTwoSize = secondOrderPhaseTwoSize();

    if (iLowCompute == true) {
        if (iHighCompute == true) {
            switch (kType) {
                case LINEAR:
                    secondOrderPhaseOne<true, true, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                              (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case POLYNOMIAL:
                    secondOrderPhaseOne<true, true, Polynomial> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                  (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case GAUSSIAN:
                    secondOrderPhaseOne<true, true, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case SIGMOID:
                    secondOrderPhaseOne<true, true, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                               (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
            }
        } else if (iHighCompute == false) {
            switch (kType) {
                case LINEAR:
                    secondOrderPhaseOne<true, false, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                               (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case POLYNOMIAL:
                    secondOrderPhaseOne<true, false, Polynomial> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                   (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case GAUSSIAN:
                    secondOrderPhaseOne<true, false, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                 (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case SIGMOID:
                    secondOrderPhaseOne<true, false, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
            }

        }
    } else if (iLowCompute == false) {
        if (iHighCompute == true) {
            switch (kType) {
                case LINEAR:
                    secondOrderPhaseOne<false, true, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                               (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case POLYNOMIAL:
                    secondOrderPhaseOne<false, true, Polynomial> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                   (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case GAUSSIAN:
                    secondOrderPhaseOne<false, true, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                 (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case SIGMOID:
                    secondOrderPhaseOne<false, true, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
            }
        } else if (iHighCompute == false) {
            switch (kType) {
                case LINEAR:
                    secondOrderPhaseOne<false, false, Linear> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case POLYNOMIAL:
                    secondOrderPhaseOne<false, false, Polynomial> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                    (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case GAUSSIAN:
                    secondOrderPhaseOne<false, false, Gaussian> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                  (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
                case SIGMOID:
                    secondOrderPhaseOne<false, false, Sigmoid> << < blocksConfig, threadsConfig, phaseOneSize >> >
                                                                                                 (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRH, devLocalFsRH);
                    break;
            }

        }
    }


    secondOrderPhaseTwo << < 1, globalThreadsConfig, phaseTwoSize >> >
                                                     (devResult, devLocalIndicesRH, devLocalFsRH, blocksConfig.x);


    checkCudaErrors(cudaMemcpy((void *) hostResult, devResult, 8 * sizeof(float), cudaMemcpyDeviceToHost));

    //float eta = *(hostResult);
    float bHigh = *(hostResult + 3);
    iHigh = *((int *) hostResult + 7);

    kernelCache->findData(iHigh, iHighCacheIndex, iHighCompute);

    int phaseThreeSize = secondOrderPhaseThreeSize(iHighCompute, nDimension);


    if (iHighCompute == true) {
        switch (kType) {
            case LINEAR:
                secondOrderPhaseThree<true, Linear> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                      (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
            case POLYNOMIAL:
                secondOrderPhaseThree<true, Polynomial> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                          (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
            case GAUSSIAN:
                secondOrderPhaseThree<true, Gaussian> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                        (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
            case SIGMOID:
                secondOrderPhaseThree<true, Sigmoid> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                       (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
        }
    } else {
        switch (kType) {
            case LINEAR:
                secondOrderPhaseThree<false, Linear> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                       (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
            case POLYNOMIAL:
                secondOrderPhaseThree<false, Polynomial> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                           (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
            case GAUSSIAN:
                secondOrderPhaseThree<false, Gaussian> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                         (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
            case SIGMOID:
                secondOrderPhaseThree<false, Sigmoid> << < blocksConfig, threadsConfig, phaseThreeSize >> >
                                                                                        (devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, epsilon, cEpsilon, devAlpha, devF, bHigh, iHigh, devCache, devCachePitchInFloats, iHighCacheIndex, nDimension, nPoints, parameterA, parameterB, parameterC, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj);
                break;
        }
    }
    /* if (iteration == 1722) { */
/*    float* localObjsMaxObj = (float*)malloc(blocksConfig.x * sizeof(float)); */
/*    int* localIndicesMaxObj = (int*)malloc(blocksConfig.x * sizeof(int)); */
/*    cudaMemcpy(localObjsMaxObj, devLocalObjsMaxObj, sizeof(float) * blocksConfig.x, cudaMemcpyDeviceToHost); */
/*    cudaMemcpy(localIndicesMaxObj, devLocalIndicesMaxObj, sizeof(int) * blocksConfig.x, cudaMemcpyDeviceToHost); */
/*    for(int i = 0; i < blocksConfig.x; i++) { */
/*      printf("(%i: %f)\n", localIndicesMaxObj[i], localObjsMaxObj[i]); */
/*    } */
/*    free(localObjsMaxObj); */
/*    free(localIndicesMaxObj); */
/*    }       */

    secondOrderPhaseFour << < 1, globalThreadsConfig, phaseTwoSize >> >
                                                      (devLabels, devKernelDiag, devF, devAlpha, cost, iHigh, bHigh, devResult, devCache, devCachePitchInFloats, iHighCacheIndex, devLocalFsRL, devLocalIndicesMaxObj, devLocalObjsMaxObj, blocksConfig.x, iteration);

}


#endif
