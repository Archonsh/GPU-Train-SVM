#ifndef FIRSTORDER_H
#define FIRSTORDER_H
#include "reduce.h"
#include "memoryRoutines.h"
#include <float.h>

int firstOrderPhaseOneSize(bool iLowCompute, bool iHighCompute, int nDimension) {
	int size = 0;
	if (iHighCompute) { size+= sizeof(float) * nDimension; }
	if (iLowCompute) { size+= sizeof(float) * nDimension; }
	return size;
}


int firstOrderPhaseTwoSize() {
	int size = 0;
	return size;
}

// added by Zhu Lei
template<bool iLowCompute, bool iHighCompute, class Kernel>
  __global__ void firstOrderMultiPhaseOne(bool* iLowComputeArray, bool* iHighComputeArray, float** devDataArray, int* devDataPitchInFloatsArray, float** devTransposedDataArray, int* devTransposedDataPitchInFloatsArray, float** devLabelsArray, int* nPointsArray, int* nDimensionArray, float* epsilonArray, float* cEpsilonArray, float** devAlphaArray, float** devFArray, float* alpha1DiffArray, float* alpha2DiffArray, int* iLowArray, int* iHighArray, float* parameterAArray, float* parameterBArray, float* parameterCArray, float** devCacheArray, int* devCachePitchInFloatsArray, int* iLowCacheIndexArray, int* iHighCacheIndexArray, int** devLocalIndicesRLArray, int** devLocalIndicesRHArray, float** devLocalFsRLArray, float** devLocalFsRHArray) {


  int i = blockIdx.y;
  extern __shared__ float xIHigh[];
  float* xILow;
  __shared__ int tempLocalIndices[BLOCKSIZE];
  __shared__ float tempLocalFs[BLOCKSIZE];
  
  
  if (iHighComputeArray[i]) {
    xILow = &xIHigh[nDimensionArray[i]];
  } else {
    xILow = xIHigh;
  }

  
  if (iHighComputeArray[i]) {
    //Load xIHigh into shared memory
    coopExtractRowVector(devTransposedDataArray[i], devTransposedDataPitchInFloatsArray[i], iHighArray[i], nDimensionArray[i], xIHigh);
  }

  if (iLowComputeArray[i]) {
    //Load xILow into shared memory
    coopExtractRowVector(devTransposedDataArray[i], devTransposedDataPitchInFloatsArray[i], iLowArray[i], nDimensionArray[i], xILow);
  }
  
  
  __syncthreads();
 


  
  int globalIndex = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;

  float alpha;
  float f;
  float label;
  int reduceFlag;
  
  if (globalIndex < nPointsArray[i]) {
    alpha = devAlphaArray[i][globalIndex];

    f = devFArray[i][globalIndex];
    label = devLabelsArray[i][globalIndex];
    
    
    if (alpha > epsilonArray[i]) {
      if (alpha < cEpsilonArray[i]) {
        reduceFlag = REDUCE0 | REDUCE1; //Unbound support vector (I0)
      } else {
        if (label > 0) {
          reduceFlag = REDUCE0; //Bound positive support vector (I3)
        } else {
          reduceFlag = REDUCE1; //Bound negative support vector (I2)
        }
      }
    } else {
      if (label > 0) {
        reduceFlag = REDUCE1; //Positive nonsupport vector (I1)
      } else {
        reduceFlag = REDUCE0; //Negative nonsupport vector (I4)
      }
    }
  } else {
    reduceFlag = NOREDUCE;
  }

  
  float highKernel = 0;
  float lowKernel = 0;
  if (reduceFlag > 0) {
    if (!iHighComputeArray[i]) {
      highKernel = devCacheArray[i][(devCachePitchInFloatsArray[i] * iHighCacheIndexArray[i]) + globalIndex];
    }
    if (!iLowComputeArray[i]) {
      lowKernel = devCacheArray[i][(devCachePitchInFloatsArray[i] * iLowCacheIndexArray[i]) + globalIndex];
    }
    if (iHighComputeArray[i] && iLowComputeArray[i]) {
      Kernel::dualKernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i], devDataArray[i] + globalIndex + (devDataPitchInFloatsArray[i] * nDimensionArray[i]), xIHigh, 1, xILow, 1, parameterAArray[i], parameterBArray[i], parameterCArray[i], highKernel, lowKernel);
    } else if (iHighComputeArray[i]) {
      highKernel = Kernel::kernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i], devDataArray[i] + globalIndex + (devDataPitchInFloatsArray[i] * nDimensionArray[i]), xIHigh, 1, parameterAArray[i], parameterBArray[i], parameterCArray[i]);
    } else if (iLowComputeArray[i]) {
      lowKernel = Kernel::kernel(devDataArray[i] + globalIndex, devDataPitchInFloatsArray[i], devDataArray[i] + globalIndex + (devDataPitchInFloatsArray[i] * nDimensionArray[i]), xILow, 1, parameterAArray[i], parameterBArray[i], parameterCArray[i]);
    }

     
    f = f + alpha1DiffArray[i] * highKernel;
    f = f + alpha2DiffArray[i] * lowKernel;
    if (iLowComputeArray[i]) {
      devCacheArray[i][(devCachePitchInFloatsArray[i] * iLowCacheIndexArray[i]) + globalIndex] = lowKernel;
    }
    if (iHighComputeArray[i]) {
      devCacheArray[i][(devCachePitchInFloatsArray[i] * iHighCacheIndexArray[i]) + globalIndex] = highKernel;
    }

    devFArray[i][globalIndex] = f;
    
  
  }
  __syncthreads();


  if ((reduceFlag & REDUCE0) == 0) {
    tempLocalFs[threadIdx.x] = -FLT_MAX; //Ignore me
  } else {
    tempLocalFs[threadIdx.x] = f;
    tempLocalIndices[threadIdx.x] = globalIndex;
  }

  __syncthreads();
  
  argmaxReduce(tempLocalFs, tempLocalIndices);
  
  
  if (threadIdx.x == 0) {
    devLocalIndicesRLArray[i][blockIdx.x] = tempLocalIndices[0];
    devLocalFsRLArray[i][blockIdx.x] = tempLocalFs[0];
  }
 
  __syncthreads();

  if ((reduceFlag & REDUCE1) == 0) {
    tempLocalFs[threadIdx.x] = FLT_MAX; //Ignore me
  } else {
    tempLocalFs[threadIdx.x] = f;
    tempLocalIndices[threadIdx.x] = globalIndex;
  }
  __syncthreads();
   
  argminReduce(tempLocalFs, tempLocalIndices);
  if (threadIdx.x == 0) {
    devLocalIndicesRHArray[i][blockIdx.x] = tempLocalIndices[0];
    devLocalFsRHArray[i][blockIdx.x] = tempLocalFs[0];
  }
}


// added by Zhu Lei
template<class Kernel>
__global__ void firstOrderMultiPhaseTwo(float** devDataArray, int* devDataPitchInFloatsArray, float** devTransposedDataArray, int* devTransposedDataPitchInFloatsArray, float** devLabelsArray, float** devKernelDiagArray, float** devAlphaArray, void** devResultArray, float* costArray, int* nDimensionArray, float* parameterAArray, float* parameterBArray, float* parameterCArray, int** devLocalIndicesRLArray, int** devLocalIndicesRHArray, float** devLocalFsRLArray, float** devLocalFsRHArray, int* inputSizeArray) {
  int idx = blockIdx.x;

  __shared__ int tempIndices[BLOCKSIZE];
  __shared__ float tempFs[BLOCKSIZE];
  
  
  //Load elements
  if (threadIdx.x < inputSizeArray[idx]) {
    tempIndices[threadIdx.x] = devLocalIndicesRHArray[idx][threadIdx.x];
    tempFs[threadIdx.x] = devLocalFsRHArray[idx][threadIdx.x];
  } else {
    tempFs[threadIdx.x] = FLT_MAX;
  }

  if (inputSizeArray[idx] > BLOCKSIZE) {
    for (int i = threadIdx.x + BLOCKSIZE; i < inputSizeArray[idx]; i += blockDim.x) {
      argMin(tempIndices[threadIdx.x], tempFs[threadIdx.x], devLocalIndicesRHArray[idx][i], devLocalFsRHArray[idx][i], tempIndices + threadIdx.x, tempFs + threadIdx.x);
    }
  }
  __syncthreads();
  argminReduce(tempFs, tempIndices);
  int iHigh = tempIndices[0];
  float bHigh = tempFs[0];

  //Load elements
  if (threadIdx.x < inputSizeArray[idx]) {
    tempIndices[threadIdx.x] = devLocalIndicesRLArray[idx][threadIdx.x];
    tempFs[threadIdx.x] = devLocalFsRLArray[idx][threadIdx.x];
  } else {
    tempFs[threadIdx.x] = -FLT_MAX;
  }

  if (inputSizeArray[idx] > BLOCKSIZE) {
    for (int i = threadIdx.x + BLOCKSIZE; i < inputSizeArray[idx]; i += blockDim.x) {
      argMax(tempIndices[threadIdx.x], tempFs[threadIdx.x], devLocalIndicesRLArray[idx][i], devLocalFsRLArray[idx][i], tempIndices + threadIdx.x, tempFs + threadIdx.x);
    }
  }
  __syncthreads();
  
  argmaxReduce(tempFs, tempIndices);
  
  int iLow = tempIndices[0];
  float bLow = tempFs[0];

  
  
  float* highPointer = devTransposedDataArray[idx] + (iHigh * devTransposedDataPitchInFloatsArray[idx]);
  float* lowPointer = devTransposedDataArray[idx] + (iLow * devTransposedDataPitchInFloatsArray[idx]);  
  
  Kernel::parallelKernel(highPointer, highPointer + nDimensionArray[idx], lowPointer, tempFs, parameterAArray[idx], parameterBArray[idx], parameterCArray[idx]);

  if (threadIdx.x == 0) {
    
    float eta = devKernelDiagArray[idx][iHigh] + devKernelDiagArray[idx][iLow];
      
    float kernelEval = tempFs[0];
    eta = eta - 2*kernelEval;
      
    float alpha1Old = devAlphaArray[idx][iHigh];
    float alpha2Old = devAlphaArray[idx][iLow];
    float alphaDiff = alpha2Old - alpha1Old;
    float lowLabel = devLabelsArray[idx][iLow];
    float sign = devLabelsArray[idx][iHigh] * lowLabel;
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
      alpha2New = alpha2Old + lowLabel*(bHigh - bLow)/eta;
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
    float alpha1Diff = -sign*alpha2Diff;
    float alpha1New = alpha1Old + alpha1Diff;

    *((float*)devResultArray[idx] + 0) = alpha2Old;
    *((float*)devResultArray[idx] + 1) = alpha1Old;
    *((float*)devResultArray[idx] + 2) = bLow;
    *((float*)devResultArray[idx] + 3) = bHigh;
    devAlphaArray[idx][iLow] = alpha2New;
    devAlphaArray[idx][iHigh] = alpha1New;
    *((float*)devResultArray[idx] + 4) = alpha2New;
    *((float*)devResultArray[idx] + 5) = alpha1New;
    *((int*)devResultArray[idx] + 6) = iLow;
    *((int*)devResultArray[idx] + 7) = iHigh;
  }
}




template<bool iLowCompute, bool iHighCompute, class Kernel>
  __global__ void	firstOrderPhaseOne(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, int nPoints, int nDimension, float epsilon, float cEpsilon, float* devAlpha, float* devF, float alpha1Diff, float alpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, int* devLocalIndicesRL, int* devLocalIndicesRH, float* devLocalFsRL, float* devLocalFsRH) {


  extern __shared__ float xIHigh[];
	float* xILow;
	__shared__ int tempLocalIndices[BLOCKSIZE];
	__shared__ float tempLocalFs[BLOCKSIZE];
  
  
	if (iHighCompute) {
		xILow = &xIHigh[nDimension];
	} else {
		xILow = xIHigh;
	}

  
  if (iHighCompute) {
    //Load xIHigh into shared memory
    coopExtractRowVector(devTransposedData, devTransposedDataPitchInFloats, iHigh, nDimension, xIHigh);
	}

  if (iLowCompute) {
    //Load xILow into shared memory
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
    
    
		if (alpha > epsilon) {
			if (alpha < cEpsilon) {
				reduceFlag = REDUCE0 | REDUCE1; //Unbound support vector (I0)
			} else {
				if (label > 0) {
					reduceFlag = REDUCE0; //Bound positive support vector (I3)
				} else {
					reduceFlag = REDUCE1; //Bound negative support vector (I2)
				}
			}
		} else {
			if (label > 0) {
				reduceFlag = REDUCE1; //Positive nonsupport vector (I1)
			} else {
				reduceFlag = REDUCE0; //Negative nonsupport vector (I4)
			}
		}
	} else {
		reduceFlag = NOREDUCE;
	}

  
  float highKernel = 0;
  float lowKernel = 0;
	if (reduceFlag > 0) {
		if (!iHighCompute) {
			highKernel = devCache[(devCachePitchInFloats * iHighCacheIndex) + globalIndex];
		}
		if (!iLowCompute) {
			lowKernel = devCache[(devCachePitchInFloats * iLowCacheIndex) + globalIndex];
		}
    if (iHighCompute && iLowCompute) {
      Kernel::dualKernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1, xILow, 1, parameterA, parameterB, parameterC, highKernel, lowKernel);
    } else if (iHighCompute) {
      highKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xIHigh, 1, parameterA, parameterB, parameterC);
    } else if (iLowCompute) {
      lowKernel = Kernel::kernel(devData + globalIndex, devDataPitchInFloats, devData + globalIndex + (devDataPitchInFloats * nDimension), xILow, 1, parameterA, parameterB, parameterC);
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
    tempLocalFs[threadIdx.x] = -FLT_MAX; //Ignore me
  } else {
    tempLocalFs[threadIdx.x] = f;
    tempLocalIndices[threadIdx.x] = globalIndex;
  }

  __syncthreads();
  
  argmaxReduce(tempLocalFs, tempLocalIndices);
  
  
  if (threadIdx.x == 0) {
		devLocalIndicesRL[blockIdx.x] = tempLocalIndices[0];
		devLocalFsRL[blockIdx.x] = tempLocalFs[0];
  }
 
  __syncthreads();

  if ((reduceFlag & REDUCE1) == 0) {
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

template<class Kernel>
__global__ void firstOrderPhaseTwo(float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float* devKernelDiag, float* devAlpha, void* devResult, float cost, int nDimension, float parameterA, float parameterB, float parameterC, int* devLocalIndicesRL, int* devLocalIndicesRH, float* devLocalFsRL, float* devLocalFsRH, int inputSize) {
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
      argMin(tempIndices[threadIdx.x], tempFs[threadIdx.x], devLocalIndicesRH[i], devLocalFsRH[i], tempIndices + threadIdx.x, tempFs + threadIdx.x);
    }
  }
  __syncthreads();
  argminReduce(tempFs, tempIndices);
  int iHigh = tempIndices[0];
  float bHigh = tempFs[0];

  //Load elements
  if (threadIdx.x < inputSize) {
    tempIndices[threadIdx.x] = devLocalIndicesRL[threadIdx.x];
    tempFs[threadIdx.x] = devLocalFsRL[threadIdx.x];
  } else {
    tempFs[threadIdx.x] = -FLT_MAX;
  }

  if (inputSize > BLOCKSIZE) {
    for (int i = threadIdx.x + BLOCKSIZE; i < inputSize; i += blockDim.x) {
      argMax(tempIndices[threadIdx.x], tempFs[threadIdx.x], devLocalIndicesRL[i], devLocalFsRL[i], tempIndices + threadIdx.x, tempFs + threadIdx.x);
    }
  }
  __syncthreads();
  
  argmaxReduce(tempFs, tempIndices);
  
  int iLow = tempIndices[0];
  float bLow = tempFs[0];

  
  
  float* highPointer = devTransposedData + (iHigh * devTransposedDataPitchInFloats);
  float* lowPointer = devTransposedData + (iLow * devTransposedDataPitchInFloats);  
  
  Kernel::parallelKernel(highPointer, highPointer + nDimension, lowPointer, tempFs, parameterA, parameterB, parameterC);

  if (threadIdx.x == 0) {
    
    float eta = devKernelDiag[iHigh] + devKernelDiag[iLow];
      
    float kernelEval = tempFs[0];
    eta = eta - 2*kernelEval;
      
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
      alpha2New = alpha2Old + lowLabel*(bHigh - bLow)/eta;
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
    float alpha1Diff = -sign*alpha2Diff;
    float alpha1New = alpha1Old + alpha1Diff;

    *((float*)devResult + 0) = alpha2Old;
    *((float*)devResult + 1) = alpha1Old;
    *((float*)devResult + 2) = bLow;
    *((float*)devResult + 3) = bHigh;
    devAlpha[iLow] = alpha2New;
    devAlpha[iHigh] = alpha1New;
    *((float*)devResult + 4) = alpha2New;
    *((float*)devResult + 5) = alpha1New;
    *((int*)devResult + 6) = iLow;
    *((int*)devResult + 7) = iHigh;
  }
}

// added by Zhu Lei
void launchMultiFirstOrder(int num_svm, bool* iLowComputeArray, bool* iHighComputeArray, int kType, int* nPointsArray, int* nDimensionArray, int* devnDimensionArray, dim3 blocksConfig, dim3 threadsConfig, dim3 globalThreadsConfig, float** devDataArray, int* devDataPitchInFloatsArray, float** devTransposedDataArray, int* devTransposedDataPitchInFloatsArray, float** devLabelsArray, float* epsilonArray, float* cEpsilonArray, float** devAlphaArray, float** devFArray, float* sAlpha1DiffArray, float* sAlpha2DiffArray, int* iLowArray, int* iHighArray, float* parameterAArray, float* parameterBArray, float* parameterCArray, float** devCacheArray, int* devCachePitchInFloatsArray, int* iLowCacheIndexArray, int* iHighCacheIndexArray, int** devLocalIndicesRLArray, int** devLocalIndicesRHArray, float** devLocalFsRHArray, float** devLocalFsRLArray, float** devKernelDiagArray, void** devResultArray, float* costArray, int* blockWidthArray, bool* deviLowComputeArray, bool* deviHighComputeArray) {

  int i;
  int phaseOneSize = 0;
  for(i = 0; i < num_svm; i++) {
    int temp = firstOrderPhaseOneSize(iLowComputeArray[i], iHighComputeArray[i], nDimensionArray[i]);
    if(phaseOneSize < temp) phaseOneSize = temp;
  }
  int phaseTwoSize = 0;
  for(i = 0; i < num_svm; i++) {
    int temp = firstOrderPhaseTwoSize();
    if(phaseTwoSize < temp) phaseTwoSize = temp;
  }

  if (iLowComputeArray[0] == true) {
    if (iHighComputeArray[0] == true) {
      switch (kType) {
      case LINEAR:
        firstOrderMultiPhaseOne <true, true, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case POLYNOMIAL:
        firstOrderMultiPhaseOne <true, true, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case GAUSSIAN:
        firstOrderMultiPhaseOne <true, true, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case SIGMOID:
        firstOrderMultiPhaseOne <true, true, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      }
    } else if (iHighComputeArray[0] == false) {
      switch (kType) {
      case LINEAR:
        firstOrderMultiPhaseOne <true, false, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case POLYNOMIAL:
        firstOrderMultiPhaseOne <true, false, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case GAUSSIAN:
        firstOrderMultiPhaseOne <true, false, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case SIGMOID:
        firstOrderMultiPhaseOne <true, false, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      }
          
    }
  } else if (iLowComputeArray[0] == false) {
    if (iHighComputeArray[0] == true) {
      switch (kType) {
      case LINEAR:
        firstOrderMultiPhaseOne <false, true, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case POLYNOMIAL:
        firstOrderMultiPhaseOne <false, true, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case GAUSSIAN:
        firstOrderMultiPhaseOne <false, true, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case SIGMOID:
        firstOrderMultiPhaseOne <false, true, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      }
    } else if (iHighComputeArray[0] == false) {
      switch (kType) {
      case LINEAR:
        firstOrderMultiPhaseOne <false, false, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case POLYNOMIAL:
        firstOrderMultiPhaseOne <false, false, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case GAUSSIAN:
        firstOrderMultiPhaseOne <false, false, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      case SIGMOID:
        firstOrderMultiPhaseOne <false, false, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(deviLowComputeArray, deviHighComputeArray, devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, nPointsArray, devnDimensionArray, epsilonArray, cEpsilonArray, devAlphaArray, devFArray, sAlpha1DiffArray, sAlpha2DiffArray, iLowArray, iHighArray, parameterAArray, parameterBArray, parameterCArray, devCacheArray, devCachePitchInFloatsArray, iLowCacheIndexArray, iHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray);
        break;
      }
          
    }
  }

  switch (kType) {
  case LINEAR:
    firstOrderMultiPhaseTwo<Linear><<<num_svm, globalThreadsConfig, phaseTwoSize>>>(devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, devAlphaArray, devResultArray, costArray, devnDimensionArray, parameterAArray, parameterBArray, parameterCArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray, blockWidthArray);
    break;
  case POLYNOMIAL:
    firstOrderMultiPhaseTwo<Polynomial><<<num_svm, globalThreadsConfig, phaseTwoSize>>>(devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, devAlphaArray, devResultArray, costArray, devnDimensionArray, parameterAArray, parameterBArray, parameterCArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray, blockWidthArray);
    break;
  case GAUSSIAN:
    firstOrderMultiPhaseTwo<Gaussian><<<num_svm, globalThreadsConfig, phaseTwoSize>>>(devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, devAlphaArray, devResultArray, costArray, devnDimensionArray, parameterAArray, parameterBArray, parameterCArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray, blockWidthArray);
    break;
  case SIGMOID:
    firstOrderMultiPhaseTwo<Sigmoid><<<num_svm, globalThreadsConfig, phaseTwoSize>>>(devDataArray, devDataPitchInFloatsArray, devTransposedDataArray, devTransposedDataPitchInFloatsArray, devLabelsArray, devKernelDiagArray, devAlphaArray, devResultArray, costArray, devnDimensionArray, parameterAArray, parameterBArray, parameterCArray, devLocalIndicesRLArray, devLocalIndicesRHArray, devLocalFsRLArray, devLocalFsRHArray, blockWidthArray);
    break;
  }
} 

void launchFirstOrder(bool iLowCompute, bool iHighCompute, int kType, int nPoints, int nDimension, dim3 blocksConfig, dim3 threadsConfig, dim3 globalThreadsConfig, float* devData, int devDataPitchInFloats, float* devTransposedData, int devTransposedDataPitchInFloats, float* devLabels, float epsilon, float cEpsilon, float* devAlpha, float* devF, float sAlpha1Diff, float sAlpha2Diff, int iLow, int iHigh, float parameterA, float parameterB, float parameterC, float* devCache, int devCachePitchInFloats, int iLowCacheIndex, int iHighCacheIndex, int* devLocalIndicesRL, int* devLocalIndicesRH, float* devLocalFsRH, float* devLocalFsRL, float* devKernelDiag, void* devResult, float cost) {

  int phaseOneSize = firstOrderPhaseOneSize(iLowCompute, iHighCompute, nDimension);
  int phaseTwoSize = firstOrderPhaseTwoSize();
  if (iLowCompute == true) {
    if (iHighCompute == true) {
      switch (kType) {
      case LINEAR:
        firstOrderPhaseOne <true, true, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case POLYNOMIAL:
        firstOrderPhaseOne <true, true, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case GAUSSIAN:
        firstOrderPhaseOne <true, true, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case SIGMOID:
        firstOrderPhaseOne <true, true, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      }
    } else if (iHighCompute == false) {
      switch (kType) {
      case LINEAR:
        firstOrderPhaseOne <true, false, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case POLYNOMIAL:
        firstOrderPhaseOne <true, false, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case GAUSSIAN:
        firstOrderPhaseOne <true, false, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case SIGMOID:
        firstOrderPhaseOne <true, false, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      }
          
    }
  } else if (iLowCompute == false) {
    if (iHighCompute == true) {
      switch (kType) {
      case LINEAR:
        firstOrderPhaseOne <false, true, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case POLYNOMIAL:
        firstOrderPhaseOne <false, true, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case GAUSSIAN:
        firstOrderPhaseOne <false, true, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case SIGMOID:
        firstOrderPhaseOne <false, true, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      }
    } else if (iHighCompute == false) {
      switch (kType) {
      case LINEAR:
        firstOrderPhaseOne <false, false, Linear><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case POLYNOMIAL:
        firstOrderPhaseOne <false, false, Polynomial><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case GAUSSIAN:
        firstOrderPhaseOne <false, false, Gaussian><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      case SIGMOID:
        firstOrderPhaseOne <false, false, Sigmoid><<<blocksConfig, threadsConfig, phaseOneSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, nPoints, nDimension, epsilon, cEpsilon, devAlpha, devF, sAlpha1Diff, sAlpha2Diff, iLow, iHigh, parameterA, parameterB, parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH);
        break;
      }
          
    }
  }

  
  switch (kType) {
  case LINEAR:
    firstOrderPhaseTwo<Linear><<<1, globalThreadsConfig, phaseTwoSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, devAlpha, devResult, cost, nDimension, parameterA, parameterB, parameterC, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH, blocksConfig.x);
    break;
  case POLYNOMIAL:
    firstOrderPhaseTwo<Polynomial><<<1, globalThreadsConfig, phaseTwoSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, devAlpha, devResult, cost, nDimension, parameterA, parameterB, parameterC, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH, blocksConfig.x);
    break;
  case GAUSSIAN:
    firstOrderPhaseTwo<Gaussian><<<1, globalThreadsConfig, phaseTwoSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, devAlpha, devResult, cost, nDimension, parameterA, parameterB, parameterC, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH, blocksConfig.x);
    break;
  case SIGMOID:
    firstOrderPhaseTwo<Sigmoid><<<1, globalThreadsConfig, phaseTwoSize>>>(devData, devDataPitchInFloats, devTransposedData, devTransposedDataPitchInFloats, devLabels, devKernelDiag, devAlpha, devResult, cost, nDimension, parameterA, parameterB, parameterC, devLocalIndicesRL, devLocalIndicesRH, devLocalFsRL, devLocalFsRH, blocksConfig.x);
    break;
  }


  
   
} 
#endif
