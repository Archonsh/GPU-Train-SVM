/*
 * processData.cu
 *
 *  Add on: Jul 14, 2016
 *  Add by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  To make spliting the dataset for cross validation faster.
 */

#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#define BDIMXL 1024
#define MAXBLOCK 65535

using namespace std;

struct threadData{
	int gpuId;
	int svmId;
	int numSVMGpu;
	int folder;

	float** dataArray;
	int* nPointsArray;
	int* nDimensionArray;
	int** permutation;

	// output data
	float** dataArrayOut;
	float** transposedDataArrayOut;
	float** testDataArrayOut;
	float** testTransposedDataArrayOut;
};

__global__ void process(int folder, float* devData, int* devPermutation, int* devnPointsArray, int* devnDimensionArray, int* devPartition, float* devDataOut, float* devTransposedDataOut) {
	int i;

	int blockId = blockIdx.x; 
	int svmId = blockId / folder;
	int partId = blockId % folder;
	int pointId = threadIdx.x;

	int inPointer = 0;
	int inPermId = 0;
	int outStartPointer = 0;
	int outPartitionPointer = 0;
	for(i = 0; i < svmId; i++) {
		inPermId += devnPointsArray[i];
		inPointer += devnPointsArray[i] * devnDimensionArray[i];
		outStartPointer += folder* devnPointsArray[i] * devnDimensionArray[i];
	}

	for(i = 0; i < partId; i++) {
		outStartPointer += devnPointsArray[svmId] * devnDimensionArray[svmId];
	}

	outPartitionPointer = outStartPointer + devPartition[blockId] * devnDimensionArray[svmId];

	while(pointId < devnPointsArray[svmId]) {
		int outPointer = 0;
		int outPointerTran = 0;
		int stride = 0;
		// test data portion
		if(pointId % folder == partId) {
			outPointer = outPartitionPointer + devnDimensionArray[svmId] * (pointId / folder);
			outPointerTran = outPartitionPointer + pointId / folder;
			stride = devnPointsArray[svmId] - devPartition[blockId];
		} else {
			int idx = pointId % folder;
			if(idx > partId)
				idx = idx - 1;
			outPointer = outStartPointer + devnDimensionArray[svmId] * ((folder-1) * (pointId/folder) + idx);
			outPointerTran = outStartPointer + (folder-1) * (pointId/folder) + idx;
			stride = devPartition[blockId];
		}

		int tmpInPermId = inPermId + pointId;
		int tmpInPointer = inPointer + devnDimensionArray[svmId] * devPermutation[tmpInPermId];

		for(i = 0; i < devnDimensionArray[svmId]; i++) {
			devDataOut[outPointer+i] = devData[tmpInPointer+i];
			devTransposedDataOut[outPointerTran+i*stride] = devData[tmpInPointer+i];
		}
		pointId += BDIMXL;
	}
}

sem_t processMutex;
bool *isGpuAvailable;
int *numSVMPerGPU;
void *processThread(void * threadarg) {
	struct threadData *myData;
	myData = (struct threadData *) threadarg;

	int i, j;
	int gpuId = myData->gpuId;
	checkCudaErrors(cudaSetDevice(gpuId));

	int svmId = myData->svmId;
	int numSVMGpu = myData->numSVMGpu;
	int folder = myData->folder;

	float** dataArray = myData->dataArray;
	int* nPointsArray = myData->nPointsArray;
	int* nDimensionArray = myData->nDimensionArray;
	int** permutation = myData->permutation;

	float** dataArrayOut = myData->dataArrayOut;
	float** transposedDataArrayOut = myData->transposedDataArrayOut;
	float** testTransposedDataArrayOut = myData->testTransposedDataArrayOut;
	float** testDataArrayOut = myData->testDataArrayOut;

	int origSize = 0;
	for(i = 0; i < numSVMGpu; i++) {
		origSize += nPointsArray[svmId+i] * nDimensionArray[svmId+i];
	}

	float* devData;
	checkCudaErrors(cudaMalloc((float**)&devData, sizeof(float)*origSize));

	int count = 0;
	for(i = 0; i < numSVMGpu; i++) {
		// copy data from host to device
		checkCudaErrors(cudaMemcpy(devData+count, dataArray[svmId+i], sizeof(float)*nPointsArray[svmId+i]*nDimensionArray[svmId+i], cudaMemcpyHostToDevice));
		count += nPointsArray[svmId+i] * nDimensionArray[svmId+i];
	}
	
	int permSize = 0;
	for(i = 0; i < numSVMGpu; i++) {
		permSize += nPointsArray[svmId+i];
	}

	int* devPermutation;
	checkCudaErrors(cudaMalloc((int**)&devPermutation, sizeof(int)*permSize));

	count = 0;
	for(i = 0; i < numSVMGpu; i++) {
		checkCudaErrors(cudaMemcpy(devPermutation+count, permutation[svmId+i], sizeof(int)*nPointsArray[svmId+i], cudaMemcpyHostToDevice));
		count += nPointsArray[svmId+i];
	}

	int* devnPointsArray;
	checkCudaErrors(cudaMalloc((int**)&devnPointsArray, sizeof(int)*numSVMGpu));
	checkCudaErrors(cudaMemcpy(devnPointsArray, nPointsArray+svmId, sizeof(int)*numSVMGpu, cudaMemcpyHostToDevice));

	int* devnDimensionArray;
	checkCudaErrors(cudaMalloc((int**)&devnDimensionArray, sizeof(int)*numSVMGpu));
	checkCudaErrors(cudaMemcpy(devnDimensionArray, nDimensionArray+svmId, sizeof(int)*numSVMGpu, cudaMemcpyHostToDevice));

	int totalNumSVM = folder * numSVMGpu;
	int* partition = (int *) malloc(sizeof(int) * totalNumSVM);

	for(i = 0; i < numSVMGpu; i++) {
		int tmpValueA = nPointsArray[i] / folder;
		int tmpValueB = nPointsArray[i] % folder;
		for(j = 0; j < folder; j++) {
			if(j + 1 > tmpValueB)
				partition[i*folder+j] = nPointsArray[i] - tmpValueA;
			else
				partition[i*folder+j] = nPointsArray[i] - (tmpValueA + 1); 
		}
	}

	int* devPartition;
	checkCudaErrors(cudaMalloc((int**)&devPartition, sizeof(int)*totalNumSVM));
	checkCudaErrors(cudaMemcpy(devPartition, partition, sizeof(int)*totalNumSVM, cudaMemcpyHostToDevice));

	int size = 0;
	for(i = 0; i < numSVMGpu; i++) {
		size += folder * nPointsArray[svmId+i] * nDimensionArray[svmId+i];
	}

	float* devDataOut;
	float* devTransposedDataOut;
	checkCudaErrors(cudaMalloc((float**)&devDataOut, sizeof(float)*size));
	checkCudaErrors(cudaMalloc((float**)&devTransposedDataOut, sizeof(float)*size));

	// assume totalNumSVM < 2147483647
	float avg = 0;
	for(i = 0; i < numSVMGpu; i++) {
		avg += nPointsArray[i];
	}
	avg = avg / numSVMGpu;
	int BDIMX = (int) avg;

	if(BDIMX > BDIMXL) BDIMX = BDIMXL;

	cout << "GPU " << gpuId << " start processing." << endl;
	process<<<totalNumSVM, BDIMX>>>(folder, devData, devPermutation, devnPointsArray, devnDimensionArray, devPartition, devDataOut, devTransposedDataOut);
	cout << "GPU " << gpuId << " finish processing." << endl;

	int cursor = 0;
	for(i = 0; i < numSVMGpu; i++) {
		for(j = 0; j < folder; j++) {
			if(dataArrayOut != NULL)
				checkCudaErrors(cudaMemcpy(dataArrayOut[svmId*folder+i*folder+j], devDataOut+cursor, sizeof(float)*nDimensionArray[i]*partition[i*folder+j], cudaMemcpyDeviceToHost));
			if(transposedDataArrayOut != NULL)
				checkCudaErrors(cudaMemcpy(transposedDataArrayOut[svmId*folder+i*folder+j], devTransposedDataOut+cursor, sizeof(float)*nDimensionArray[i]*partition[i*folder+j], cudaMemcpyDeviceToHost));
			if(testTransposedDataArrayOut != NULL)
				checkCudaErrors(cudaMemcpy(testTransposedDataArrayOut[svmId*folder+i*folder+j], devTransposedDataOut+cursor+nDimensionArray[i]*partition[i*folder+j], sizeof(float)*nDimensionArray[i]*(nPointsArray[i]-partition[i*folder+j]), cudaMemcpyDeviceToHost));
			if(testDataArrayOut != NULL)
				checkCudaErrors(cudaMemcpy(testDataArrayOut[svmId*folder+i*folder+j], devDataOut+cursor+nDimensionArray[i]*partition[i*folder+j], sizeof(float)*nDimensionArray[i]*(nPointsArray[i]-partition[i*folder+j]), cudaMemcpyDeviceToHost));
			cursor += nDimensionArray[i] * nPointsArray[i];
		}
	}

	// free pointers
	checkCudaErrors(cudaFree(devData));
	checkCudaErrors(cudaFree(devPartition));
	checkCudaErrors(cudaFree(devPermutation));
	checkCudaErrors(cudaFree(devnDimensionArray));
	checkCudaErrors(cudaFree(devnPointsArray));
	checkCudaErrors(cudaFree(devDataOut));
	checkCudaErrors(cudaFree(devTransposedDataOut));

	cout << "Release GPU " << gpuId << "." << endl;
	sem_wait(&processMutex);
	isGpuAvailable[gpuId] = true;
	sem_post(&processMutex);
	numSVMPerGPU[gpuId] += numSVMGpu;

	return NULL;
}

int calMemForSingleSVM(int folder, int nPoints, int nDimension) {
	int size = 0;
	// memory for original data and output data
	size += sizeof(float) * nPoints * nDimension * (2 * folder + 1);
	// memory for permutation indexes
	size += sizeof(int) * nPoints; 
	// memory for partition
	size += sizeof(int) * folder;
	// memory for nPoints
	size += sizeof(int);
	// memory for nDimension
	size += sizeof(int);

	return size;
}

// each element in dataArray is arranged as row major matrix
int* processData(int ngpus, int numSVM, int folder, float** dataArray, int* nPointsArray, int* nDimensionArray, int** permutation, float** dataArrayOut, float** transposedDataArrayOut, float** testDataArrayOut ,float** testTransposedDataArrayOut) {
	int i;

	size_t * freeMem = (size_t *) malloc(sizeof(size_t) * ngpus);
	size_t * totalMem = (size_t *) malloc(sizeof(size_t) * ngpus);

	isGpuAvailable = (bool *) malloc(sizeof(bool) * ngpus);
	numSVMPerGPU = (int *) malloc(sizeof(int) * ngpus);
	for(i = 0; i < ngpus; i++) {
		isGpuAvailable[i] = true;
		numSVMPerGPU[i] = 0;
	}

	cout << "Process Data Start." << endl;

	// initialize processMutex
    sem_init(&processMutex, 0, 1);

	// initialize and set thread joinable
	int rc;
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	vector<pthread_t *> threads;
	int nextSVM = 0;
	pthread_t *newThread;
	struct threadData *td;

	while(nextSVM < numSVM) {
		int availableGPUId = -1;

		sem_wait(&processMutex);
		for(i = 0; i < ngpus; i++) {
			if(isGpuAvailable[i]) {
				availableGPUId = i;
				isGpuAvailable[i] = false;
				break;
			}
		}
		sem_post(&processMutex);

		if(availableGPUId != -1) {
			int numSVMGpu = 0;
			size_t size = 0;
			int index = nextSVM;

			cudaSetDevice(availableGPUId);
			cudaMemGetInfo(&freeMem[availableGPUId], &totalMem[availableGPUId]);
			printf("Total mem: %Zu, Free mem: %Zu, Used mem: %Zu\n", totalMem[availableGPUId], freeMem[availableGPUId], totalMem[availableGPUId] - freeMem[availableGPUId]);

			while(size < freeMem[availableGPUId] * 0.95 && index < numSVM) {
				numSVMGpu ++;
				size += calMemForSingleSVM(folder, nPointsArray[index], nDimensionArray[index]);
				index ++;
			}

			if(numSVMGpu * folder > MAXBLOCK)
				numSVMGpu = MAXBLOCK / folder;

			cout << "GPU " << availableGPUId << " is processing " << numSVMGpu << " SVMs." << endl;

			td = (struct threadData *) malloc(sizeof(struct threadData));

			td->gpuId = availableGPUId;
			td->svmId = nextSVM;
			td->numSVMGpu = numSVMGpu;
			td->folder = folder;
			td->dataArray = dataArray;
			td->nPointsArray = nPointsArray;
			td->nDimensionArray = nDimensionArray;
			td->permutation = permutation;

			td->dataArrayOut = dataArrayOut;
			td->transposedDataArrayOut = transposedDataArrayOut;
			td->testDataArrayOut = testDataArrayOut;
			td->testTransposedDataArrayOut = testTransposedDataArrayOut;

			newThread = (pthread_t *) malloc(sizeof(pthread_t));

			// create new thread for training multple SVMs
			rc = pthread_create(newThread, NULL, processThread, (void*)td);
			threads.push_back(newThread);
			nextSVM += numSVMGpu;
		}
	}

	// free attribute and wait for the other threads
	pthread_attr_destroy(&attr);
	for(i = 0; i < threads.size(); i++) {
		rc = pthread_join(*threads[i], &status);
		if (rc) {
			cout << "Error: unable to join, " << rc << endl;
			exit(-1);
		}
	}

	for(i = 0; i < ngpus; i++) {
		cout << "GPU " << i << " has processed " << numSVMPerGPU[i] << " SVMs." << endl;
	}
	cout << "Process Data End." << endl;
	return numSVMPerGPU;
}