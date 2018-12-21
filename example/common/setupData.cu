#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>

using namespace std;

#define BSIZE 256

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// rearrange the data by classes
__global__ void separateClass(int nPoints, int nDimension, int* devPointIdx, float* devTranData, float* devTranDataSortClass) {
	int pointIdx = blockIdx.x * BSIZE + threadIdx.x;

	if(pointIdx < nPoints) {
		int i;
		int inPos = pointIdx * nDimension;
		int outPos = devPointIdx[pointIdx] * nDimension;
		for(i = 0; i < nDimension; i++)
			devTranDataSortClass[outPos+i] = devTranData[inPos+i];
	}

}

// create dataArray in GPU
__global__ void createSVMData(int numClass, int* devClassDist, int* devClassAccuDist, int* devNPointsArray, int* devNPointsAccuArray, int nDimension, float* devDataArray, float* devTranDataSortClass) {
	int svmId = blockIdx.x;
	int classA = 0;
	while(svmId >= (classA+1)*(numClass-1)-(classA+1)*classA/2)
		classA ++;
	int classB = svmId - classA*(numClass-1) + classA*(classA-1)/2 + classA + 1;
	int pointId = threadIdx.x;

	while(pointId < devNPointsArray[svmId]) {
		int i;
		int inPos = 0;
		if(pointId >= devClassDist[classA])
			inPos = (devClassAccuDist[classB] + pointId - devClassDist[classA]) * nDimension;
		else
			inPos = (devClassAccuDist[classA] + pointId) * nDimension;

		int outPos = devNPointsAccuArray[svmId] * nDimension + pointId;
		for(i = 0; i < nDimension; i++)
			devDataArray[outPos+i*devNPointsArray[svmId]] = devTranDataSortClass[inPos+i];

		pointId += BSIZE;
	}

}


/**
 * Setup both dataArray and transposedDataArray for one-againse-one multi-class training
 * @param numSVM the number of SVMs for training, equals to category.size() * (category.size()-1) / 2
 * @param category stores the labels for different classes
 * @param classDist stores the class distribution for each class, where classDist[0] is the number of points in the original data set which has label category[0]
 * @param nPoints the number of points in the original data
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the row major store of the original data
 * @param labels store the labels for the orignal data
 * @param nPointsArray the number of points for each SVM, where nPointsArray[svmId] = classDist[i] + classDist[j] and there is a one to one mapping between svmId and (i, j), given by svmId = i * (category.size()-1) - i*(i-1)/2 + j - i - 1. Please refer to createSVMData() for the detailed mapping;  
 * @param dataArray is used to store the output column major data for each SVM
 * @param transposedDataArray is used to store the output row major data for each SVM
 */
void setupData(int numSVM, vector<float> category, int* classDist, int nPoints, int nDimension, float* transposedData, float* labels, int* nPointsArray, float** dataArray, float** transposedDataArray) {
	int i;

	// 1. create transposed data sorted by classes
	int* classCount = (int*)calloc(category.size(), sizeof(int));
	int* pointIdx = (int*)malloc(sizeof(int)*nPoints);
	int* classAccuDist = (int*)malloc(sizeof(int)*category.size());
	classAccuDist[0] = 0;
	for(i = 1; i < category.size(); i++)
		classAccuDist[i] = classAccuDist[i-1] + classDist[i-1];

	for(i = 0; i < nPoints; i++) {
		int classIdx = find(category.begin(), category.end(), labels[i]) - category.begin();
		pointIdx[i] = classAccuDist[classIdx] + classCount[classIdx];
		classCount[classIdx] ++;
	}

	int* devPointIdx;
	CHECK(cudaMalloc((int**)&devPointIdx, sizeof(int)*nPoints));
	CHECK(cudaMemcpy(devPointIdx, pointIdx, sizeof(int)*nPoints, cudaMemcpyHostToDevice));

	float* devTranData;
	CHECK(cudaMalloc((float**)&devTranData, sizeof(float)*nPoints*nDimension));
	CHECK(cudaMemcpy(devTranData, transposedData, sizeof(float)*nPoints*nDimension, cudaMemcpyHostToDevice));

	float* devTranDataSortClass;
	CHECK(cudaMalloc((float**)&devTranDataSortClass, sizeof(float)*nPoints*nDimension));

	int gridSize = (nPoints / BSIZE) + 1;
	// sort the transposed data by classes
	separateClass<<<gridSize,  BSIZE>>> (nPoints, nDimension, devPointIdx, devTranData, devTranDataSortClass);

	CHECK(cudaFree(devPointIdx));
	CHECK(cudaFree(devTranData));
	free(classCount);
	free(pointIdx);

	// 2. setup transposedDataArray
	float* transposedDataSortClass = (float*) malloc(sizeof(float)*nPoints*nDimension);
	CHECK(cudaMemcpy(transposedDataSortClass, devTranDataSortClass, sizeof(float)*nPoints*nDimension, cudaMemcpyDeviceToHost));

	int classA = 0;
	int classB = 1;
	// copy data to transposedDataArray
	for(i = 0; i < numSVM; i++) {
		memcpy(transposedDataArray[i], transposedDataSortClass+classAccuDist[classA]*nDimension, sizeof(float)*classDist[classA]*nDimension);
		memcpy(transposedDataArray[i]+classDist[classA]*nDimension, transposedDataSortClass+classAccuDist[classB]*nDimension, sizeof(float)*classDist[classB]*nDimension);

		if(classB == category.size()-1) {
			classA ++;
			classB = classA + 1;
		} else
			classB ++;
	}

	free(transposedDataSortClass);

	// 3. setup dataArray
	int* devNPointsArray;
	CHECK(cudaMalloc((int**)&devNPointsArray, sizeof(int)*numSVM));
	CHECK(cudaMemcpy(devNPointsArray, nPointsArray, sizeof(int)*numSVM, cudaMemcpyHostToDevice));

	int* devClassDist;
	CHECK(cudaMalloc((int**)&devClassDist, sizeof(int)*category.size()));
	CHECK(cudaMemcpy(devClassDist, classDist, sizeof(int)*category.size(), cudaMemcpyHostToDevice));

	int* devClassAccuDist;
	CHECK(cudaMalloc((int**)&devClassAccuDist, sizeof(int)*category.size()));
	CHECK(cudaMemcpy(devClassAccuDist, classAccuDist, sizeof(int)*category.size(), cudaMemcpyHostToDevice));

	int* nPointsAccuArray = (int*)malloc(sizeof(int)*numSVM);
	nPointsAccuArray[0] = 0;
	for(i = 1; i < numSVM; i++) 
		nPointsAccuArray[i] = nPointsAccuArray[i-1] + nPointsArray[i-1];

	int* devNPointsAccuArray;
	CHECK(cudaMalloc((int**)&devNPointsAccuArray, sizeof(int)*numSVM));
	CHECK(cudaMemcpy(devNPointsAccuArray, nPointsAccuArray, sizeof(int)*numSVM, cudaMemcpyHostToDevice));

	int totalPoints = nPointsAccuArray[numSVM-1]+nPointsArray[numSVM-1];
	float* devDataArray;
	CHECK(cudaMalloc((float**)&devDataArray, sizeof(float)*totalPoints*nDimension));

	// create devDataArray
	createSVMData<<<numSVM, BSIZE>>>(category.size(), devClassDist, devClassAccuDist, devNPointsArray, devNPointsAccuArray, nDimension, devDataArray, devTranDataSortClass);

	// copy data to dataArray
	for(i = 0; i < numSVM; i++)
		CHECK(cudaMemcpy(dataArray[i], devDataArray+nPointsAccuArray[i]*nDimension, sizeof(float)*nPointsArray[i]*nDimension, cudaMemcpyDeviceToHost));

	// free pointers
	CHECK(cudaFree(devClassDist));
	CHECK(cudaFree(devClassAccuDist));
	CHECK(cudaFree(devNPointsArray));
	CHECK(cudaFree(devNPointsAccuArray));
	CHECK(cudaFree(devDataArray));
	CHECK(cudaFree(devTranDataSortClass));
	free(classAccuDist);
	free(nPointsAccuArray);
}

// randomly partition the data into folder parts
__global__ void partitionData(int folder, int nPoints, int nDimension, int* devDataPartitionSize, int* devDataPartitionAccuSize, float* devTranData, int* devPermutation, float* devDataPar, float* devTranDataPar) {
	int pointId = threadIdx.x;

	while(pointId < nPoints) {
		int inPos = devPermutation[pointId]*nDimension;
		int partId = pointId % folder;
		int partPointId = pointId / folder;
		int outPosData = devDataPartitionAccuSize[partId]*nDimension + partPointId;
		int outPosTranData = (devDataPartitionAccuSize[partId]+partPointId) * nDimension;
		int i;
		for(i = 0; i < nDimension; i++) {
			devDataPar[outPosData+i*devDataPartitionSize[partId]] = devTranData[inPos+i];
			devTranDataPar[outPosTranData+i] = devTranData[inPos+i];
		}

		pointId += BSIZE;
	}

}

void setupDataCV(int folder, int nPoints, int nDimension, float* transposedData, int* permutation, float** dataPartitionArray, float** dataTranPartitionArray) {
	int i;

	// dataPartitionSize stores sizes for each partition
	int* dataPartitionSize = (int*)malloc(sizeof(int)*folder);
	for(i = 0; i < folder; i++) {
		dataPartitionSize[i] = nPoints / folder;
		if(nPoints % folder != 0 && i < nPoints % folder)
			dataPartitionSize[i] ++;
	}

	int* dataPartitionAccuSize = (int*)malloc(sizeof(int)*folder);
	dataPartitionAccuSize[0] = 0;
	for(i = 1; i < folder; i++) {
		dataPartitionAccuSize[i] = dataPartitionAccuSize[i-1] + dataPartitionSize[i-1];
	}

	int* devDataPartitionSize;
	CHECK(cudaMalloc((int**)&devDataPartitionSize, sizeof(int)*folder));
	CHECK(cudaMemcpy(devDataPartitionSize, dataPartitionSize, sizeof(int)*folder, cudaMemcpyHostToDevice));

	int* devDataPartitionAccuSize;
	CHECK(cudaMalloc((int**)&devDataPartitionAccuSize, sizeof(int)*folder));
	CHECK(cudaMemcpy(devDataPartitionAccuSize, dataPartitionAccuSize, sizeof(int)*folder, cudaMemcpyHostToDevice));

	float* devTranData;
	CHECK(cudaMalloc((float**)&devTranData, sizeof(float)*nPoints*nDimension));
	CHECK(cudaMemcpy(devTranData, transposedData, sizeof(float)*nPoints*nDimension, cudaMemcpyHostToDevice));

	int* devPermutation;
	CHECK(cudaMalloc((int**)&devPermutation, sizeof(int)*nPoints));
	CHECK(cudaMemcpy(devPermutation, permutation, sizeof(int)*nPoints, cudaMemcpyHostToDevice));

	float* devDataPar;
	CHECK(cudaMalloc((float**)&devDataPar, sizeof(float)*nPoints*nDimension));

	float* devTranDataPar;
	CHECK(cudaMalloc((float**)&devTranDataPar, sizeof(float)*nPoints*nDimension));

	partitionData<<<1, BSIZE>>>(folder, nPoints, nDimension, devDataPartitionSize, devDataPartitionAccuSize, devTranData, devPermutation, devDataPar, devTranDataPar);

	for(i = 0; i < folder; i++) {
		CHECK(cudaMemcpy(dataPartitionArray[i], devDataPar+dataPartitionAccuSize[i]*nDimension, sizeof(float)*dataPartitionSize[i]*nDimension, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(dataTranPartitionArray[i], devTranDataPar+dataPartitionAccuSize[i]*nDimension, sizeof(float)*dataPartitionSize[i]*nDimension, cudaMemcpyDeviceToHost));
	}

	// free pointers
	free(dataPartitionSize);
	free(dataPartitionAccuSize);
	CHECK(cudaFree(devDataPartitionSize));
	CHECK(cudaFree(devDataPartitionAccuSize));
	CHECK(cudaFree(devTranData));
	CHECK(cudaFree(devPermutation));
	CHECK(cudaFree(devDataPar));
	CHECK(cudaFree(devTranDataPar));
}