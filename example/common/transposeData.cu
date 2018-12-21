#include <stdio.h>

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

__global__ void transpose(int nRows, int nCols, float* devData, float* devTranData) {
	int pointId = threadIdx.x;

	while(pointId < nRows) {
		int inPos = pointId*nCols;
		int i;
		for(i = 0; i < nCols; i++)
			devTranData[pointId+i*nRows] = devData[inPos+i];

		pointId += BSIZE;
	}

}

void transposeData(int nRows, int nCols, float* data, float* transposedData) {
	float* devData;
	CHECK(cudaMalloc((float**)&devData, sizeof(float)*nRows*nCols));
	CHECK(cudaMemcpy(devData, data, sizeof(float)*nRows*nCols, cudaMemcpyHostToDevice));

	float* devTranData;
	CHECK(cudaMalloc((float**)&devTranData, sizeof(float)*nRows*nCols));

	transpose<<<1, BSIZE>>>(nRows, nCols, devData, devTranData);

	CHECK(cudaMemcpy(transposedData, devTranData, sizeof(float)*nRows*nCols, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(devData));
	CHECK(cudaFree(devTranData));
}