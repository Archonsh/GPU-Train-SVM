/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
/* Includes, cuda */
#include "cublas.h"
#include "helper_cuda.h"
#include "cuda.h"

/* Includes, project */
#include "../common/framework.h"
#include "../common/deviceSelect.h"
#include "svmClassify.h"
#include "svmClassifyKernels.h"


/**
 * Performs SVM classification.
 * @param data the data to be classfied, stored as a flat column major array.
 * @param nData the number of data points being classified
 * @param supportVectors the support vectors of the classifier, stored as a flat column major array.
 * @param nSV the number of support vectors of the classifier
 * @param nDimension the dimensionality of the data and support vectors
 * @param kp a struct containing all the information about the kernel parameters
 * @param p_result a pointer to a float pointer where the results will be placed.  The perform classification routine will allocate the output buffer.
 */
void performClassification(float *data, int nData, float *supportVectors, int nSV, int nDimension, float* alpha, Kernel_params kp, float** p_result, int gpuId)
{	
	if(gpuId == -1)
		chooseLargestGPU(true);

	int total_nPoints = nData;
	int nPoints;	
	float gamma,coef0,b;
	int degree;
	
	if(kp.kernel_type.compare(0,3,"rbf") == 0)
	{
		printf("Found RBF kernel\n");
		gamma=kp.gamma;
		b=kp.b;
	}
	else if(kp.kernel_type.compare(0,10,"polynomial") == 0)
	{
		printf("Found polynomial kernel\n");
		gamma=kp.gamma;
		degree=kp.degree;
		coef0 = kp.coef0;
		b=kp.b;
	}
	else if(kp.kernel_type.compare(0,6,"linear") == 0)
	{
		printf("Found linear kernel\n");
		gamma = 1.0;
		b=kp.b;
	}
	else if(kp.kernel_type.compare(0,7,"sigmoid") == 0)
	{
		printf("Found sigmoid kernel\n");
		gamma = kp.gamma;
		coef0 = kp.coef0;
		b=kp.b;
		//printf("gamma = %f coef0=%f\n",gamma, coef0);
	}
	else
	{
		printf("Error: Unknown kernel type - %s\n",kp.kernel_type.c_str());
		exit(0);
	}
	
	int nBlocksSV = intDivideRoundUp(nSV,BLOCKSIZE);

	cublasStatus status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("CUBLAS initialization error\n");
		exit(0);
	}


	float* devSV;
	size_t devSVPitch;
	checkCudaErrors(cudaMallocPitch((void**)&devSV, &devSVPitch, nSV*sizeof(float), nDimension));
	checkCudaErrors(cudaMemcpy2D(devSV, devSVPitch, supportVectors, nSV*sizeof(float), nSV*sizeof(float), nDimension, cudaMemcpyHostToDevice));
	int devSVPitchInFloats = ((int)devSVPitch) / sizeof(float);


	float* devAlpha;
	checkCudaErrors(cudaMalloc((void**)&devAlpha, nSV*sizeof(float)));
	checkCudaErrors(cudaMemcpy(devAlpha, alpha, nSV*sizeof(float), cudaMemcpyHostToDevice));
	
	


	float* devLocalValue;

	float* devResult;
	
	float* result = (float*)malloc(total_nPoints*sizeof(float));
	*(p_result) = result;

	float* devSVDots;
	checkCudaErrors(cudaMalloc((void**)&devSVDots, sizeof(float)*nSV));


	size_t free_memory,total_memory;
	cuMemGetInfo(&free_memory,&total_memory);
	//printf("\nChecking GPU Memory status...\n");
	//printf("Total Memory=%d bytes   Available Memory=%d bytes\n",total_memory, free_memory);
	size_t free_memory_floats = free_memory/sizeof(float);
	
	free_memory_floats = (size_t)(0.9 * free_memory_floats); 

	nPoints = ((free_memory_floats-devSVPitchInFloats*nDimension-nSV-nSV)/(nDimension+1+devSVPitchInFloats+1+nBlocksSV));
	nPoints = (nPoints>>7)<<7;		//for pitch limitations assigning to be a multiple of 128
	
	nPoints = min(nPoints, total_nPoints); //for few points
	nPoints = min(nPoints, (int)MAX_POINTS); //for too many points	

	//printf("Max points that can reside in GPU memory per call = %d\n\n", nPoints);
	
	dim3 mapGrid(intDivideRoundUp(nSV, BLOCKSIZE), nPoints);
	dim3 mapBlock(BLOCKSIZE);
	

	dim3 reduceGrid(1, nPoints);
	dim3 reduceBlock(mapGrid.x, 1);


	float* devData;
	size_t devDataPitch;
	checkCudaErrors(cudaMallocPitch((void**)&devData, &devDataPitch, nPoints*sizeof(float), nDimension));

	int devDataPitchInFloats = ((int)devDataPitch) / sizeof(float);

	float* devDataDots;
	checkCudaErrors(cudaMalloc((void**)&devDataDots, sizeof(float)*nPoints));

	checkCudaErrors(cudaMalloc((void**)&devLocalValue, sizeof(float)*mapGrid.x*mapGrid.y));
	
	checkCudaErrors(cudaMalloc((void**)&devResult, sizeof(float)*mapGrid.y));
	
	float* devDots;
	size_t devDotsPitch;
	checkCudaErrors(cudaMallocPitch((void**)&devDots, &devDotsPitch, nSV*sizeof(float), nPoints));


		
	
	dim3 threadsLinear(BLOCKSIZE);
	if(kp.kernel_type.compare(0,3,"rbf")==0)
	{
		dim3 blocksSVLinear(intDivideRoundUp(nSV, BLOCKSIZE));
		makeSelfDots<<<blocksSVLinear, threadsLinear>>>(devSV, devSVPitchInFloats, devSVDots, nSV, nDimension);
	}

	int iteration=1;

	for(int dataoffset=0; dataoffset<total_nPoints; dataoffset += nPoints) 
	{
		// code for copying data
		if(dataoffset+nPoints > total_nPoints)
		{
			nPoints = total_nPoints-dataoffset;
			mapGrid=dim3(intDivideRoundUp(nSV, BLOCKSIZE), nPoints);
			mapBlock=dim3(BLOCKSIZE);
	
			reduceGrid=dim3(1, nPoints);
			reduceBlock=dim3(mapGrid.x, 1);

			checkCudaErrors(cudaFree(devLocalValue));
			checkCudaErrors(cudaMalloc((void**)&devLocalValue, sizeof(float)*mapGrid.x*mapGrid.y));
	
			//resize & copy devdata, devdots,
			checkCudaErrors(cudaFree(devData));
			checkCudaErrors(cudaMallocPitch((void**)&devData, &devDataPitch, nPoints*sizeof(float), nDimension));
			devDataPitchInFloats = devDataPitch/sizeof(float);
		}
		
		//printf("Number of Points in call #%d=%d \n",iteration, nPoints);
		
		if(total_nPoints*sizeof(float) < MAX_PITCH)
		{	
			checkCudaErrors(cudaMemcpy2D(devData, devDataPitch, data+dataoffset, total_nPoints*sizeof(float), nPoints*sizeof(float), nDimension, cudaMemcpyHostToDevice));
		}
		else
		{
			for(int nd=0;nd<nDimension;nd++)
			{
				checkCudaErrors(cudaMemcpy(devData+nd*devDataPitchInFloats, data+nd*total_nPoints+dataoffset, nPoints*sizeof(float), cudaMemcpyHostToDevice));	
			}
		}

		dim3 blocksDataLinear(intDivideRoundUp(nPoints, BLOCKSIZE));
		dim3 threadsDots(BLOCKSIZE, 1);
		dim3 blocksDots(intDivideRoundUp(nSV, BLOCKSIZE), intDivideRoundUp(nPoints, BLOCKSIZE));
		int devDotsPitchInFloats = ((int)devDotsPitch)/ sizeof(float);
	
		if(kp.kernel_type.compare(0,3,"rbf")==0)
		{
			makeSelfDots<<<blocksDataLinear, threadsLinear>>>(devData, devDataPitchInFloats, devDataDots, nPoints, nDimension);
		
			checkCudaErrors(cudaMemset(devDots, 0, sizeof(float)*devDotsPitchInFloats*nPoints));

			makeDots<<<blocksDots, threadsDots>>>(devDots, devDotsPitchInFloats, devSVDots, devDataDots, nSV, nPoints);
	
			cudaThreadSynchronize(); //unnecessary..onyl for timing..
		}

		float sgemmAlpha, sgemmBeta;
		if(kp.kernel_type.compare(0,3,"rbf") == 0)
		{
			sgemmAlpha = 2*gamma;
			sgemmBeta = -gamma;
		}
		else
		{
			sgemmAlpha = gamma;
			sgemmBeta = 0.0f;
		}

		cublasSgemm('n', 't', nSV, nPoints, nDimension, sgemmAlpha, devSV, devSVPitchInFloats, devData, devDataPitchInFloats, sgemmBeta, devDots, devDotsPitchInFloats);

		cudaThreadSynchronize();

		int reduceOffset = (int)pow(2, ceil(log2((float)BLOCKSIZE))-1);
		//printf("size: %d -> reduceOffset: %d\n", BLOCKSIZE, reduceOffset);
		int sharedSize = sizeof(float)*(BLOCKSIZE);

    
		if(kp.kernel_type.compare(0,3,"rbf") == 0)
		{
			computeKernelsReduce<<<mapGrid, mapBlock, sharedSize>>>(devDots, devDotsPitchInFloats, devAlpha, nPoints, nSV, RBF, 0,1, devLocalValue, 1<<int(ceil(log2((float)BLOCKSIZE))-1));
		}
		else if(kp.kernel_type.compare(0,10,"polynomial") == 0)
		{
			computeKernelsReduce<<<mapGrid, mapBlock, sharedSize>>>(devDots, devDotsPitchInFloats, devAlpha, nPoints, nSV, POLYNOMIAL, coef0, degree, devLocalValue, 1<<int(ceil(log2((float)BLOCKSIZE))-1));
		}
		else if(kp.kernel_type.compare(0,6,"linear") == 0)
		{
			computeKernelsReduce<<<mapGrid, mapBlock, sharedSize>>>(devDots, devDotsPitchInFloats, devAlpha, nPoints, nSV, LINEAR, 0,1, devLocalValue, 1<<int(ceil(log2((float)BLOCKSIZE))-1));
		}
		else if(kp.kernel_type.compare(0,7,"sigmoid") == 0)
		{
			computeKernelsReduce<<<mapGrid, mapBlock, sharedSize>>>(devDots, devDotsPitchInFloats, devAlpha, nPoints, nSV, SIGMOID, coef0, 1, devLocalValue, 1<<int(ceil(log2((float)BLOCKSIZE))-1));
		}

	
		reduceOffset = (int)pow(2, ceil(log2((float)mapGrid.x))-1);
		sharedSize = sizeof(float)*mapGrid.x;

		doClassification<<<reduceGrid, reduceBlock, sharedSize>>>(devResult, b, devLocalValue, reduceOffset, mapGrid.x);
	
		cudaThreadSynchronize(); //unnecessary..onyl for timing..
	
		//printf("rest of stuff = %f\n",blas1time+(float)((f.tv_sec-s.tv_sec)+(f.tv_usec-s.tv_usec)/1e6));
	
		cudaMemcpy(result+dataoffset, devResult, nPoints*sizeof(float), cudaMemcpyDeviceToHost);


		iteration++;
	}
	
	
	checkCudaErrors(cudaFree(devResult));
	checkCudaErrors(cudaFree(devAlpha));
	checkCudaErrors(cudaFree(devData));
	checkCudaErrors(cudaFree(devLocalValue));
	checkCudaErrors(cudaFree(devDots));
	checkCudaErrors(cudaFree(devSV));
	checkCudaErrors(cudaFree(devSVDots));
	checkCudaErrors(cudaFree(devDataDots));
	checkCudaErrors(cublasShutdown());
}
