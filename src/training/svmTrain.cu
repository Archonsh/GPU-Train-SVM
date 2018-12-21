/*
 * svmTrain.cu
 *
 *  Modified on: May 27, 2016
 *  Modified by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  Augmented power: can train, predict and cross-validation multiple SVMs concurrently using multiple GPUs
 *  Please view the README file for more details
 *
 *  Reference: 
 *    Catanzaro, B., Sundaram, N., & Keutzer, K. (2008). Fast support vector machine training and classification on graphics processors. Proceedings of the 25th International Conference on Machine Learning - ICML '08. doi:10.1145/1390156.1390170
 *    
 */

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <semaphore.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

// randomness
#include <algorithm>
#include <ctime>
#include <random>

#include "svmCommon.h"
#include "../common/svmIO.h"
#include "kernelType.h"

#include "svmTrain.h"
#include "../common/framework.h"
#include "../common/deviceSelect.h"
#include "Cache.h"
#include "Controller.h"
#include "svmKernels.h"
#include "initialize.h"
#include "firstOrder.h"
#include "secondOrder.h"

// for classification
#include "../../include/svmClassify.h"

// include file for matrix transpose
#include "transposeRectangle.cu"

// include file for data processing
#include "processData.h"
#include <unistd.h>
#include "splitfeature.h"

using namespace std;

#define MAXBLOCK 65535

/**
 * Uses multiple GPUs to train [subset], predict [subset] and cross validation many SVMs concurrently (Notice: this function uses the default values for each parameter of different kernels)
 * @param functionality specifies the purpose of using this function: 0-training, 1-prediction, 2-cross validation, 3-subset training, 4-subset prediction
 * @param numSVM the number of SVMs
 * @param dataArray the training data of all SVMs
 * @param nPointsArray records the number of training points for each SVM
 * @param nDimensionArray records the number of dimensions for each SVM
 * @param labelsArray records the lables of all training points
 * @param kpArray specifies the training paramters for each SVM
 * @param costArray the training cost parameter C for each SVM
 * @param heuristicMethodArray specifies selection heuristic method for each SVM.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param epsilonArray this parameter controls which training points are counted as support vectors for each SVM.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param toleranceArray this parameter controls how close to the optimal solution the optimization process must go for each SVM.  Default is 1e-3f.
 * @param transposedDataArray the transposed dataArray
 * @param folder specifies the folder for cross validation
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param modelFilename specifies the output file name for training
 * @param testDataArray the testing data of all SVMs
 * @param testNPointsArray records the number of testing points for each SVM
 * @param predictFilename specifies the output file name for prediction
 * @param testLabelsArray records the labels of all testing point
 * @param subsetIdx records the subset indexes for training or prediction (used when functionality is 3 or 4)
 * @param isMultiClass specifies whether the code is doing cross validation for multi-classes. If so, make suitable changes inside the function to facilitate one-against-all multi-class cross validation
 */
void performMultiGPUTraining(int functionality, int numSVM, float **dataArray, int *nPointsArray, int *nDimensionArray,
                             float **labelsArray, Kernel_params **kpArray, float *costArray,
                             SelectionHeuristic *heuristicMethodArray, float *epsilonArray, float *toleranceArray,
                             float **transposedDataArray, int folder = -1, float ratio = 0.5,
                             char **modelFilename = NULL, float **testDataArray = NULL, int *testNPointsArray = NULL,
                             char **predictFilename = NULL, float **testLabelsArray = NULL, int **subsetIdx = NULL,
                             bool isMultiClass = false);

/**
 * Uses multiple GPUs to train [subset], predict [subset] and cross validation many SVMs concurrently from input files (Notice: this function uses the default values for each parameter of different kernels)
 * @param functionality specifies the purpose of using this function: 0-training, 1-prediction, 2-cross validation, 3-subset training, 4-subset prediction
 * @param numSVM the number of SVM for training
 * @param trainingFile the names of the input training files
 * @param folder specifies the folder for cross validation
 * @param testingFile the names of testing files
 * @param dataFile the names of the file for subset training and prediction (used only when functionality is 3 or 4)
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param kernelType the kernel type for trainging (eg. LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID)
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param cost the training cost parameter C
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 */
void performMultiGPUTrainingFromFiles(int functionality, int numSVM, char **trainingFile, int folder = -1,
                                      char **testingFile = NULL, char *dataFile = NULL, float ratio = 0.5,
                                      int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE,
                                      float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

void performMultiTraining(int numSVM, float **dataArray, int *nPointsArray, int *nDimensionArray, float **labelsArray,
                          float ***p_alphaArray, Kernel_params **kpArray, float *costArray,
                          SelectionHeuristic *heuristicMethodArray, float *epsilonArray, float *toleranceArray,
                          float **transposedDataArray, int gpuId);

// wrapper functions for API, easy to set relevent parameters
void svmTrainFromFile(int numSVM, char **trainingFile, float ratio, int kernelType, SelectionHeuristic heuristicMethod,
                      float cost, float tolerance, float epsilon) {
    int functionality = 0;
    performMultiGPUTrainingFromFiles(functionality, numSVM, trainingFile, -1, NULL, NULL, ratio, kernelType,
                                     heuristicMethod, cost, tolerance, epsilon);
}

void svmPredictFromFile(int numSVM, char **trainingFile, char **testingFile, float ratio, int kernelType,
                        SelectionHeuristic heuristicMethod, float cost, float tolerance, float epsilon) {
    int functionality = 1;
    performMultiGPUTrainingFromFiles(functionality, numSVM, trainingFile, -1, testingFile, NULL, ratio, kernelType,
                                     heuristicMethod, cost, tolerance, epsilon);
}

void svmCrossValidationFromFile(int numSVM, char **trainingFile, int folder, float ratio, int kernelType,
                                SelectionHeuristic heuristicMethod, float cost, float tolerance, float epsilon) {
    int functionality = 2;
    performMultiGPUTrainingFromFiles(functionality, numSVM, trainingFile, folder, NULL, NULL, ratio, kernelType,
                                     heuristicMethod, cost, tolerance, epsilon);
}

void svmSubsetTrainFromFile(int numSVM, char *dataFile, char **subsetFile, float ratio, int kernelType,
                            SelectionHeuristic heuristicMethod, float cost, float tolerance, float epsilon) {
    int functionality = 3;
    performMultiGPUTrainingFromFiles(functionality, numSVM, subsetFile, -1, NULL, dataFile, ratio, kernelType,
                                     heuristicMethod, cost, tolerance, epsilon);
}

void
svmSubsetPredictFromFile(int numSVM, char *dataFile, char **subsetFile, char **testingFile, float ratio, int kernelType,
                         SelectionHeuristic heuristicMethod, float cost, float tolerance, float epsilon) {
    int functionality = 4;
    performMultiGPUTrainingFromFiles(functionality, numSVM, subsetFile, -1, testingFile, dataFile, ratio, kernelType,
                                     heuristicMethod, cost, tolerance, epsilon);
}

void svmTrain(int numSVM, float **dataArray, int *nPointsArray, int *nDimensionArray, float **labelsArray,
              Kernel_params **kpArray, float *costArray, SelectionHeuristic *heuristicMethodArray, float *epsilonArray,
              float *toleranceArray, float **transposedDataArray, char **modelFilename, float ratio) {
    int functionality = 0;
    performMultiGPUTraining(functionality, numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray,
                            costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, -1,
                            ratio, modelFilename);
}

float **mPredictResult = NULL;

void svmPredict(int numSVM, float **dataArray, int *nPointsArray, int *nDimensionArray, float **labelsArray,
                Kernel_params **kpArray, float *costArray, SelectionHeuristic *heuristicMethodArray,
                float *epsilonArray, float *toleranceArray, float **transposedDataArray, float **testDataArray,
                int *testNPointsArray, float **testLabelsArray, char **predictFilename, float ratio, bool isMultiClass,
                float ***inMPredictResult) {
    int functionality = 1;
    performMultiGPUTraining(functionality, numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray,
                            costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, -1,
                            ratio, NULL, testDataArray, testNPointsArray, predictFilename, testLabelsArray, NULL,
                            isMultiClass);
    if (isMultiClass) {
        *(inMPredictResult) = mPredictResult;
    }
}

float **mcvResult = NULL;
float *accuracyResult = NULL;

void
svmCrossValidation(int numSVM, int *nPointsArray, int *nDimensionArray, float **labelsArray, Kernel_params **kpArray,
                   float *costArray, SelectionHeuristic *heuristicMethodArray, float *epsilonArray,
                   float *toleranceArray, float **transposedDataArray, int folder, float ratio, bool isMultiClass,
                   float ***inMcvResult, bool outputAccuracy, float *inAccuracyResult) {

    int functionality = 2;
    //isMultiClass = true;
    if (outputAccuracy)
        accuracyResult = (float *) malloc(sizeof(float) * numSVM);
    performMultiGPUTraining(functionality, numSVM, NULL, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                            heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, ratio,
                            NULL, NULL, NULL, NULL, NULL, NULL, isMultiClass);
    if (isMultiClass) {
        *(inMcvResult) = mcvResult;
    }
    if (outputAccuracy) {
        int i;
        for (i = 0; i < numSVM; i++)
            inAccuracyResult[i] = accuracyResult[i];
        free(accuracyResult);
    }
}

void svmSubsetTrain(int numSVM, int *nPointsArray, int *nDimensionArray, float **labelsArray, Kernel_params **kpArray,
                    float *costArray, SelectionHeuristic *heuristicMethodArray, float *epsilonArray,
                    float *toleranceArray, float **transposedDataArray, int **subsetIdx, char **modelFilename,
                    float ratio) {
    int functionality = 3;
    performMultiGPUTraining(functionality, numSVM, NULL, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                            heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, -1, ratio,
                            modelFilename, NULL, NULL, NULL, NULL, subsetIdx);
}

void svmSubsetPredict(int numSVM, int *nPointsArray, int *nDimensionArray, float **labelsArray, Kernel_params **kpArray,
                      float *costArray, SelectionHeuristic *heuristicMethodArray, float *epsilonArray,
                      float *toleranceArray, float **transposedDataArray, int **subsetIdx, float **testDataArray,
                      int *testNPointsArray, float **testLabelsArray, char **predictFilename, float ratio) {
    int functionality = 4;
    performMultiGPUTraining(functionality, numSVM, NULL, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                            heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, -1, ratio, NULL,
                            testDataArray, testNPointsArray, predictFilename, testLabelsArray, subsetIdx);
}

void printHelp() {
    printf("Usage 1: svmTrain [options] trainingData.svm\n");
    printf("Usage 2: svmTrain -x 0 trainingData_1.svm ... trainingData_n.svm\n");
    printf("Usage 3: svmTrain -x 1 trainingData_1.svm ... trainingData_n.svm testingData_1 ... testingData_n\n");
    printf("Usage 4: svmTrain -x 2 -b n trainingData_1.svm ... trainingData_n.svm\n");
    printf("Usage 5: svmTrain -x 3 trainingData.svm subset_1.svm ... subset_n.svm\n");
    printf("Usage 6: svmTrain -x 4 trainingData.svm subset_1.svm ... subset_n.svm testingData_1 ... testingData_n\n");
    printf("Options:\n");
    printf("\t-o outputFilename\t Location of output file\n");
    printf("\t-x functionality\t Specify the purpose of using this program: 0-training, 1-prediction, 2-cross validation\n");
    printf("Cross validation:\n");
    printf("\t-b\t Number of folder for cross validation (default = 5)\n");
    printf("Kernel types:\n");
    printf("\t--gaussian\tGaussian or RBF kernel (default): Phi(x, y; gamma) = exp{-gamma*||x-y||^2}\n");
    printf("\t--linear\tLinear kernel: Phi(x, y) = x . y\n");
    printf("\t--polynomial\tPolynomial kernel: Phi(x, y; a, r, d) = (ax . y + r)^d\n");
    printf("\t--sigmoid\tSigmoid kernel: Phi(x, y; a, r) = tanh(ax . y + r)\n");
    printf("Parameters:\n");
    printf("\t-c, --cost\tSVM training cost C (default = 1)\n");
    printf("\t-g\tGamma for Gaussian kernel (default = 1/#dimension)\n");
    printf("\t-a\tParameter a for Polynomial and Sigmoid kernels (default = 1/l)\n");
    printf("\t-r\tParameter r for Polynomial and Sigmoid kernels (default = 1)\n");
    printf("\t-d\tParameter d for Polynomial kernel (default = 3)\n");
    printf("Convergence parameters:\n");
    printf("\t--tolerance, -t\tTermination criterion tolerance (default = 0.001)\n");
    printf("\t--epsilon, -e\tSupport vector threshold (default = 1e-5)\n");
    printf("Internal options:\n");
    printf("\t--heuristic, -h\tWorking selection heuristic:\n");
    printf("\t\t0: First order\n");
    printf("\t\t1: Second order\n");
    printf("\t\t2: Random (either first or second order)\n");
    printf("\t\t3: Adaptive (default)\n");
}

// Calculate the memory needed for a single svm with dimension "dim", points "numPoint", and cache-ratio "ratio".
// The output size is an upper bound for the memory needed for a single SVM included the cache size for kernel matrixes.
size_t calMemForSingleSVM(int numPoint, int dim, float ratio) {
    size_t size = 0;
    int power = 0;
    while (pow(2, power) < numPoint) {
        power++;
    }
    int pitch = (int) pow(2, power); // = upper bound of points
    size += pitch * sizeof(float) * dim * 2;
    size += numPoint * sizeof(float) * 4;
    size += ((numPoint / BLOCKSIZE) + 1) * sizeof(float) * 3;
    size += ((numPoint / BLOCKSIZE) + 1) * sizeof(int) * 3;
    size += pitch * sizeof(float) * 2;
    size += (int) (pitch * sizeof(float) * numPoint * ratio);
    return size;
}

// data structure for passing variables to gpu thread
struct gpuTrainData {
    int gpuId;
    int numSVMGpu;
    float **inData;
    int *inNPoints;
    int *inNDimension;
    float **inLabels;
    float **inPAlpha;
    Kernel_params **inKp;
    float *inCost;
    SelectionHeuristic *inHeuristicMethod;
    float *inEpsilon;
    float *inTolerance;
    float **inTransposedData;
};

sem_t trainMutex;
bool *isAvailableForTraining;
float *trainTimePerGpu;

/**
 * This is the gpu thread for training, cross validation and prediction on multiple SVMs
 * threadarg->functionality = 0 : training multiple SVMs and output model files
 * threadarg->functionality = 1 : training multiple SVM models and using them to predict test points
 * threadarg->functionality = 2 : cross validation on multiple SVMs
 */
void *gpuTrainThread(void *threadarg) {
    struct gpuTrainData *myData;
    myData = (struct gpuTrainData *) threadarg;

    int gpuId = myData->gpuId;
    checkCudaErrors(cudaSetDevice(gpuId));

    int numSVMGpu = myData->numSVMGpu;
    float **inData = myData->inData;
    int *inNPoints = myData->inNPoints;
    int *inNDimension = myData->inNDimension;
    float **inLabels = myData->inLabels;
    float **inPAlpha = myData->inPAlpha;
    Kernel_params **inKp = myData->inKp;
    float *inCost = myData->inCost;
    SelectionHeuristic *inHeuristicMethod = myData->inHeuristicMethod;
    float *inEpsilon = myData->inEpsilon;
    float *inTolerance = myData->inTolerance;
    float **inTransposedData = myData->inTransposedData;

    struct timeval trainStart;
    gettimeofday(&trainStart, 0);

    // Using "gpuId"-GPU to train multiple SVMs concurrently
    performMultiTraining(numSVMGpu, inData, inNPoints, inNDimension, inLabels, &inPAlpha, inKp, inCost,
                         inHeuristicMethod, inEpsilon, inTolerance, inTransposedData, gpuId);

    struct timeval trainEnd;
    gettimeofday(&trainEnd, 0);
    float trainingTime =
            (float) (trainEnd.tv_sec - trainStart.tv_sec) + ((float) (trainEnd.tv_usec - trainStart.tv_usec)) * 1e-6;
    trainTimePerGpu[gpuId] += trainingTime;

    sem_wait(&trainMutex);
    isAvailableForTraining[gpuId] = true;
    sem_post(&trainMutex);

    return NULL;
}

struct gpuPredictData {
    int gpuId;
    int svmId;
    int numSVM;

    int *nPointsArray;
    float **inPAlphaAllSVMs;
    float *epsilonArray;
    int *nDimensionArray;
    float **labelsArray;
    float **dataArray;
    Kernel_params **kpArray;

    float **result;
    float **testDataArray;
    int *testNPointsArray;
};

//the nextSVM for prediction is shared across GPUs and protected by a semaphore
sem_t predictMutex;
int nextSVMForPrediction = 0;
int *limits;
int interval = 10000;

void *gpuPredictThread(void *threadarg) {
    struct gpuPredictData *myData;
    myData = (struct gpuPredictData *) threadarg;

    int gpuId = myData->gpuId;
    checkCudaErrors(cudaSetDevice(gpuId));
    int svmId = myData->svmId;
    int numSVM = myData->numSVM;

    int *nPointsArray = myData->nPointsArray;
    float **inPAlphaAllSVMs = myData->inPAlphaAllSVMs;
    float *epsilonArray = myData->epsilonArray;
    int *nDimensionArray = myData->nDimensionArray;
    float **labelsArray = myData->labelsArray;
    float **dataArray = myData->dataArray;
    Kernel_params **kpArray = myData->kpArray;

    float **result = myData->result;
    float **testDataArray = myData->testDataArray;
    int *testNPointsArray = myData->testNPointsArray;

    while (svmId < numSVM) {
        int nSV = 0;
        int j;
        for (j = 0; j < nPointsArray[svmId]; j++) {
            if (inPAlphaAllSVMs[svmId][j] > epsilonArray[svmId]) // eps default = 1e-5, if alpha > eps, then alpha > 0
                nSV++;
        }

        float *supportVectors = (float *) malloc(sizeof(float) * nSV * nDimensionArray[svmId]);
        float *alpha = (float *) malloc(sizeof(float) * nSV);
        int count = 0;
        for (j = 0; j < nPointsArray[svmId]; j++) {
            if (inPAlphaAllSVMs[svmId][j] > epsilonArray[svmId]) {
                int p;
                alpha[count] = inPAlphaAllSVMs[svmId][j] * labelsArray[svmId][j]; //alpha_i * y_i
                for (p = 0; p < nDimensionArray[svmId]; p++) {
                    supportVectors[p * nSV + count] = dataArray[svmId][p * nPointsArray[svmId] + j];
                }
                count++;
            }
        }

        // modify inKp for classification, refer to line 80 of file SVMIO.cpp
        kpArray[svmId]->b = -kpArray[svmId]->b;
        cout << "GPU " << gpuId << " start predicting SVM " << svmId << endl;
        performClassification(testDataArray[svmId], testNPointsArray[svmId], supportVectors, nSV,
                              nDimensionArray[svmId], alpha, *(kpArray[svmId]), &(result[svmId]), gpuId);
        cout << "GPU " << gpuId << " finish predicting SVM " << svmId << endl;

        if (svmId > limits[gpuId]) {
            checkCudaErrors(cudaDeviceReset());
            limits[gpuId] += interval;
        }

        free(supportVectors);
        free(alpha);

        sem_wait(&predictMutex);
        svmId = nextSVMForPrediction;
        nextSVMForPrediction++;
        sem_post(&predictMutex);

    }

    return NULL;
}


struct gpuSubsetProcessData {
    int gpuId;
    int svmId;
    int numSVM;
    int *nPointsArray;
    int *nDimensionArray;
    int **subsetIdx;
    float **labelsArray;
    float **transposedDataArray;

    float **labelsArrayTmp;
    float **transposedDataArrayTmp;
    float **dataArrayTmp;
};

sem_t subsetProcessMutex;
int nextSVMForSubsetProcess = 0;
int *subsetLimits;
int subsetInterval = 10000;

void *gpuSubsetProcessThread(void *threadarg) {
    struct gpuSubsetProcessData *myData;
    myData = (struct gpuSubsetProcessData *) threadarg;

    int gpuId = myData->gpuId;
    checkCudaErrors(cudaSetDevice(gpuId));
    int svmId = myData->svmId;
    int numSVM = myData->numSVM;
    int *nPointsArray = myData->nPointsArray;
    int *nDimensionArray = myData->nDimensionArray;
    int **subsetIdx = myData->subsetIdx;
    float **labelsArray = myData->labelsArray;
    float **transposedDataArray = myData->transposedDataArray;

    float **labelsArrayTmp = myData->labelsArrayTmp;
    float **transposedDataArrayTmp = myData->transposedDataArrayTmp;
    float **dataArrayTmp = myData->dataArrayTmp;

    while (svmId < numSVM) {
        int p;
        labelsArrayTmp[svmId] = (float *) malloc(sizeof(float) * nPointsArray[svmId]);
        dataArrayTmp[svmId] = (float *) malloc(sizeof(float) * nPointsArray[svmId] * nDimensionArray[svmId]);
        transposedDataArrayTmp[svmId] = (float *) malloc(sizeof(float) * nPointsArray[svmId] * nDimensionArray[svmId]);
        cout << "GPU " << gpuId << " is processing SVM " << svmId << endl;
        for (p = 0; p < nPointsArray[svmId]; p++) {
            int pointIdx = subsetIdx[svmId][p];
            labelsArrayTmp[svmId][p] = labelsArray[0][pointIdx];
            memcpy(transposedDataArrayTmp[svmId] + p * nDimensionArray[svmId],
                   transposedDataArray[0] + pointIdx * nDimensionArray[svmId], sizeof(float) * nDimensionArray[svmId]);
        }

        transposeMatrix(transposedDataArrayTmp[svmId], dataArrayTmp[svmId], nPointsArray[svmId],
                        nDimensionArray[svmId]);
        cout << "GPU " << gpuId << " finish processing SVM " << svmId << endl;

        if (svmId > subsetLimits[gpuId]) {
            checkCudaErrors(cudaDeviceReset());
            subsetLimits[gpuId] += subsetInterval;
        }

        sem_wait(&subsetProcessMutex);
        svmId = nextSVMForSubsetProcess;
        nextSVMForSubsetProcess++;
        sem_post(&subsetProcessMutex);
    }

    return NULL;
}

float cvDataProcessTime = 0;
float subsetDataProcessTime = 0;
float trainTime = 0;
float predictTime = 0;
float outputTime = 0;

// using multiple GPU train SVM with cache-ratio "ratio"
void performMultiGPUTraining(int functionality, int numSVM, float **dataArray, int *nPointsArray, int *nDimensionArray,
                             float **labelsArray, Kernel_params **kpArray, float *costArray,
                             SelectionHeuristic *heuristicMethodArray, float *epsilonArray, float *toleranceArray,
                             float **transposedDataArray, int folder, float ratio, char **modelFilename,
                             float **testDataArray, int *testNPointsArray, char **predictFilename,
                             float **testLabelsArray, int **subsetIdx, bool isMultiClass) {
    int i;
    int rc;
    int ngpus;

    // setup time variable and get GPU memory info
    checkCudaErrors(cudaGetDeviceCount(&ngpus));
    trainTimePerGpu = (float *) malloc(sizeof(float) * ngpus);
    size_t freeMem[ngpus];
    size_t totalMem[ngpus];
    int numSVMPerGpu[ngpus];
    isAvailableForTraining = (bool *) malloc(sizeof(bool) * ngpus);
    for (i = 0; i < ngpus; i++) {
        trainTimePerGpu[i] = 0;
        isAvailableForTraining[i] = true;
        numSVMPerGpu[i] = 0;
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaMemGetInfo(&freeMem[i], &totalMem[i]));
        printf("Total mem: %Zu, Free mem: %Zu, Used mem: %Zu\n", totalMem[i], freeMem[i], totalMem[i] - freeMem[i]);
    }

    int *processDataInfo;
    // preprocess data
    switch (functionality) {
        // random split data for folder cross validation
        case 2: {
            // check folder
            if (folder <= 1) {
                cout << "Error: folder should be bigger than one!" << endl;
                return;
            }

            bool changeFolder = false;
            for (i = 0; i < numSVM; i++) {
                if (folder > nPointsArray[i]) {
                    folder = nPointsArray[i];
                    changeFolder = true;
                }
            }

            if (changeFolder)
                cout
                        << "Notice: change the value of folder to be equal to number of points, otherwise cannot partition the data for cross valiation!"
                        << endl;

            // record cross validation data splitting time
            struct timeval cvDataSplitStart;
            gettimeofday(&cvDataSplitStart, 0);
            int numSVMTmp = numSVM * folder;

            float **dataArrayTmp;
            int *nPointsArrayTmp;
            int *nDimensionArrayTmp;
            float **labelsArrayTmp;
            Kernel_params **kpArrayTmp;
            float *costArrayTmp;
            SelectionHeuristic *heuristicMethodArrayTmp;
            float *epsilonArrayTmp;
            float *toleranceArrayTmp;
            float **transposedDataArrayTmp;

            // variables for prediction
            float **testDataArrayTmp;
            int *testNPointsArrayTmp;
            float **testLabelsArrayTmp;

            dataArrayTmp = (float **) malloc(sizeof(float *) * numSVMTmp);
            nPointsArrayTmp = (int *) malloc(sizeof(int) * numSVMTmp);
            nDimensionArrayTmp = (int *) malloc(sizeof(int) * numSVMTmp);

            //not altered by processDataInfo(), directly go to 2 stage
            labelsArrayTmp = (float **) malloc(sizeof(float *) * numSVMTmp);
            kpArrayTmp = (Kernel_params **) malloc(sizeof(Kernel_params *) * numSVMTmp);
            costArrayTmp = (float *) malloc(sizeof(float) * numSVMTmp);
            heuristicMethodArrayTmp = (SelectionHeuristic *) malloc(sizeof(SelectionHeuristic) * numSVMTmp);
            epsilonArrayTmp = (float *) malloc(sizeof(float) * numSVMTmp);
            toleranceArrayTmp = (float *) malloc(sizeof(float) * numSVMTmp);
            transposedDataArrayTmp = (float **) malloc(sizeof(float *) * numSVMTmp);
            //

            testDataArrayTmp = (float **) malloc(sizeof(float *) * numSVMTmp);
            testNPointsArrayTmp = (int *) malloc(sizeof(int) * numSVMTmp);
            testLabelsArrayTmp = (float **) malloc(sizeof(float *) * numSVMTmp);

            // random split the datasets into "folder" parts
            srand(unsigned(time(0)));
            int **permutation = (int **) malloc(sizeof(int *) * numSVM);
            for (i = 0; i < numSVM; i++)
            {
                permutation[i] = (int *) malloc(nPointsArray[i] * sizeof(int));
                int j;
                if (isMultiClass && i > 0) {
                    // make the permutation the same for all classes to facilitate one-against-all multi-class cross validation
                    memcpy(permutation[i], permutation[i - 1], sizeof(int) * nPointsArray[i]);
                } else {
                    for (j = 0; j < nPointsArray[i]; j++)
                        permutation[i][j] = j;
                    if (!isMultiClass)
                        random_shuffle(permutation[i], permutation[i] + nPointsArray[i]);
                }
            }
            //permutation[i] is the random_shuffle of i_th SVM's data points

            for (i = 0; i < numSVM; i++) {
                int j;
                int tmpValueA = nPointsArray[i] / folder; // # of points in a fold
                int tmpValueB = nPointsArray[i] % folder; // # of point left
                int numPointTrainPartition = 0;
                int numPointTestPartition = 0;
                for (j = 0; j < folder; j++) {
                    if (j + 1 > tmpValueB) {
                        numPointTrainPartition = nPointsArray[i] - tmpValueA;
                        numPointTestPartition = tmpValueA;
                    } else {
                        numPointTrainPartition = nPointsArray[i] - (tmpValueA + 1);
                        numPointTestPartition = tmpValueA + 1;
                    }

                    dataArrayTmp[i * folder + j] = (float *) malloc(
                            sizeof(float) * nDimensionArray[i] * numPointTrainPartition);
                    transposedDataArrayTmp[i * folder + j] = (float *) malloc(
                            sizeof(float) * nDimensionArray[i] * numPointTrainPartition);
                    labelsArrayTmp[i * folder + j] = (float *) malloc(sizeof(float) * numPointTrainPartition);

                    testDataArrayTmp[i * folder + j] = (float *) malloc(
                            sizeof(float) * nDimensionArray[i] * numPointTestPartition);
                    testLabelsArrayTmp[i * folder + j] = (float *) malloc(sizeof(float) * numPointTestPartition);

                    int p;
                    int countTrainPoint = 0;
                    int countTestPoint = 0;
                    for (p = 0; p < nPointsArray[i]; p++) {
                        if (p % folder == j) {
                            int testPointIdx = permutation[i][p];
                            testLabelsArrayTmp[i * folder + j][countTestPoint] = labelsArray[i][testPointIdx];
                            countTestPoint++;
                        } else {
                            int trainPointIdx = permutation[i][p];
                            labelsArrayTmp[i * folder + j][countTrainPoint] = labelsArray[i][trainPointIdx];
                            countTrainPoint++;
                        }
                    }

                    nPointsArrayTmp[i * folder + j] = numPointTrainPartition;
                    nDimensionArrayTmp[i * folder + j] = nDimensionArray[i];
                    kpArrayTmp[i * folder + j] = (Kernel_params *) malloc(sizeof(Kernel_params));
                    memcpy(kpArrayTmp[i * folder + j], kpArray[i], sizeof(Kernel_params));
                    costArrayTmp[i * folder + j] = costArray[i];
                    heuristicMethodArrayTmp[i * folder + j] = heuristicMethodArray[i];
                    epsilonArrayTmp[i * folder + j] = epsilonArray[i];
                    toleranceArrayTmp[i * folder + j] = toleranceArray[i];
                    testNPointsArrayTmp[i * folder + j] = numPointTestPartition;
                }
            }

            //split data according to the folder
            processDataInfo = processData(ngpus, numSVM, folder, transposedDataArray, nPointsArray, nDimensionArray,
                                          permutation, transposedDataArrayTmp, dataArrayTmp, NULL, testDataArrayTmp);
            dataArray = dataArrayTmp;
            nPointsArray = nPointsArrayTmp;
            nDimensionArray = nDimensionArrayTmp;
            labelsArray = labelsArrayTmp;
            kpArray = kpArrayTmp;
            costArray = costArrayTmp;
            heuristicMethodArray = heuristicMethodArrayTmp;
            epsilonArray = epsilonArrayTmp;
            toleranceArray = toleranceArrayTmp;
            transposedDataArray = transposedDataArrayTmp;

            testDataArray = testDataArrayTmp;
            testNPointsArray = testNPointsArrayTmp;
            testLabelsArray = testLabelsArrayTmp;

            // free variable
            for (i = 0; i < numSVM; i++) {
                free(permutation[i]);
            }
            free(permutation);

            numSVM = numSVMTmp;

            float **dataArrayOut, **transposedDataArrayOut, **labelsArrayOut;
            int *nPointsArrayOut, *nDimensionArrayOut;
            Kernel_params **kpArrayOut;
            float *costArrayOut;
            SelectionHeuristic *heuristicMethodArrayOut;
            float *epsilonArrayOut;
            float *toleranceArrayOut;
            float **testDataArrayOut;
            int *testNPointsArrayOut;
            float **testLabelsArrayOut;

            int numSVMfinal = numSVM * nDimensionArray[0] * 2;

            dataArrayOut = (float **) malloc(sizeof(float *) * numSVMfinal);
            transposedDataArrayOut = (float **) malloc(sizeof(float *) * numSVMfinal);
            labelsArrayOut = (float **) malloc(sizeof(float *) * numSVMfinal);

            nPointsArrayOut = (int *) malloc(sizeof(int) * numSVMfinal);
            nDimensionArrayOut = (int *) malloc(sizeof(int) * numSVMfinal);

            kpArrayOut = (Kernel_params **) malloc(sizeof(Kernel_params * ) * numSVMfinal);
            costArrayOut = (float *) malloc(sizeof(float) * numSVMfinal);
            heuristicMethodArrayOut = (SelectionHeuristic *) malloc(sizeof(SelectionHeuristic) * numSVMfinal);
            epsilonArrayOut = (float *) malloc(sizeof(float) * numSVMfinal);
            toleranceArrayOut = (float *) malloc(sizeof(float) * numSVMfinal);

            testDataArrayOut = (float **) malloc(sizeof(float *) * numSVMfinal);
            testLabelsArrayOut = (float **) malloc(sizeof(float *) * numSVMfinal);
            testNPointsArrayOut = (int *) malloc(sizeof(int) * numSVMfinal);

            struct timeval featureSplitStart;
            gettimeofday(&featureSplitStart, 0);

            //split data according to the feature
            splitfeatures(numSVM,
                          numSVMfinal,
                          folder,
                          nPointsArray,
                          nPointsArrayOut,
                          nDimensionArray,
                          nDimensionArrayOut,
                          dataArray,
                          dataArrayOut,
                          transposedDataArray,
                          transposedDataArrayOut,
                          testDataArray,
                          testDataArrayOut,
                          testNPointsArray,
                          testNPointsArrayOut,
                          testLabelsArray,
                          testLabelsArrayOut,
                          labelsArray,
                          labelsArrayOut,
                          kpArray,
                          kpArrayOut,
                          costArray,
                          costArrayOut,
                          heuristicMethodArray,
                          heuristicMethodArrayOut,
                          epsilonArray,
                          epsilonArrayOut,
                          toleranceArray,
                          toleranceArrayOut);
            numSVM = numSVMfinal;
            //original mem space fred in splitfeatures()
            dataArray = dataArrayOut;
            transposedDataArray = transposedDataArrayOut;
            labelsArray = labelsArrayOut;
            nPointsArray = nPointsArrayOut;
            nDimensionArray = nDimensionArrayOut;
            kpArray = kpArrayOut;
            costArray = costArrayOut;
            heuristicMethodArray = heuristicMethodArrayOut;
            epsilonArray = epsilonArrayOut;
            toleranceArray = toleranceArrayOut;
            testDataArray = testDataArrayOut;
            testNPointsArray = testNPointsArrayOut;
            testLabelsArray = testLabelsArrayOut;

            struct timeval cvDataSplitEnd;
            gettimeofday(&cvDataSplitEnd, 0);

            printf("Data Split time: %f\n", (float) (cvDataSplitEnd.tv_sec - featureSplitStart.tv_sec)) +
                                            ((float) (cvDataSplitEnd.tv_usec - featureSplitStart.tv_usec));

            cvDataProcessTime = (float) (cvDataSplitEnd.tv_sec - cvDataSplitStart.tv_sec) +
                                ((float) (cvDataSplitEnd.tv_usec - cvDataSplitStart.tv_usec)) * 1e-6;
        }
            break;
            // create subset data from subsetIdx
        case 3:
        case 4: {
            // record creating subset time
            struct timeval subsetDataStart;
            gettimeofday(&subsetDataStart, 0);
            float **dataArrayTmp;
            float **labelsArrayTmp;
            float **transposedDataArrayTmp;

            dataArrayTmp = (float **) malloc(sizeof(float *) * numSVM);
            labelsArrayTmp = (float **) malloc(sizeof(float *) * numSVM);
            transposedDataArrayTmp = (float **) malloc(sizeof(float *) * numSVM);

            sem_init(&subsetProcessMutex, 0, 1);

            int i, rc;

            subsetLimits = (int *) malloc(sizeof(int) * ngpus);
            for (i = 0; i < ngpus; i++)
                subsetLimits[i] = subsetInterval;

            pthread_attr_t attr;
            void *status;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
            vector < pthread_t * > threads;
            pthread_t *newThread;
            struct gpuSubsetProcessData *td;

            nextSVMForSubsetProcess = ngpus;
            for (i = 0; i < ngpus; i++) {
                td = (struct gpuSubsetProcessData *) malloc(sizeof(struct gpuSubsetProcessData));
                td->gpuId = i;
                td->svmId = i;
                td->numSVM = numSVM;
                td->nPointsArray = nPointsArray;
                td->nDimensionArray = nDimensionArray;
                td->subsetIdx = subsetIdx;
                td->labelsArray = labelsArray;
                td->transposedDataArray = transposedDataArray;
                td->labelsArrayTmp = labelsArrayTmp;
                td->transposedDataArrayTmp = transposedDataArrayTmp;
                td->dataArrayTmp = dataArrayTmp;

                newThread = (pthread_t *) malloc(sizeof(pthread_t));
                // create new thread for training multple SVMs
                rc = pthread_create(newThread, NULL, gpuSubsetProcessThread, (void *) td);
                threads.push_back(newThread);
            }

            // free attribute and wait for the other threads
            pthread_attr_destroy(&attr);
            for (i = 0; i < threads.size(); i++) {
                rc = pthread_join(*threads[i], &status);
                if (rc) {
                    cout << "Error: unable to join, " << rc << endl;
                    exit(-1);
                }
            }

            dataArray = dataArrayTmp;
            labelsArray = labelsArrayTmp;
            transposedDataArray = transposedDataArrayTmp;

            struct timeval subsetDataEnd;
            gettimeofday(&subsetDataEnd, 0);
            subsetDataProcessTime = (float) (subsetDataEnd.tv_sec - subsetDataStart.tv_sec) +
                                    ((float) (subsetDataEnd.tv_usec - subsetDataStart.tv_usec)) * 1e-6;
        }
            break;
    }

    // a records the prediction accuracy for all SVMs
    double *accuInfo;
    switch (functionality) {
        case 1:
        case 2:
        case 4:
            accuInfo = (double *) malloc(sizeof(double) * numSVM);
            break;
        default:
            break;
    }
    struct timeval trainStart;
    gettimeofday(&trainStart, 0);

    float **inPAlphaAllSVMs = (float **) malloc(sizeof(float *) * numSVM);

    // initialize trainMutex
    sem_init(&trainMutex, 0, 1);  // init a semaphore, shared within intra-process threads, init value = 1

    // initialize and set thread joinable
    pthread_attr_t attr;
    void *status;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    vector < pthread_t * > threads;
    int nextSVM = 0;
    pthread_t *newThread;
    struct gpuTrainData *td;

    // load work to each GPU
    while (nextSVM < numSVM) {
        int availableGPUId = -1;

        //critical section
        sem_wait(&trainMutex);
        for (i = 0; i < ngpus; i++) {
            if (isAvailableForTraining[i]) {
                availableGPUId = i;
                isAvailableForTraining[i] = false;
                break;
            }
        }
        sem_post(&trainMutex);
        //end

        if (availableGPUId != -1 && nextSVM < numSVM) {
            int numSVMGpu = 0;
            size_t size = 0;
            int index = nextSVM;
            //nextSVM is the starting SVM of the cur GPU


            while (size < freeMem[availableGPUId] * 0.95 && index < numSVM) {
                numSVMGpu++;
                size += calMemForSingleSVM(nPointsArray[index], nDimensionArray[index], ratio);
                index++;
            }

            int maxPointNum = 0; // this should be the same for feature selection work
            for (i = nextSVM; i < index; i++) {
                if (nPointsArray[i] > maxPointNum)
                    maxPointNum = nPointsArray[i];
            }

            int width = intDivideRoundUp(maxPointNum, BLOCKSIZE);
            //width = number of threads needed, 1 thread - 1 point

            if (numSVMGpu * (width + 1) > MAXBLOCK)
                numSVMGpu = MAXBLOCK / (width + 1);

            // set variables for each gpu thread
            // copy pointers to the data only / no actual data movement
            float **inData;
            int *inNPoints;
            int *inNDimension;
            float **inLabels;
            Kernel_params **inKp;
            float *inCost;
            SelectionHeuristic *inHeuristicMethod;
            float *inEpsilon;
            float *inTolerance;
            float **inTransposedData;

            inData = (float **) malloc(sizeof(float *) * numSVMGpu);
            inNPoints = (int *) malloc(sizeof(int) * numSVMGpu);
            inNDimension = (int *) malloc(sizeof(int) * numSVMGpu);
            inLabels = (float **) malloc(sizeof(float *) * numSVMGpu);
            inKp = (Kernel_params **) malloc(sizeof(Kernel_params *) * numSVMGpu);
            inCost = (float *) malloc(sizeof(float) * numSVMGpu);
            inHeuristicMethod = (SelectionHeuristic *) malloc(sizeof(SelectionHeuristic) * numSVMGpu);
            inEpsilon = (float *) malloc(sizeof(float) * numSVMGpu);
            inTolerance = (float *) malloc(sizeof(float) * numSVMGpu);
            inTransposedData = (float **) malloc(sizeof(float *) * numSVMGpu);
            for (int j = 0; j < numSVMGpu; j++) {
                inData[j] = dataArray[nextSVM + j]; //? (nextSVM+j) ? SVM
                inNPoints[j] = nPointsArray[nextSVM + j];
                inNDimension[j] = nDimensionArray[nextSVM + j];
                inLabels[j] = labelsArray[nextSVM + j];
                inKp[j] = kpArray[nextSVM + j];
                inCost[j] = costArray[nextSVM + j];
                inHeuristicMethod[j] = heuristicMethodArray[nextSVM + j];
                inEpsilon[j] = epsilonArray[nextSVM + j];
                inTolerance[j] = toleranceArray[nextSVM + j];
                inTransposedData[j] = transposedDataArray[nextSVM + j];
            }

            float **inPAlpha = &(inPAlphaAllSVMs[nextSVM]);

            td = (struct gpuTrainData *) malloc(sizeof(struct gpuTrainData));
            td->gpuId = availableGPUId;
            td->numSVMGpu = numSVMGpu;
            td->inData = inData;
            td->inNPoints = inNPoints;
            td->inNDimension = inNDimension;
            td->inLabels = inLabels;
            td->inPAlpha = inPAlpha;
            td->inKp = inKp;
            td->inCost = inCost;
            td->inHeuristicMethod = inHeuristicMethod;
            td->inEpsilon = inEpsilon;
            td->inTolerance = inTolerance;
            td->inTransposedData = inTransposedData;

            newThread = (pthread_t *) malloc(sizeof(pthread_t));
            // create new thread for training multple SVMs
            rc = pthread_create(newThread, NULL, gpuTrainThread, (void *) td);
            if (rc) {
                cout << "Error: unable to create thread, " << rc << endl;
                exit(-1);
            }
            threads.push_back(newThread);
            nextSVM += numSVMGpu;
            numSVMPerGpu[availableGPUId] += numSVMGpu;
        }
    }

    // free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for (i = 0; i < threads.size(); i++) {
        rc = pthread_join(*threads[i], &status);
        if (rc) {
            cout << "Error: unable to join, " << rc << endl;
            exit(-1);
        }
    }

    struct timeval trainEnd;
    gettimeofday(&trainEnd, 0);
    trainTime =
            (float) (trainEnd.tv_sec - trainStart.tv_sec) + ((float) (trainEnd.tv_usec - trainStart.tv_usec)) * 1e-6;

    switch (functionality) {
        case 0:
        case 3: {
            struct timeval outputStart;
            gettimeofday(&outputStart, 0);
            // printout the model files
//        if(modelFilename == NULL) {
//          for(i = 0; i < numSVM; i++) {
//            stringstream ss;
//            ss << i;
//            string filename = "./data/svm" + ss.str() + ".mdl";
//            char oFilename[1024];
//            strncpy(oFilename, filename.c_str(), sizeof(oFilename));
//            printModel(oFilename, *(kpArray[i]), inPAlphaAllSVMs[i], labelsArray[i], dataArray[i], nPointsArray[i], nDimensionArray[i], epsilonArray[i]);
//          }
//        } else {
//          for(i = 0; i < numSVM; i++)
//            printModel(modelFilename[i], *(kpArray[i]), inPAlphaAllSVMs[i], labelsArray[i], dataArray[i], nPointsArray[i], nDimensionArray[i], epsilonArray[i]);
//        }

            struct timeval outputEnd;
            gettimeofday(&outputEnd, 0);
            outputTime = (float) (outputEnd.tv_sec - outputStart.tv_sec) +
                         ((float) (outputEnd.tv_usec - outputStart.tv_usec)) * 1e-6;
        }
            break;
        case 1:
        case 2:
        case 4: {
            struct timeval predictStart;
            gettimeofday(&predictStart, 0);

            // initialize trainMutex
            sem_init(&predictMutex, 0, 1);

            limits = (int *) malloc(sizeof(int) * ngpus);
            for (i = 0; i < ngpus; i++)
                limits[i] = interval;

            float **result = (float **) malloc(sizeof(float *) * numSVM);
            vector < pthread_t * > threads;
            pthread_t *newThread;
            struct gpuPredictData *td;

            nextSVMForPrediction = ngpus;
            for (i = 0; i < ngpus; i++) {
                td = (struct gpuPredictData *) malloc(sizeof(struct gpuPredictData));

                td->gpuId = i;
                td->svmId = i;
                td->numSVM = numSVM;
                td->nPointsArray = nPointsArray;
                td->inPAlphaAllSVMs = inPAlphaAllSVMs;
                td->epsilonArray = epsilonArray;
                td->nDimensionArray = nDimensionArray;
                td->labelsArray = labelsArray;
                td->dataArray = dataArray;
                td->kpArray = kpArray;
                td->result = result;
                td->testDataArray = testDataArray;
                td->testNPointsArray = testNPointsArray;

                newThread = (pthread_t *) malloc(sizeof(pthread_t));
                // create new thread for training multple SVMs
                rc = pthread_create(newThread, NULL, gpuPredictThread, (void *) td);
                threads.push_back(newThread);
            }

            // free attribute and wait for the other threads
            pthread_attr_destroy(&attr);
            for (i = 0; i < threads.size(); i++) {
                rc = pthread_join(*threads[i], &status);
                if (rc) {
                    cout << "Error: unable to join, " << rc << endl;
                    exit(-1);
                }
            }


            // reconstruct the seeds
            int total_dim;
            if(nDimensionArray[0] != nDimensionArray[1]) // odd even
                total_dim = nDimensionArray[0] + nDimensionArray[1];
            else
                total_dim = nDimensionArray[0] * 2;
            printf("nDim0:%d nDim1:%d, total_dim:%d\n", nDimensionArray[0], nDimensionArray[1], total_dim);
            srand(FIRSTSEED);
            unsigned int *secondseeds = (unsigned int *) malloc(sizeof(unsigned int) * total_dim);
            for (i = 0; i < total_dim; i++)
            {
                secondseeds[i] = rand();
            }

            // in = feature is present, out = feature is not present
            double *importance_in = (double *) malloc(sizeof(double) * total_dim);
            double *importance_out = (double *) malloc(sizeof(double) * total_dim);

            //printf("sizeof importance in %d\n", sizeof(importance_in));
            memset(importance_in, 0, sizeof(double) * total_dim);
            memset(importance_out, 0, sizeof(double) * total_dim);


            int* permutation = (int *) malloc(sizeof(int) * total_dim);

            for(int j=0; j<total_dim;j++)
                permutation[j] = j;
            int *shuffle_result = (int *) malloc(sizeof(int) * total_dim);


            for (i = 0; i < numSVM; i++) {
                struct timeval outputStart;
                gettimeofday(&outputStart, 0);

                // printout prediction files
                if (functionality == 1 || functionality == 4) {
                    if (predictFilename == NULL) {
                        stringstream ss;
                        ss << i;
                        string filename = "./data/svm" + ss.str() + ".dat";
                        char oFilename[1024];
                        strncpy(oFilename, filename.c_str(), sizeof(oFilename));
                        printClassification(oFilename, result[i], testNPointsArray[i]);
                    } else {
                        printClassification(predictFilename[i], result[i], testNPointsArray[i]);
                    }
                }

                struct timeval outputEnd;
                gettimeofday(&outputEnd, 0);
                outputTime += (float) (outputEnd.tv_sec - outputStart.tv_sec) +
                              ((float) (outputEnd.tv_usec - outputStart.tv_usec)) * 1e-6;


                // print and record accuracy information
                if (testLabelsArray != NULL && !isMultiClass) {
                    float class1Label = 1.0f;
                    float class2Label = -1.0f;
                    int confusionMatrix[] = {0, 0, 0, 0};

                    for (int idx = 0; idx < testNPointsArray[i]; idx++) {
                        if ((testLabelsArray[i][idx] == class2Label) && (result[i][idx] < 0)) {
                            confusionMatrix[0]++;
                        } else if ((testLabelsArray[i][idx] == class2Label) && (result[i][idx] >= 0)) {
                            confusionMatrix[1]++;
                        } else if ((testLabelsArray[i][idx] == class1Label) && (result[i][idx] < 0)) {
                            confusionMatrix[2]++;
                        } else if ((testLabelsArray[i][idx] == class1Label) && (result[i][idx] >= 0)) {
                            confusionMatrix[3]++;
                        }
                    }

                    double accuracy =
                            (double) (confusionMatrix[0] + confusionMatrix[3]) * 100.0 / ((double) testNPointsArray[i]);

//                    printf("Accuracy: %f (%d / %d) \n", accuracy, confusionMatrix[0] + confusionMatrix[3],
//                           testNPointsArray[i]);
                    accuInfo[i] = accuracy;
//                    cout<<"Acc: "<< accuInfo[i]<<endl;
                }


            }

            // calculate importance of the features
            for (i=0;i<numSVM;i+=2)
            {
                memcpy(shuffle_result, permutation, sizeof(int) * total_dim);
                shuffle(shuffle_result, shuffle_result+total_dim, default_random_engine(secondseeds[(i/2) % total_dim]));

                // total_dim = 9, 0-4, 5-8
                // total_dim = 8, 0-3, 4-7
                for(int j=0;j<(total_dim+1)/2;j++) // bucket 1
                {
                   importance_in[shuffle_result[j]] += double(accuInfo[i]);
                   importance_out[shuffle_result[j]] += double(accuInfo[i+1]);
                }

                for(int j=(total_dim+1)/2;j<total_dim;j++) // bucket 2
                {
                    importance_in[shuffle_result[j]] += double(accuInfo[i+1]);
                    importance_out[shuffle_result[j]] += double(accuInfo[i]);
                }

            }

            vector<pair<int, float>>importance_rank;

            for(i=0;i<total_dim;i++)
            {
                //if(importance_in[i] < 1e-6 || importance_out[i] < 1e-6 )
                //printf("Problem on feature %d, in:%f, out:%f\n", i, importance_in[i], importance_out[i]);
                //cout<< "feature" << i << " in: "<< importance_in[i] << " out: "<< importance_out[i] << endl;
                importance_rank.push_back({i,importance_in[i]/importance_out[i]});
            }


            sort(importance_rank.begin(), importance_rank.end(),
                 [](pair<int,float>&a, pair<int,float>&b) -> bool {return b.second < a.second;} );

            printf("Importance (high to low) :\n");
            for(auto feature: importance_rank)
            {
                printf("Feature %d = %lf\n",feature.first, feature.second);
            }


            // end importance

            if (isMultiClass && functionality == 2)
                mcvResult = result;

            if (isMultiClass && functionality == 1)
                mPredictResult = result;

            struct timeval predictEnd;
            gettimeofday(&predictEnd, 0);
            predictTime = (float) (predictEnd.tv_sec - predictStart.tv_sec) +
                          ((float) (predictEnd.tv_usec - predictStart.tv_usec)) * 1e-6 - outputTime;

            free(secondseeds);
        }

            break;
        default:
            break;
    }

//    No need of this cv accuracy
//    if (functionality == 2 && !isMultiClass) {
//        for (i = 0; i < numSVM / folder; i++) {
//            float average = 0.0f;
//            float count = 0;
//            for (int j = 0; j < folder; j++) {
//                average += accuInfo[i * folder + j] * testNPointsArray[i * folder + j];
//                count += testNPointsArray[i * folder + j];
//            }
//
//            average = average / count;
//            cout << "SVM " << i << " cross-validation accuracy: " << average << "%" << endl;
//
//            if (accuracyResult != NULL)
//                accuracyResult[i] = average;
//        }
//    }

    for (i = 0; i < ngpus; i++) {
        if (functionality == 2) {
            cout << "GPU " << i << " has processed " << processDataInfo[i] << " SVMs." << endl;
        }
        cout << "GPU " << i << " has trained " << numSVMPerGpu[i] << " SVMs, " << "train time: " << trainTimePerGpu[i]
             << endl;
    }

    switch (functionality) {
        case 0:
            cout << "Train time: " << trainTime << endl;
            cout << "Output time: " << outputTime << endl;
            break;
        case 1:
            cout << "Train time: " << trainTime << endl;
            cout << "Predict time: " << predictTime << endl;
            cout << "Output time: " << outputTime << endl;
            break;
        case 2:
            cout << "Cross validation data processing time: " << cvDataProcessTime << endl;
            cout << "Train time: " << trainTime << endl;
            cout << "Predict time: " << predictTime << endl;
            cout << "Output time: " << outputTime << endl;
            break;
        case 3:
            cout << "SubSet data processing time: " << subsetDataProcessTime << endl;
            cout << "Train time: " << trainTime << endl;
            cout << "Output time: " << outputTime << endl;
            break;
        case 4:
            cout << "SubSet data processing time: " << subsetDataProcessTime << endl;
            cout << "Train time: " << trainTime << endl;
            cout << "Predict time: " << predictTime << endl;
            cout << "Output time: " << outputTime << endl;
            break;
        default:
            break;
    }

    // free variables
    switch (functionality) {
        case 1:
        case 2:
        case 4:
            free(accuInfo);
            break;
        default:
            break;
    }
    switch (functionality) {
        case 2:
            for (i = 0; i < numSVM; i+=2) {
                //cout<<"freed"<<i<<endl;
                free(dataArray[i]);
                free(labelsArray[i]);
                free(kpArray[i]);
                free(transposedDataArray[i]);
                free(testDataArray[i]);
                free(testLabelsArray[i]);

            }

            free(dataArray);
            free(nPointsArray);
            free(nDimensionArray);
            free(labelsArray);
            free(kpArray);
            free(costArray);
            free(heuristicMethodArray);
            free(epsilonArray);
            free(toleranceArray);
            free(transposedDataArray);

            free(testDataArray);
            free(testLabelsArray);
            free(testNPointsArray);
            break;
        case 3:
        case 4:
            for (i = 0; i < numSVM; i++) {
                free(dataArray[i]);
                free(labelsArray[i]);
                free(transposedDataArray[i]);
            }

            free(dataArray);
            free(labelsArray);
            free(transposedDataArray);
            break;
        default:
            break;
    }
}


// read from an integer file
vector<int> readSubset(const char *file_name) {
    FILE *file = fopen(file_name, "r");
    int i = 0;
    vector<int> index;

    fscanf(file, "%d", &i);
    while (!feof(file)) {
        index.push_back(i);
        fscanf(file, "%d", &i);
    }
    fclose(file);
    return index;
}

// training multiple SVMs from input files with cache-ratio "ratio"
void
performMultiGPUTrainingFromFiles(int functionality, int numSVM, char **trainingFile, int folder, char **testingFile,
                                 char *dataFile, float ratio, int kernelType, SelectionHeuristic heuristicMethod,
                                 float cost, float tolerance, float epsilon) {
    // record time
    struct timeval start;
    gettimeofday(&start, 0);
    struct timeval readStart;
    gettimeofday(&readStart, 0);

    int i;
    float **dataArray;
    int *nPointsArray;
    int *nDimensionArray;
    float **labelsArray;
    Kernel_params **kpArray;
    float *costArray;
    SelectionHeuristic *heuristicMethodArray;
    float *epsilonArray;
    float *toleranceArray;
    float **transposedDataArray;
    char **modelFilename;

    // variables for prediction
    float **testDataArray;
    int *testNPointsArray;
    int *testNDimensionArray;
    float **testLabelsArray;
    char **predictFilename;

    // variable for subset index
    int **subsetIdx;

    // functionality specifies the purpose of using this function: 0-training, 1-prediction,
    // 2-cross validation, 3-subset training, 4-subset prediction
    // allocate memory for variables
    switch (functionality) {
        case 0:
        case 1:
        case 2:
            dataArray = (float **) malloc(sizeof(float *) * numSVM);  //sizeof(float pointer)
            labelsArray = (float **) malloc(sizeof(float *) * numSVM);
            transposedDataArray = (float **) malloc(sizeof(float *) * numSVM);
            break;
        case 3:
        case 4:
            dataArray = (float **) malloc(sizeof(float *) * 1);
            labelsArray = (float **) malloc(sizeof(float *) * 1);
            transposedDataArray = (float **) malloc(sizeof(float *) * 1);
            subsetIdx = (int **) malloc(sizeof(int *) * numSVM);
            break;
        default:
            break;
    }

    nPointsArray = (int *) malloc(sizeof(int) * numSVM);
    nDimensionArray = (int *) malloc(sizeof(int) * numSVM);
    kpArray = (Kernel_params **) malloc(sizeof(Kernel_params *) * numSVM);
    costArray = (float *) malloc(sizeof(float) * numSVM);
    heuristicMethodArray = (SelectionHeuristic *) malloc(sizeof(SelectionHeuristic) * numSVM);
    epsilonArray = (float *) malloc(sizeof(float) * numSVM);
    toleranceArray = (float *) malloc(sizeof(float) * numSVM);

    switch (functionality) {
        case 0:
        case 3:
            modelFilename = (char **) malloc(sizeof(char *) * numSVM);
            break;
        case 1:
        case 4:
            testDataArray = (float **) malloc(sizeof(float *) * numSVM);
            testNPointsArray = (int *) malloc(sizeof(int) * numSVM);
            testNDimensionArray = (int *) malloc(sizeof(int) * numSVM);
            testLabelsArray = (float **) malloc(sizeof(float *) * numSVM);
            predictFilename = (char **) malloc(sizeof(char *) * numSVM);
            break;
        default:
            break;
    }

    int nPoints;
    int nDimension;
    if (functionality == 3 || functionality == 4) {
        readSvm(dataFile, &(dataArray[0]), &(labelsArray[0]), &nPoints, &nDimension, &(transposedDataArray[0]));
        for (i = 0; i < numSVM; i++) {
            vector<int> tmp = readSubset(trainingFile[i]);
            nPointsArray[i] = tmp.size();
            nDimensionArray[i] = nDimension;
            subsetIdx[i] = (int *) malloc(sizeof(int) * tmp.size());
            int j;
            for (j = 0; j < tmp.size(); j++) {
                subsetIdx[i][j] = tmp[j];
            }
        }
    }

    // read in data from input files
    Kernel_params kp;
    for (i = 0; i < numSVM; i++) {
        // read in testing file data
        if (functionality == 1 || functionality == 4)
            readSvm(testingFile[i], &(testDataArray[i]), &(testLabelsArray[i]), &(testNPointsArray[i]),
                    &(testNDimensionArray[i])); //TODO: @param transposedataarray dont have default value

        // read in training file data
        switch (functionality) {
            case 0:
            case 1:
            case 2:
                readSvm(trainingFile[i], &(dataArray[i]), &(labelsArray[i]), &(nPointsArray[i]), &(nDimensionArray[i]),
                        &(transposedDataArray[i]));
                //filename, start addr of the ith data / label / # of sample / # of dimension / transposed data array
                break;
            default:
                break;
        }

        //printf("File %d, input data found: %d points, %d dimension\n", i, nPointsArray[i], nDimensionArray[i]);

        // these hyper params are the same across all SVMs
        costArray[i] = cost;
        toleranceArray[i] = tolerance;
        epsilonArray[i] = epsilon;
        heuristicMethodArray[i] = heuristicMethod;


        // set Kernel_params
        float parameterA = -0.125f;
        float parameterB = 1.0f;
        float parameterC = 3.0f;
        kpArray[i] = (Kernel_params *) malloc(sizeof(Kernel_params));
        if (kernelType == LINEAR) {
            printf("Linear kernel\n");
            kp.kernel_type = "linear";
        } else if (kernelType == POLYNOMIAL) {
            parameterC = 3.0f;
            parameterA = 1.0 / nPointsArray[i];
            parameterB = 0.0f;
            //printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
            if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) {
                printf("Invalid parameters\n");
                exit(1);
            }
            kp.kernel_type = "polynomial";
            kp.gamma = parameterA;
            kp.coef0 = parameterB;
            kp.degree = (int) parameterC;
        } else if (kernelType == GAUSSIAN) {
            parameterA = 1.0 / nDimensionArray[i];
            //printf("Gaussian kernel: gamma = %f\n", parameterA);
            if (parameterA < 0) {
                printf("Invalid parameters\n");
                exit(1);
            }
            kp.kernel_type = "rbf";
            kp.gamma = parameterA;
        } else if (kernelType == SIGMOID) {
            parameterA = 1.0 / nPointsArray[i];
            parameterB = 0.0f;
            //printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
            if ((parameterA <= 0) || (parameterB < 0)) {
                printf("Invalid Parameters\n");
                exit(1);
            }
            kp.kernel_type = "sigmoid";
            kp.gamma = parameterA;
            kp.coef0 = parameterB;
        }

        memcpy(kpArray[i], &kp, sizeof(Kernel_params));

        switch (functionality) {
            case 0:
            case 3: {
                // create model file names, output a model
                int trainingFileLength = strlen(trainingFile[i]);
                modelFilename[i] = (char *) malloc(sizeof(char) * (trainingFileLength + 5));
                strncpy(modelFilename[i], trainingFile[i], trainingFileLength + 4);
                char *period = strrchr(modelFilename[i], '.');
                if (period == NULL) {
                    period = modelFilename[i] + trainingFileLength;
                }
                strncpy(period, ".mdl\0", 5);
            }
                break;
            case 1:
            case 4: {
                //create predict file names, direct output the prediction result instead of a model
                int testingFileLength = strlen(testingFile[i]);
                predictFilename[i] = (char *) malloc(sizeof(char) * (testingFileLength + 5));
                strncpy(predictFilename[i], testingFile[i], testingFileLength + 4);
                char *period = strrchr(predictFilename[i], '.');
                if (period == NULL) {
                    period = predictFilename[i] + testingFileLength;
                }
                strncpy(period, ".dat\0", 5);
            }
                break;
            default:
                break;
        }

    }

    struct timeval readEnd;
    gettimeofday(&readEnd, 0);
    float readTime =
            (float) (readEnd.tv_sec - readStart.tv_sec) + ((float) (readEnd.tv_usec - readStart.tv_usec)) * 1e-6;

    switch (functionality) {
        case 0:
            svmTrain(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                     heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, modelFilename, ratio);
            break;
        case 1:
            svmPredict(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                       heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, testDataArray,
                       testNPointsArray, testLabelsArray, predictFilename, ratio);
            break;
        case 2:
            // dataArry can be NULL
            svmCrossValidation(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                               heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, ratio);
            break;
        case 3:
            // dataArray can be NULL
            svmSubsetTrain(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray,
                           epsilonArray, toleranceArray, transposedDataArray, subsetIdx, modelFilename, ratio);
            break;
        case 4:
            // dataArray can be NULL
            svmSubsetPredict(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray,
                             heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, subsetIdx,
                             testDataArray, testNPointsArray, testLabelsArray, predictFilename, ratio);
            break;
        default:
            break;
    }

    // output running time information
    printf("Reading data time = %f seconds\n", readTime);

    struct timeval finish;
    gettimeofday(&finish, 0);
    float totalTime = (float) (finish.tv_sec - start.tv_sec) + ((float) (finish.tv_usec - start.tv_usec)) * 1e-6;
    printf("Total time = %f seconds\n", totalTime);

    cout << "Multi-GPU Training From Files End." << endl;

    // free variable
    for (i = 0; i < numSVM; i++) {
        switch (functionality) {
            case 0:
            case 1:
            case 2:
                free(dataArray[i]);
                free(labelsArray[i]);
                free(transposedDataArray[i]);
                break;
            case 3:
            case 4:
                if (i == 0) {
                    free(dataArray[i]);
                    free(labelsArray[i]);
                    free(transposedDataArray[i]);
                    free(subsetIdx[i]);
                }
                break;
            default:
                break;
        }

        free(kpArray[i]);

        switch (functionality) {
            case 0:
            case 3:
                free(modelFilename[i]);
                break;
            case 1:
            case 4:
                free(testDataArray[i]);
                free(testLabelsArray[i]);
                free(predictFilename[i]);
                break;
            default:
                break;
        }
    }

    switch (functionality) {
        case 3:
        case 4:
            free(subsetIdx);
            break;
        default:
            break;
    }

    free(dataArray);
    free(nPointsArray);
    free(nDimensionArray);
    free(labelsArray);
    free(kpArray);
    free(costArray);
    free(heuristicMethodArray);
    free(epsilonArray);
    free(toleranceArray);
    free(transposedDataArray);

    switch (functionality) {
        case 0:
        case 3:
            free(modelFilename);
            break;
        case 1:
        case 4:
            free(testDataArray);
            free(testLabelsArray);
            free(testNPointsArray);
            free(testNDimensionArray);
            free(predictFilename);
            break;
        default:
            break;
    }
}

void formModel(float *trainingPoints, int nTrainingPoints, int nDimension, float *trainingAlpha, float *trainingLabels,
               float **supportVectors, int *nSV, float **alpha, float epsilon) {
    int count = 0;

    for (int i = 0; i < nTrainingPoints; i++) {
        if (trainingAlpha[i] > epsilon) {
            count++;
        }
    }
    *nSV = count;
    printf("%i support vectors found\n", count);
    float *mySupportVectors = *supportVectors;
    mySupportVectors = (float *) malloc(count * nDimension * sizeof(float));
    *supportVectors = mySupportVectors;

    float *myAlpha = *alpha;
    myAlpha = (float *) malloc(count * sizeof(float));
    *alpha = myAlpha;
    int currentOutput = 0;
    for (int i = 0; i < nTrainingPoints; i++) {
        if (trainingAlpha[i] > epsilon) {
            float *sourcePointer = &trainingPoints[i];
            float *destinationPointer = &mySupportVectors[currentOutput];
            for (int j = 0; j < nDimension; j++) {
                *destinationPointer = *sourcePointer;
                sourcePointer += nTrainingPoints;
                destinationPointer += count;
            }
            myAlpha[currentOutput] = trainingAlpha[i] * trainingLabels[i];
            currentOutput++;
        }
    }
}

// assume the kernel types of all svms are the same
void
performMultiTraining(const int numSVM, float **dataArray, int *nPointsArray, int *nDimensionArray, float **labelsArray,
                     float ***p_alphaArray, Kernel_params **kpArray, float *costArray,
                     SelectionHeuristic *heuristicMethodArray, float *epsilonArray, float *toleranceArray,
                     float **transposedDataArray, int gpuId) {
    if (gpuId == -1)
        chooseLargestGPU(true);

    int i;
    float *cEpsilonArray;
    cEpsilonArray = (float *) malloc(numSVM * sizeof(float));

    Controller progress(2.0, heuristicMethodArray[0], 64, nPointsArray[0]);

    int *kernelTypeArray;
    kernelTypeArray = (int *) malloc(numSVM * sizeof(int));
    float *parameterAArray = (float *) malloc(numSVM * sizeof(float));
    float *parameterBArray = (float *) malloc(numSVM * sizeof(float));
    float *parameterCArray = (float *) malloc(numSVM * sizeof(float));

    float **h_devDataArray = (float **) malloc(sizeof(float *) * numSVM);
    float **h_devTransposedDataArray = (float **) malloc(sizeof(float *) * numSVM);
    size_t *h_devDataPitchArray = (size_t *) malloc(sizeof(size_t) * numSVM);
    size_t *h_devTransposedDataPitchArray = (size_t *) malloc(sizeof(size_t) * numSVM);
    int *hostPitchInFloatsArray = (int *) malloc(sizeof(int) * numSVM);
    float **hostDataArray = (float **) malloc(sizeof(float *) * numSVM);
    bool *hostDataAllocedArray = (bool *) malloc(sizeof(bool) * numSVM);

    bool *transposedDataAllocedArray = (bool *) malloc(sizeof(bool) * numSVM);
    float **alphaArray = (float **) malloc(sizeof(float *) * numSVM);
    float **h_devLabelsArray = (float **) malloc(sizeof(float *) * numSVM);
    float **h_devKernelDiagArray = (float **) malloc(sizeof(float *) * numSVM);
    float **h_devAlphaArray = (float **) malloc(sizeof(float *) * numSVM);
    float **h_devFArray = (float **) malloc(sizeof(float *) * numSVM);
    void **h_devResultArray = (void **) malloc(sizeof(void *) * numSVM);
    //float **hostResultArray = (float **) malloc(sizeof(float *) * numSVM);
    float **hostResultArray;
    checkCudaErrors(cudaMallocHost((void **) (&hostResultArray), sizeof(float *) * numSVM));


    int *blockWidthArray = (int *) malloc(sizeof(int) * numSVM);
    float **h_devLocalFsRLArray = (float **) malloc(sizeof(float *) * numSVM);
    float **h_devLocalFsRHArray = (float **) malloc(sizeof(float *) * numSVM);
    int **h_devLocalIndicesRLArray = (int **) malloc(sizeof(int *) * numSVM);
    int **h_devLocalIndicesRHArray = (int **) malloc(sizeof(int *) * numSVM);
    float **h_devLocalObjsMaxObjArray = (float **) malloc(sizeof(float *) * numSVM);
    int **h_devLocalIndicesMaxObjArray = (int **) malloc(sizeof(int *) * numSVM);

    void **tempArray = (void **) malloc(sizeof(void *) * numSVM);
    size_t *rowPitchArray = (size_t *) malloc(sizeof(size_t) * numSVM);

    for (i = 0; i < numSVM; i++) {
        cEpsilonArray[i] = costArray[i] - epsilonArray[i];
        kernelTypeArray[i] = GAUSSIAN;
        if (kpArray[i]->kernel_type.compare(0, 3, "rbf") == 0) {
            parameterAArray[i] = -kpArray[i]->gamma;
            kernelTypeArray[i] = GAUSSIAN;
            //printf("Gaussian kernel: gamma = %f\n", -parameterAArray[i]);
        } else if (kpArray[i]->kernel_type.compare(0, 10, "polynomial") == 0) {
            parameterAArray[i] = kpArray[i]->gamma;
            parameterBArray[i] = kpArray[i]->coef0;
            parameterCArray[i] = kpArray[i]->degree;
            kernelTypeArray[i] = POLYNOMIAL;
            printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterAArray[i], parameterBArray[i],
                   parameterCArray[i]);
        } else if (kpArray[i]->kernel_type.compare(0, 6, "linear") == 0) {
            kernelTypeArray[i] = LINEAR;
            printf("Linear kernel\n");
        } else if (kpArray[i]->kernel_type.compare(0, 7, "sigmoid") == 0) {
            kernelTypeArray[i] = SIGMOID;
            parameterAArray[i] = kpArray[i]->gamma;
            parameterBArray[i] = kpArray[i]->coef0;
            printf("Sigmoid kernel: a = %f, r = %f\n", parameterAArray[i], parameterBArray[i]);
            if ((parameterAArray[i] <= 0) || (parameterBArray[i] < 0)) {
                printf("Invalid Parameters\n");
                exit(1);
            }
        }
        //printf("Cost: %f, Tolerance: %f, Epsilon: %f\n", costArray[i], toleranceArray[i], epsilonArray[i]);
        hostPitchInFloatsArray[i] = nPointsArray[i];
        hostDataAllocedArray[i] = false;
        checkCudaErrors(cudaMallocPitch((void **) &(h_devDataArray[i]), &(h_devDataPitchArray[i]),
                                        nPointsArray[i] * sizeof(float), nDimensionArray[i]));

        //allocate pitched/padded memory space for 2D array
        //width = num of samples in SVM_i - for features of all samples
        //height = num of features/dims in SVM_i - for different features


        //align training data and copy to device
        if (h_devDataPitchArray[i] == nPointsArray[i] * sizeof(float)) {
            printf("Data %d is already aligned\n", i);
            hostDataArray[i] = dataArray[i];
        } else {
            hostPitchInFloatsArray[i] = h_devDataPitchArray[i] / sizeof(float);
            hostDataArray[i] = (float *) malloc(h_devDataPitchArray[i] * nDimensionArray[i]); //same space as the device data array
            hostDataAllocedArray[i] = true;
            //printf("Realigning data %d to a pitch of %i floats\n", i, hostPitchInFloatsArray[i]);
            for (int p = nDimensionArray[i] - 1; p >= 0; p--) {
                for (int q = nPointsArray[i] - 1; q >= 0; q--) {
                    hostDataArray[i][p * hostPitchInFloatsArray[i] + q] = dataArray[i][p * nPointsArray[i] + q];
                    //dataArray is the original linear data array
                    //hostDataArray is padded/pitched to the same shape of the device h_devDataPitchArray
                }
            }
        }
        checkCudaErrors(cudaMemcpy(h_devDataArray[i], hostDataArray[i], h_devDataPitchArray[i] * nDimensionArray[i],
                                   cudaMemcpyHostToDevice));

        //make transposed data if not passed in
        transposedDataAllocedArray[i] = false;
        if (transposedDataArray[i] == 0) {
            transposedDataArray[i] = (float *) malloc(sizeof(float) * nPointsArray[i] * nDimensionArray[i]);
            transposedDataAllocedArray[i] = true;
            for (int p = 0; p < nPointsArray[i]; p++) {
                for (int q = 0; q < nDimensionArray[i]; q++) {
                    transposedDataArray[i][p * nDimensionArray[i] + q] = hostDataArray[i][
                            q * hostPitchInFloatsArray[i] + p];
                }
            }
        }


        alphaArray[i] = (float *) malloc(sizeof(float) * nPointsArray[i]);
        (*p_alphaArray)[i] = alphaArray[i];


        checkCudaErrors(cudaMallocPitch((void **) &(h_devTransposedDataArray[i]), &(h_devTransposedDataPitchArray[i]),
                                        nDimensionArray[i] * sizeof(float), nPointsArray[i]));
        //width = feature 1-N for sample 1
        //height = num of data points

        checkCudaErrors(
                cudaMemcpy2D(h_devTransposedDataArray[i], h_devTransposedDataPitchArray[i], transposedDataArray[i],
                             nDimensionArray[i] * sizeof(float), nDimensionArray[i] * sizeof(float),
                             nPointsArray[i], cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc((void **) &(h_devLabelsArray[i]), nPointsArray[i] * sizeof(float)));
        checkCudaErrors(cudaMemcpy(h_devLabelsArray[i], labelsArray[i], nPointsArray[i] * sizeof(float),
                                   cudaMemcpyHostToDevice));


        checkCudaErrors(cudaMalloc((void **) &(h_devKernelDiagArray[i]), nPointsArray[i] * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &(h_devAlphaArray[i]), nPointsArray[i] * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &(h_devFArray[i]), nPointsArray[i] * sizeof(float)));
        checkCudaErrors(cudaMalloc(&(h_devResultArray[i]), 8 * sizeof(float)));

        //hostResultArray[i] = (float *) malloc(8 * sizeof(float));
        //checkCudaErrors(cudaMallocHost((void **) &(hostResultArray[i]), 8 * sizeof(float)));

        blockWidthArray[i] = intDivideRoundUp(nPointsArray[i], BLOCKSIZE);

        checkCudaErrors(cudaMalloc((void **) &(h_devLocalFsRLArray[i]), blockWidthArray[i] * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &(h_devLocalFsRHArray[i]), blockWidthArray[i] * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &(h_devLocalIndicesRLArray[i]), blockWidthArray[i] * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &(h_devLocalIndicesRHArray[i]), blockWidthArray[i] * sizeof(int)));

        checkCudaErrors(cudaMalloc((void **) &(h_devLocalObjsMaxObjArray[i]), blockWidthArray[i] * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **) &(h_devLocalIndicesMaxObjArray[i]), blockWidthArray[i] * sizeof(int)));


        //get the row pitch by allocating a [num of points * 2] matrix and then free it
        checkCudaErrors(cudaMallocPitch(&(tempArray[i]), &(rowPitchArray[i]), nPointsArray[i] * sizeof(float), 2));
        checkCudaErrors(cudaFree(tempArray[i]));
    }

    //allocate contiguous memory for the host resultarray
    //8 float per line, 8 * 4 = 32 byte
    //group 8KB = 8 * 1024 / 32 = 256 arrays together
    const int groups = intDivideRoundUp(numSVM, GROUPSIZE);
    for (i = 0; i < groups; i++) {
        checkCudaErrors(cudaMallocHost((void **) &(hostResultArray[i * GROUPSIZE]), GROUPSIZE * 8 * sizeof(float)));
        for (int j = 0; j < GROUPSIZE; j++) {
            if (i * groups + j >= numSVM)
                break;
            hostResultArray[i * GROUPSIZE + j] = &(hostResultArray[i * GROUPSIZE][j * 8]);
        }
    }





    size_t remainingMemory;
    size_t totalMemory;
    cuMemGetInfo(&remainingMemory, &totalMemory);

    size_t *sizeOfCacheArray = (size_t *) malloc(sizeof(size_t) * numSVM);
    int totalSize = 0;
    for (i = 0; i < numSVM; i++) {
        totalSize += nPointsArray[i];
    }
    for (i = 0; i < numSVM; i++) {
        sizeOfCacheArray[i] = (int) ((((1.0f * remainingMemory * nPointsArray[i]) / totalSize) / rowPitchArray[i]) *
                                     0.95);
        if (nPointsArray[i] < sizeOfCacheArray[i]) {
            sizeOfCacheArray[i] = nPointsArray[i];
        }
    }

#ifdef __DEVICE_EMULATION__
                                                                                                                            for(int p = 0; p < numSVM; p++) {
    sizeOfCacheArray[p] = nPointsArray[p];
  }
#endif

    printf("%Zu bytes of memory found on device, %Zu bytes currently free\n", totalMemory, remainingMemory);
//    for (i = 0; i < numSVM; i++) {
//        printf("for svm %d: %Zu rows of kernel matrix will be cached (%Zu bytes per row)\n", i, sizeOfCacheArray[i],
//               rowPitchArray[i]);
//    }

    float **h_devCacheArray = (float **) malloc(sizeof(float *) * numSVM);
    size_t *cachePitchArray = (size_t *) malloc(sizeof(size_t) * numSVM);
    Cache **kernelCacheArray = (Cache **) malloc(sizeof(Cache *) * numSVM);
    int *devCachePitchInFloatsArray = (int *) malloc(sizeof(int) * numSVM);

    cudaError_t err;
    for (i = 0; i < numSVM; i++) {
        checkCudaErrors(
                cudaMallocPitch((void **) &(h_devCacheArray[i]), &(cachePitchArray[i]), nPointsArray[i] * sizeof(float),
                                sizeOfCacheArray[i]));
        kernelCacheArray[i] = new Cache(nPointsArray[i], sizeOfCacheArray[i]);
        devCachePitchInFloatsArray[i] = (int) cachePitchArray[i] / (sizeof(float));
        err = cudaGetLastError();
        if (err) printf("Svm %d error: %s\n", i, cudaGetErrorString(err));
        //printf("Svm %d: allocated arrays on GPU\n", i);
    }


    int *devDataPitchInFloatsArray = (int *) malloc(sizeof(int) * numSVM);
    int *devTransposedDataPitchInFloatsArray = (int *) malloc(sizeof(int) * numSVM);

    for (i = 0; i < numSVM; i++) {
        devDataPitchInFloatsArray[i] = ((int) h_devDataPitchArray[i]) >> 2;
        devTransposedDataPitchInFloatsArray[i] = ((int) h_devTransposedDataPitchArray[i]) >> 2;
    }


    int kType = kernelTypeArray[0];

    // find the max num of blocks needed
    int maxBlockWidth = 0;
    for (i = 0; i < numSVM; i++) {
        if (blockWidthArray[i] > maxBlockWidth) {
            maxBlockWidth = blockWidthArray[i];
        }
    }


    dim3 threadsLinear = dim3(BLOCKSIZE);
    dim3 blocksLinear = dim3(maxBlockWidth, numSVM);

    int *devnPointsArray;
    int *devnDimensionArray;
    float *devParameterAArray;
    float *devParameterBArray;
    float *devParameterCArray;

    checkCudaErrors(cudaMalloc((void **) (&devnPointsArray), numSVM * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) (&devnDimensionArray), numSVM * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) (&devParameterAArray), numSVM * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) (&devParameterBArray), numSVM * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **) (&devParameterCArray), numSVM * sizeof(float)));

    checkCudaErrors(
            cudaMemcpy((void *) devnPointsArray, (void *) nPointsArray, numSVM * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *) devnDimensionArray, (void *) nDimensionArray, numSVM * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *) devParameterAArray, (void *) parameterAArray, numSVM * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *) devParameterBArray, (void *) parameterBArray, numSVM * sizeof(float),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((void *) devParameterCArray, (void *) parameterCArray, numSVM * sizeof(float),
                               cudaMemcpyHostToDevice));

    //copy the 'pointers' to device array from host to device
    //so the can access the arrays from both device and host

    void *d_devDataArray = 0;
    checkCudaErrors(cudaMalloc(&d_devDataArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devDataArray, h_devDataArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devDataArray = (float **) d_devDataArray;

    void *d_devTransposedDataArray = 0;
    checkCudaErrors(cudaMalloc(&d_devTransposedDataArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devTransposedDataArray, h_devTransposedDataArray, numSVM * sizeof(float *),
                               cudaMemcpyHostToDevice));
    float **devTransposedDataArray = (float **) d_devTransposedDataArray;

    void *d_devDataPitchInFloatsArray = 0;
    checkCudaErrors(cudaMalloc(&d_devDataPitchInFloatsArray, numSVM * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_devDataPitchInFloatsArray, devDataPitchInFloatsArray, numSVM * sizeof(int),
                               cudaMemcpyHostToDevice));
    free(devDataPitchInFloatsArray);
    devDataPitchInFloatsArray = (int *) d_devDataPitchInFloatsArray;

    void *d_devTransposedDataPitchInFloatsArray = 0;
    checkCudaErrors(cudaMalloc(&d_devTransposedDataPitchInFloatsArray, numSVM * sizeof(int)));
    checkCudaErrors(
            cudaMemcpy(d_devTransposedDataPitchInFloatsArray, devTransposedDataPitchInFloatsArray, numSVM * sizeof(int),
                       cudaMemcpyHostToDevice));
    free(devTransposedDataPitchInFloatsArray);
    devTransposedDataPitchInFloatsArray = (int *) d_devTransposedDataPitchInFloatsArray;

    void *d_devKernelDiagArray = 0;
    checkCudaErrors(cudaMalloc(&d_devKernelDiagArray, numSVM * sizeof(float *)));
    checkCudaErrors(
            cudaMemcpy(d_devKernelDiagArray, h_devKernelDiagArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devKernelDiagArray = (float **) d_devKernelDiagArray;

    void *d_devFArray = 0;
    checkCudaErrors(cudaMalloc(&d_devFArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devFArray, h_devFArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devFArray = (float **) d_devFArray;

    void *d_devLabelsArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLabelsArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devLabelsArray, h_devLabelsArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devLabelsArray = (float **) d_devLabelsArray;

    void *d_devAlphaArray = 0;
    checkCudaErrors(cudaMalloc(&d_devAlphaArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devAlphaArray, h_devAlphaArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devAlphaArray = (float **) d_devAlphaArray;

    void *d_devResultArray = 0;
    checkCudaErrors(cudaMalloc(&d_devResultArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devResultArray, h_devResultArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    void **devResultArray = (void **) d_devResultArray;

    void *d_devCacheArray = 0;
    checkCudaErrors(cudaMalloc(&d_devCacheArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devCacheArray, h_devCacheArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devCacheArray = (float **) d_devCacheArray;

    void *d_devCachePitchInFloatsArray = 0;
    checkCudaErrors(cudaMalloc(&d_devCachePitchInFloatsArray, numSVM * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_devCachePitchInFloatsArray, devCachePitchInFloatsArray, numSVM * sizeof(int),
                               cudaMemcpyHostToDevice));
    free(devCachePitchInFloatsArray);
    devCachePitchInFloatsArray = (int *) d_devCachePitchInFloatsArray;

    void *d_devLocalIndicesRLArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLocalIndicesRLArray, numSVM * sizeof(int *)));
    checkCudaErrors(cudaMemcpy(d_devLocalIndicesRLArray, h_devLocalIndicesRLArray, numSVM * sizeof(int *),
                               cudaMemcpyHostToDevice));
    int **devLocalIndicesRLArray = (int **) d_devLocalIndicesRLArray;

    void *d_devLocalIndicesRHArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLocalIndicesRHArray, numSVM * sizeof(int *)));
    checkCudaErrors(cudaMemcpy(d_devLocalIndicesRHArray, h_devLocalIndicesRHArray, numSVM * sizeof(int *),
                               cudaMemcpyHostToDevice));
    int **devLocalIndicesRHArray = (int **) d_devLocalIndicesRHArray;

    void *d_devLocalFsRHArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLocalFsRHArray, numSVM * sizeof(float *)));
    checkCudaErrors(
            cudaMemcpy(d_devLocalFsRHArray, h_devLocalFsRHArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devLocalFsRHArray = (float **) d_devLocalFsRHArray;

    void *d_devLocalFsRLArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLocalFsRLArray, numSVM * sizeof(float *)));
    checkCudaErrors(
            cudaMemcpy(d_devLocalFsRLArray, h_devLocalFsRLArray, numSVM * sizeof(float *), cudaMemcpyHostToDevice));
    float **devLocalFsRLArray = (float **) d_devLocalFsRLArray;

    void *d_devLocalObjsMaxObjArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLocalObjsMaxObjArray, numSVM * sizeof(float *)));
    checkCudaErrors(cudaMemcpy(d_devLocalObjsMaxObjArray, h_devLocalObjsMaxObjArray, numSVM * sizeof(float *),
                               cudaMemcpyHostToDevice));
    float **devLocalObjsMaxObjArray = (float **) d_devLocalObjsMaxObjArray;

    void *d_devLocalIndicesMaxObjArray = 0;
    checkCudaErrors(cudaMalloc(&d_devLocalIndicesMaxObjArray, numSVM * sizeof(int *)));
    checkCudaErrors(cudaMemcpy(d_devLocalIndicesMaxObjArray, h_devLocalIndicesMaxObjArray, numSVM * sizeof(int *),
                               cudaMemcpyHostToDevice));
    int **devLocalIndicesMaxObjArray = (int **) d_devLocalIndicesMaxObjArray;

    launchMultiInitialization(devDataArray, devDataPitchInFloatsArray, devnPointsArray, devnDimensionArray, kType,
                              devParameterAArray, devParameterBArray, devParameterCArray, devKernelDiagArray,
                              devAlphaArray, devFArray, devLabelsArray, blocksLinear, threadsLinear);
    err = cudaGetLastError();
    if (err) printf("Error: %s\n", cudaGetErrorString(err));
    printf("Initialization complete\n");

    //Choose initial points
    float *bLowArray = (float *) malloc(sizeof(float) * numSVM);
    float *bHighArray = (float *) malloc(sizeof(float) * numSVM);
    int *iterationArray = (int *) malloc(sizeof(int) * numSVM);
    int *iLowArray = (int *) malloc(sizeof(int) * numSVM);
    int *iHighArray = (int *) malloc(sizeof(int) * numSVM);

    for (int idx = 0; idx < numSVM; idx++) {
        bLowArray[idx] = 1;
        bHighArray[idx] = -1;
        iterationArray[idx] = 0;
        iLowArray[idx] = -1;
        iHighArray[idx] = -1;

        //nPointsArray = num of training points in an svm
        //labelsArray = label of all points

        for (i = 0; i < nPointsArray[idx]; i++) {
            if (labelsArray[idx][i] < 0) {
                if (iLowArray[idx] == -1) {
                    iLowArray[idx] = i;
                    // iLow = argmax(I0,I3,I4) f_i
                    // because in the initialization a=0 and f_i=-y_i, all the b_i is -y_i, i.e., -1 in the two class case
                    // any negative point is OK
                    if (iHighArray[idx] > -1) {
                        i = nPointsArray[idx]; //Terminate
                    }
                }
            } else {
                if (iHighArray[idx] == -1) {
                    iHighArray[idx] = i;
                    if (iLowArray[idx] > -1) {
                        i = nPointsArray[idx]; //Terminate
                    }
                }
            }
        }
    }

    dim3 singletonThreads(1);
    dim3 singletonBlocks(numSVM);

    float *devCostArray;
    checkCudaErrors(cudaMalloc((void **) (&devCostArray), numSVM * sizeof(float)));
    checkCudaErrors(
            cudaMemcpy((void *) devCostArray, (void *) costArray, numSVM * sizeof(float), cudaMemcpyHostToDevice));

    float *devEpsilonArray;
    checkCudaErrors(cudaMalloc((void **) (&devEpsilonArray), numSVM * sizeof(float)));
    checkCudaErrors(cudaMemcpy((void *) devEpsilonArray, (void *) epsilonArray, numSVM * sizeof(float),
                               cudaMemcpyHostToDevice));

    float *devcEpsilonArray;
    checkCudaErrors(cudaMalloc((void **) (&devcEpsilonArray), numSVM * sizeof(float)));
    checkCudaErrors(cudaMemcpy((void *) devcEpsilonArray, (void *) cEpsilonArray, numSVM * sizeof(float),
                               cudaMemcpyHostToDevice));

    int *devBlockWidthArray;
    checkCudaErrors(cudaMalloc((void **) (&devBlockWidthArray), numSVM * sizeof(int)));
    checkCudaErrors(cudaMemcpy((void *) devBlockWidthArray, (void *) blockWidthArray, numSVM * sizeof(int),
                               cudaMemcpyHostToDevice));

    int *deviLowArray;
    checkCudaErrors(cudaMalloc((void **) (&deviLowArray), numSVM * sizeof(int)));
    checkCudaErrors(
            cudaMemcpy((void *) deviLowArray, (void *) iLowArray, numSVM * sizeof(int), cudaMemcpyHostToDevice));

    int *deviHighArray;
    checkCudaErrors(cudaMalloc((void **) (&deviHighArray), numSVM * sizeof(int)));
    checkCudaErrors(
            cudaMemcpy((void *) deviHighArray, (void *) iHighArray, numSVM * sizeof(int), cudaMemcpyHostToDevice));

    //only SVMBlocks, each with 1 thread to do init
    launchMultiTakeFirstStep(devResultArray, devKernelDiagArray, devDataArray, devDataPitchInFloatsArray, devAlphaArray,
                             devCostArray, devnDimensionArray, deviLowArray, deviHighArray, kType, devParameterAArray,
                             devParameterBArray, devParameterCArray, singletonBlocks, singletonThreads);


    for (i = 0; i < numSVM; i++) {
        checkCudaErrors(cudaMemcpy(hostResultArray[i], h_devResultArray[i], 8 * sizeof(float), cudaMemcpyDeviceToHost));
    }

    float *alpha2OldArray = (float *) malloc(sizeof(float) * numSVM);
    float *alpha1OldArray = (float *) malloc(sizeof(float) * numSVM);
    float *alpha2NewArray = (float *) malloc(sizeof(float) * numSVM);
    float *alpha1NewArray = (float *) malloc(sizeof(float) * numSVM);
    float *alpha1DiffArray = (float *) malloc(sizeof(float) * numSVM);
    float *alpha2DiffArray = (float *) malloc(sizeof(float) * numSVM);
    int *iLowCacheIndexArray = (int *) malloc(sizeof(int) * numSVM);
    int *iHighCacheIndexArray = (int *) malloc(sizeof(int) * numSVM);
    bool *iLowComputeArray = (bool *) malloc(sizeof(bool) * numSVM);
    bool *iHighComputeArray = (bool *) malloc(sizeof(bool) * numSVM);

    for (i = 0; i < numSVM; i++) {
        alpha2OldArray[i] = *(hostResultArray[i] + 0);
        alpha1OldArray[i] = *(hostResultArray[i] + 1);
        bLowArray[i] = *(hostResultArray[i] + 2);
        bHighArray[i] = *(hostResultArray[i] + 3);
        alpha2NewArray[i] = *(hostResultArray[i] + 6);
        alpha1NewArray[i] = *(hostResultArray[i] + 7);

        alpha1DiffArray[i] = alpha1NewArray[i] - alpha1OldArray[i];
        alpha2DiffArray[i] = alpha2NewArray[i] - alpha2OldArray[i];
    }


    dim3 reduceThreads(BLOCKSIZE);

    printf("Starting iterations\n");

    int iteration = 1;
    SelectionHeuristic heuristicMethod = heuristicMethodArray[0];
    float *sAlpha1DiffArray = (float *) malloc(sizeof(float) * numSVM);
    float *sAlpha2DiffArray = (float *) malloc(sizeof(float) * numSVM);
    float *devsAlpha1DiffArray;
    checkCudaErrors(cudaMalloc((void **) (&devsAlpha1DiffArray), numSVM * sizeof(float)));
    float *devsAlpha2DiffArray;
    checkCudaErrors(cudaMalloc((void **) (&devsAlpha2DiffArray), numSVM * sizeof(float)));
    int *deviLowCacheIndexArray;
    checkCudaErrors(cudaMalloc((void **) (&deviLowCacheIndexArray), numSVM * sizeof(int)));
    int *deviHighCacheIndexArray;
    checkCudaErrors(cudaMalloc((void **) (&deviHighCacheIndexArray), numSVM * sizeof(int)));
    bool *deviLowComputeArray;
    checkCudaErrors(cudaMalloc((void **) (&deviLowComputeArray), numSVM * sizeof(bool)));
    bool *deviHighComputeArray;
    checkCudaErrors(cudaMalloc((void **) (&deviHighComputeArray), numSVM * sizeof(bool)));

    for (iteration = 1; true; iteration++) {
        bool converged = true;
        for (i = 0; i < numSVM; i++) {
            if (bLowArray[i] > bHighArray[i] + 2 * toleranceArray[i])  //SMO terminate
            {
                converged = false;
            }
        }

        if (converged) {
            printf("Converged\n");
            break;
        }

        if (iteration >= 16384) {
            printf("Exceed maximum iterations 16384\n");
            break;
        }

        if ((iteration & 0x7ff) == 0) {
            for (i = 0; i < numSVM; i++) {
                printf("iteration: %d; gap: %f\n", iteration, bLowArray[i] - bHighArray[i]);
            }
        }

        if ((iteration & 0x7f) == 0) {
            heuristicMethod = progress.getMethod();
        }

        for (i = 0; i < numSVM; i++) {
            //findData(index, offset, compute)
            //offset = position (offset) in the cache
            //compute = True if alrdy in cache, False if cache miss/
            kernelCacheArray[i]->findData(iHighArray[i], iHighCacheIndexArray[i], iHighComputeArray[i]);
            kernelCacheArray[i]->findData(iLowArray[i], iLowCacheIndexArray[i], iLowComputeArray[i]);
            sAlpha1DiffArray[i] = alpha1DiffArray[i] * labelsArray[i][iHighArray[i]];
            sAlpha2DiffArray[i] = alpha2DiffArray[i] * labelsArray[i][iLowArray[i]];
        }

        checkCudaErrors(cudaMemcpy((void *) devsAlpha1DiffArray, (void *) sAlpha1DiffArray, numSVM * sizeof(float),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *) devsAlpha2DiffArray, (void *) sAlpha2DiffArray, numSVM * sizeof(float),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *) deviLowCacheIndexArray, (void *) iLowCacheIndexArray, numSVM * sizeof(int),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy((void *) deviHighCacheIndexArray, (void *) iHighCacheIndexArray, numSVM * sizeof(int),
                           cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *) deviLowComputeArray, (void *) iLowComputeArray, numSVM * sizeof(bool),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy((void *) deviHighComputeArray, (void *) iHighComputeArray, numSVM * sizeof(bool),
                                   cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy((void *) deviLowArray, (void *) iLowArray, numSVM * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(
                cudaMemcpy((void *) deviHighArray, (void *) iHighArray, numSVM * sizeof(int), cudaMemcpyHostToDevice));

        if (heuristicMethod == FIRSTORDER) {
            launchMultiFirstOrder(numSVM, iLowComputeArray, iHighComputeArray, kType, devnPointsArray, nDimensionArray,
                                  devnDimensionArray, blocksLinear, threadsLinear, reduceThreads, devDataArray,
                                  devDataPitchInFloatsArray, devTransposedDataArray,
                                  devTransposedDataPitchInFloatsArray, devLabelsArray, devEpsilonArray,
                                  devcEpsilonArray, devAlphaArray, devFArray, devsAlpha1DiffArray, devsAlpha2DiffArray,
                                  deviLowArray, deviHighArray, devParameterAArray, devParameterBArray,
                                  devParameterCArray, devCacheArray, devCachePitchInFloatsArray, deviLowCacheIndexArray,
                                  deviHighCacheIndexArray, devLocalIndicesRLArray, devLocalIndicesRHArray,
                                  devLocalFsRHArray, devLocalFsRLArray, devKernelDiagArray, devResultArray,
                                  devCostArray, devBlockWidthArray, deviLowComputeArray, deviHighComputeArray);
        } else {
            launchMultiSecondOrder(numSVM, iLowComputeArray, iHighComputeArray, kType, devnPointsArray, nDimensionArray,
                                   devnDimensionArray, blocksLinear, threadsLinear, reduceThreads, devDataArray,
                                   devDataPitchInFloatsArray, devTransposedDataArray,
                                   devTransposedDataPitchInFloatsArray, devLabelsArray, devEpsilonArray,
                                   devcEpsilonArray, devAlphaArray, devFArray, devsAlpha1DiffArray, devsAlpha2DiffArray,
                                   deviLowArray, iHighArray, devParameterAArray, devParameterBArray, devParameterCArray,
                                   kernelCacheArray, devCacheArray, devCachePitchInFloatsArray, deviLowCacheIndexArray,
                                   iHighCacheIndexArray, devLocalIndicesRHArray, devLocalFsRHArray, devLocalFsRLArray,
                                   devLocalIndicesMaxObjArray, devLocalObjsMaxObjArray, devKernelDiagArray,
                                   devResultArray, hostResultArray, devCostArray, iteration, devBlockWidthArray,
                                   deviLowComputeArray, deviHighComputeArray, deviHighArray, deviHighCacheIndexArray,
                                   h_devResultArray);
        }

        for (i = 0; i < numSVM; i++) {
            checkCudaErrors(cudaMemcpy((void *) (hostResultArray[i]), h_devResultArray[i], 8 * sizeof(float),
                                       cudaMemcpyDeviceToHost));
            alpha2OldArray[i] = *(hostResultArray[i] + 0);
            alpha1OldArray[i] = *(hostResultArray[i] + 1);
            bLowArray[i] = *(hostResultArray[i] + 2);
            bHighArray[i] = *(hostResultArray[i] + 3);
            iLowArray[i] = *((int *) hostResultArray[i] + 6);
            iHighArray[i] = *((int *) hostResultArray[i] + 7);
            alpha2NewArray[i] = *(hostResultArray[i] + 4);
            alpha1NewArray[i] = *(hostResultArray[i] + 5);
            alpha1DiffArray[i] = alpha1NewArray[i] - alpha1OldArray[i];
            alpha2DiffArray[i] = alpha2NewArray[i] - alpha2OldArray[i];
        }
        progress.addIteration(bLowArray[0] - bHighArray[0]);
    }

    cuMemGetInfo(&remainingMemory, &totalMemory);
    printf("%Zu bytes of memory found on device, %Zu bytes currently free\n", totalMemory, remainingMemory);

    printf("%d iterations\n", iteration);
    for (i = 0; i < numSVM; i++) {
        //printf("svm: %d, bLow: %f, bHigh: %f\n", i, bLowArray[i], bHighArray[i]);
        kpArray[i]->b = (bLowArray[i] + bHighArray[i]) / 2;
        kernelCacheArray[i]->printStatistics();
        checkCudaErrors(cudaMemcpy((void *) (alphaArray[i]), h_devAlphaArray[i], nPointsArray[i] * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    }

    free(cEpsilonArray);
    free(kernelTypeArray);
    free(parameterAArray);
    free(parameterBArray);
    free(parameterCArray);
    for (i = 0; i < numSVM; i++) {
        cudaFree(h_devDataArray[i]);
        if (hostDataAllocedArray[i]) {
            free(hostDataArray[i]);
        }
        if (transposedDataAllocedArray[i]) {
            free(transposedDataArray[i]);
        }
        cudaFree(h_devTransposedDataArray[i]);
        cudaFree(h_devLabelsArray[i]);
        cudaFree(h_devKernelDiagArray[i]);
        cudaFree(h_devAlphaArray[i]);
        cudaFree(h_devFArray[i]);
        cudaFree(h_devResultArray[i]);

        //free(hostResultArray[i]);
        cudaFreeHost(hostResultArray[i]);

        cudaFree(h_devLocalFsRLArray[i]);
        cudaFree(h_devLocalFsRHArray[i]);
        cudaFree(h_devLocalIndicesRLArray[i]);
        cudaFree(h_devLocalIndicesRHArray[i]);
        cudaFree(h_devLocalObjsMaxObjArray[i]);
        cudaFree(h_devLocalIndicesMaxObjArray[i]);

        cudaFree(h_devCacheArray[i]);
    }
    free(h_devDataArray);
    free(h_devTransposedDataArray);
    free(h_devDataPitchArray);
    free(h_devTransposedDataPitchArray);
    free(hostPitchInFloatsArray);
    free(hostDataArray);
    free(hostDataAllocedArray);
    free(transposedDataAllocedArray);
    // free(alphaArray);
    free(h_devLabelsArray);
    free(h_devKernelDiagArray);
    free(h_devAlphaArray);
    free(h_devFArray);
    free(h_devResultArray);

    //free(hostResultArray);
    cudaFreeHost(hostResultArray);

    free(blockWidthArray);
    free(h_devLocalFsRLArray);
    free(h_devLocalFsRHArray);
    free(h_devLocalIndicesRLArray);
    free(h_devLocalIndicesRHArray);
    free(h_devLocalObjsMaxObjArray);
    free(h_devLocalIndicesMaxObjArray);
    free(rowPitchArray);

    free(sizeOfCacheArray);
    free(h_devCacheArray);
    free(cachePitchArray);
    free(kernelCacheArray);
    cudaFree(devCachePitchInFloatsArray);
    cudaFree(devDataPitchInFloatsArray);
    cudaFree(devTransposedDataPitchInFloatsArray);
    cudaFree(devnPointsArray);
    cudaFree(devnDimensionArray);
    cudaFree(devParameterAArray);
    cudaFree(devParameterBArray);
    cudaFree(devParameterCArray);
    cudaFree(devDataArray);
    cudaFree(devTransposedDataArray);
    cudaFree(devKernelDiagArray);
    cudaFree(devFArray);
    cudaFree(devLabelsArray);
    cudaFree(devAlphaArray);
    cudaFree(devResultArray);
    cudaFree(devCacheArray);
    cudaFree(devLocalIndicesRLArray);
    cudaFree(devLocalIndicesRHArray);
    cudaFree(devLocalFsRHArray);
    cudaFree(devLocalFsRLArray);
    cudaFree(devLocalObjsMaxObjArray);
    cudaFree(devLocalIndicesMaxObjArray);
    free(bLowArray);
    free(bHighArray);
    free(iterationArray);
    free(iLowArray);
    free(iHighArray);
    cudaFree(devCostArray);
    cudaFree(devEpsilonArray);
    cudaFree(devcEpsilonArray);
    cudaFree(devBlockWidthArray);
    cudaFree(deviLowArray);
    cudaFree(deviHighArray);
    free(alpha2OldArray);
    free(alpha1OldArray);
    free(alpha2NewArray);
    free(alpha1NewArray);
    free(alpha1DiffArray);
    free(alpha2DiffArray);
    free(iLowCacheIndexArray);
    free(iHighCacheIndexArray);
    free(iLowComputeArray);
    free(iHighComputeArray);
    free(sAlpha2DiffArray);
    free(sAlpha1DiffArray);
    cudaFree(devsAlpha1DiffArray);
    cudaFree(devsAlpha2DiffArray);
    cudaFree(deviLowCacheIndexArray);
    cudaFree(deviHighCacheIndexArray);
    cudaFree(deviLowComputeArray);
    cudaFree(deviHighComputeArray);

    //for profiling
    cudaDeviceReset();
}

void
performTraining(float *data, int nPoints, int nDimension, float *labels, float **p_alpha, Kernel_params *kp, float cost,
                SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float *transposedData) {
    chooseLargestGPU(true);

    float cEpsilon = cost - epsilon;
    Controller progress(2.0, heuristicMethod, 64, nPoints);

    int kType = GAUSSIAN;
    float parameterA;
    float parameterB;
    float parameterC;
    if (kp->kernel_type.compare(0, 3, "rbf") == 0) {
        parameterA = -kp->gamma;
        kType = GAUSSIAN;
        //printf("Gaussian kernel: gamma = %f\n", -parameterA);
    } else if (kp->kernel_type.compare(0, 10, "polynomial") == 0) {
        parameterA = kp->gamma;
        parameterB = kp->coef0;
        parameterC = kp->degree;
        kType = POLYNOMIAL;
        printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
    } else if (kp->kernel_type.compare(0, 6, "linear") == 0) {
        kType = LINEAR;
        printf("Linear kernel\n");
    } else if (kp->kernel_type.compare(0, 7, "sigmoid") == 0) {
        kType = SIGMOID;
        parameterA = kp->gamma;
        parameterB = kp->coef0;
        printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
        if ((parameterA <= 0) || (parameterB < 0)) {
            printf("Invalid Parameters\n");
            exit(1);
        }
    }
    //printf("Cost: %f, Tolerance: %f, Epsilon: %f\n", cost, tolerance, epsilon);


    float *devData;
    float *devTransposedData;
    size_t devDataPitch;
    size_t devTransposedDataPitch;
    int hostPitchInFloats = nPoints;

    float *hostData;
    bool hostDataAlloced = false;

    checkCudaErrors(cudaMallocPitch((void **) &devData, &devDataPitch, nPoints * sizeof(float), nDimension));
    if (devDataPitch == nPoints * sizeof(float)) {
        printf("Data is already aligned\n");
        hostData = data;
    } else {
        hostPitchInFloats = devDataPitch / sizeof(float);
        hostData = (float *) malloc(devDataPitch * nDimension);
        hostDataAlloced = true;
        //printf("Realigning data to a pitch of %i floats\n", hostPitchInFloats);
        for (int i = nDimension - 1; i >= 0; i--) {
            for (int j = nPoints - 1; j >= 0; j--) {
                hostData[i * hostPitchInFloats + j] = data[i * nPoints + j];
            }
        }
    }
    checkCudaErrors(cudaMemcpy(devData, hostData, devDataPitch * nDimension, cudaMemcpyHostToDevice));
    bool transposedDataAlloced = false;
    if (transposedData == 0) {
        transposedData = (float *) malloc(sizeof(float) * nPoints * nDimension);
        transposedDataAlloced = true;
        for (int i = 0; i < nPoints; i++) {
            for (int j = 0; j < nDimension; j++) {
                transposedData[i * nDimension + j] = hostData[j * hostPitchInFloats + i];
            }
        }
    }

    float *alpha = (float *) malloc(sizeof(float) * nPoints);
    *p_alpha = alpha;
    checkCudaErrors(cudaMallocPitch((void **) &devTransposedData, &devTransposedDataPitch, nDimension * sizeof(float),
                                    nPoints));
    checkCudaErrors(cudaMemcpy2D(devTransposedData, devTransposedDataPitch, transposedData, nDimension * sizeof(float),
                                 nDimension * sizeof(float),
                                 nPoints, cudaMemcpyHostToDevice));

    float *devLabels;
    checkCudaErrors(cudaMalloc((void **) &devLabels, nPoints * sizeof(float)));
    checkCudaErrors(cudaMemcpy(devLabels, labels, nPoints * sizeof(float), cudaMemcpyHostToDevice));


    float *devKernelDiag;
    checkCudaErrors(cudaMalloc((void **) &devKernelDiag, nPoints * sizeof(float)));


    float *devAlpha;
    checkCudaErrors(cudaMalloc((void **) &devAlpha, nPoints * sizeof(float)));

    float *devF;
    checkCudaErrors(cudaMalloc((void **) &devF, nPoints * sizeof(float)));

    void *devResult;
    checkCudaErrors(cudaMalloc(&devResult, 8 * sizeof(float)));
    float *hostResult = (float *) malloc(8 * sizeof(float));


    int blockWidth = intDivideRoundUp(nPoints, BLOCKSIZE);

    float *devLocalFsRL;
    checkCudaErrors(cudaMalloc((void **) &devLocalFsRL, blockWidth * sizeof(float)));
    float *devLocalFsRH;
    checkCudaErrors(cudaMalloc((void **) &devLocalFsRH, blockWidth * sizeof(float)));
    int *devLocalIndicesRL;
    checkCudaErrors(cudaMalloc((void **) &devLocalIndicesRL, blockWidth * sizeof(int)));
    int *devLocalIndicesRH;
    checkCudaErrors(cudaMalloc((void **) &devLocalIndicesRH, blockWidth * sizeof(int)));

    float *devLocalObjsMaxObj;
    checkCudaErrors(cudaMalloc((void **) &devLocalObjsMaxObj, blockWidth * sizeof(float)));
    int *devLocalIndicesMaxObj;
    checkCudaErrors(cudaMalloc((void **) &devLocalIndicesMaxObj, blockWidth * sizeof(int)));


    void *temp;
    size_t rowPitch;
    checkCudaErrors(cudaMallocPitch(&temp, &rowPitch, nPoints * sizeof(float), 2));
    checkCudaErrors(cudaFree(temp));


    size_t remainingMemory;
    size_t totalMemory;
    cuMemGetInfo(&remainingMemory, &totalMemory);

    size_t sizeOfCache = remainingMemory / rowPitch;
    sizeOfCache = (int) ((float) sizeOfCache * 0.95);//If I try to grab all the memory available, it'll fail
    if (nPoints < sizeOfCache) {
        sizeOfCache = nPoints;
    }

#ifdef __DEVICE_EMULATION__
    sizeOfCache = nPoints;
#endif

    printf("%Zu bytes of memory found on device, %Zu bytes currently free\n", totalMemory, remainingMemory);
    printf("%Zu rows of kernel matrix will be cached (%Zu bytes per row)\n", sizeOfCache, rowPitch);

    float *devCache;
    size_t cachePitch;
    checkCudaErrors(cudaMallocPitch((void **) &devCache, &cachePitch, nPoints * sizeof(float), sizeOfCache));
    //cudaMemset2D(devCache, cachePitch, 0x00, nPoints*sizeof(float), sizeOfCache);
    Cache kernelCache(nPoints, sizeOfCache);
    int devCachePitchInFloats = (int) cachePitch / (sizeof(float));

    cudaError_t err = cudaGetLastError();
    if (err) printf("Error: %s\n", cudaGetErrorString(err));
    printf("Allocated arrays on GPU\n");


    dim3 threadsLinear(BLOCKSIZE);
    dim3 blocksLinear(blockWidth);


    int devDataPitchInFloats = ((int) devDataPitch) >> 2;
    int devTransposedDataPitchInFloats = ((int) devTransposedDataPitch) >> 2;


    launchInitialization(devData, devDataPitchInFloats, nPoints, nDimension, kType, parameterA, parameterB, parameterC,
                         devKernelDiag, devAlpha, devF, devLabels, blocksLinear, threadsLinear);
    err = cudaGetLastError();
    if (err) printf("Error: %s\n", cudaGetErrorString(err));
    printf("Initialization complete\n");

    //Choose initial points
    float bLow = 1;
    float bHigh = -1;
    int iteration = 0;
    int iLow = -1;
    int iHigh = -1;
    for (int i = 0; i < nPoints; i++) {
        if (labels[i] < 0) {
            if (iLow == -1) {
                iLow = i;
                if (iHigh > -1) {
                    i = nPoints; //Terminate
                }
            }
        } else {
            if (iHigh == -1) {
                iHigh = i;
                if (iLow > -1) {
                    i = nPoints; //Terminate
                }
            }
        }
    }


    dim3 singletonThreads(1);
    dim3 singletonBlocks(1);
    launchTakeFirstStep(devResult, devKernelDiag, devData, devDataPitchInFloats, devAlpha, cost, nDimension, iLow,
                        iHigh, kType, parameterA, parameterB, parameterC, singletonBlocks, singletonThreads);
    checkCudaErrors(cudaMemcpy((void *) hostResult, devResult, 8 * sizeof(float), cudaMemcpyDeviceToHost));


    float alpha2Old = *(hostResult + 0);
    float alpha1Old = *(hostResult + 1);
    bLow = *(hostResult + 2);
    bHigh = *(hostResult + 3);
    float alpha2New = *(hostResult + 6);
    float alpha1New = *(hostResult + 7);

    float alpha1Diff = alpha1New - alpha1Old;
    float alpha2Diff = alpha2New - alpha2Old;

    int iLowCacheIndex;
    int iHighCacheIndex;
    bool iLowCompute;
    bool iHighCompute;


    dim3 reduceThreads(BLOCKSIZE);


    printf("Starting iterations\n");

    for (iteration = 1; true; iteration++) {

        if (bLow <= bHigh + 2 * tolerance) {
            printf("Converged\n");
            break; //Convergence!!
        }

        if ((iteration & 0x7ff) == 0) {
            printf("iteration: %d; gap: %f\n", iteration, bLow - bHigh);
        }

        if ((iteration & 0x7f) == 0) {
            heuristicMethod = progress.getMethod();
        }


        kernelCache.findData(iHigh, iHighCacheIndex, iHighCompute);

        kernelCache.findData(iLow, iLowCacheIndex, iLowCompute);


        if (heuristicMethod == FIRSTORDER) {
            launchFirstOrder(iLowCompute, iHighCompute, kType, nPoints, nDimension, blocksLinear, threadsLinear,
                             reduceThreads, devData, devDataPitchInFloats, devTransposedData,
                             devTransposedDataPitchInFloats, devLabels, epsilon, cEpsilon, devAlpha, devF,
                             alpha1Diff * labels[iHigh], alpha2Diff * labels[iLow], iLow, iHigh, parameterA, parameterB,
                             parameterC, devCache, devCachePitchInFloats, iLowCacheIndex, iHighCacheIndex,
                             devLocalIndicesRL, devLocalIndicesRH, devLocalFsRH, devLocalFsRL, devKernelDiag, devResult,
                             cost);
        } else {
            launchSecondOrder(iLowCompute, iHighCompute, kType, nPoints, nDimension, blocksLinear, threadsLinear,
                              reduceThreads, devData, devDataPitchInFloats, devTransposedData,
                              devTransposedDataPitchInFloats, devLabels, epsilon, cEpsilon, devAlpha, devF,
                              alpha1Diff * labels[iHigh], alpha2Diff * labels[iLow], iLow, iHigh, parameterA,
                              parameterB, parameterC, &kernelCache, devCache, devCachePitchInFloats, iLowCacheIndex,
                              iHighCacheIndex, devLocalIndicesRH, devLocalFsRH, devLocalFsRL, devLocalIndicesMaxObj,
                              devLocalObjsMaxObj, devKernelDiag, devResult, hostResult, cost, iteration);
        }
        checkCudaErrors(cudaMemcpy((void *) hostResult, devResult, 8 * sizeof(float), cudaMemcpyDeviceToHost));

        alpha2Old = *(hostResult + 0);
        alpha1Old = *(hostResult + 1);
        bLow = *(hostResult + 2);
        bHigh = *(hostResult + 3);
        iLow = *((int *) hostResult + 6);
        iHigh = *((int *) hostResult + 7);
        alpha2New = *(hostResult + 4);
        alpha1New = *(hostResult + 5);
        alpha1Diff = alpha1New - alpha1Old;
        alpha2Diff = alpha2New - alpha2Old;
        progress.addIteration(bLow - bHigh);

    }


    printf("%d iterations\n", iteration);
    printf("bLow: %f, bHigh: %f\n", bLow, bHigh);
    kp->b = (bLow + bHigh) / 2;
    kernelCache.printStatistics();
    checkCudaErrors(cudaMemcpy((void *) alpha, devAlpha, nPoints * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(devData);
    cudaFree(devTransposedData);
    cudaFree(devLabels);
    cudaFree(devAlpha);
    cudaFree(devF);
    cudaFree(devCache);
    cudaFree(devLocalIndicesRL);
    cudaFree(devLocalIndicesRH);
    cudaFree(devLocalFsRH);
    cudaFree(devLocalFsRL);
    cudaFree(devKernelDiag);
    cudaFree(devResult);
    cudaFree(devLocalIndicesMaxObj);
    cudaFree(devLocalObjsMaxObj);
    if (hostDataAlloced) {
        free(hostData);
    }
    if (transposedDataAlloced) {
        free(transposedData);
    }
    free(hostResult);
}
