#include <algorithm>
#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <cstring>
#include <random>
#include "../common/framework.h"
#include "svmCommon.h"

using namespace std;


struct threadData {
  unsigned int seed;
  int nPoints, nDim, dim1, nTestPoints;
  float *dataArrayIn, *transposedDataArrayIn, *labelsArrayIn, *testDataArrayIn, *testLablesArrayIn;
  float *dataArrayOut, *transposedDataArrayOut, *labelsArrayOut, *testDataArrayOut, *testLabelsArrayOut;
  float *transposedDataArrayIn2, *transposedDataArrayOut2;
};

void *splitThread(void *threadarg) {
    struct threadData *mydata;
    mydata = (struct threadData *) threadarg;

    unsigned int seed = mydata->seed;
    int i, j, nPoints = mydata->nPoints, nDim = mydata->nDim, dim1 = mydata->dim1, dim2 = nDim - dim1;
    int nTestPoints = mydata->nTestPoints;
    float *dataArrayIn = mydata->dataArrayIn;
    float *transposedDataArrayIn = mydata->transposedDataArrayIn;
    float *labelsArrayIn = mydata->labelsArrayIn;
    float *dataArrayOut = mydata->dataArrayOut, *transposedDataArrayOut = mydata->transposedDataArrayOut;
    float *labelsArrayOut = mydata->labelsArrayOut;
    float *transposedDataArrayIn2 = mydata->transposedDataArrayIn2;
    float *transposedDataArrayOut2 = mydata->transposedDataArrayOut2;

    float *testDataArrayIn = mydata->testDataArrayIn, *testLablesArrayIn = mydata->testLablesArrayIn;
    float *testDataArrayOut = mydata->testDataArrayOut, *testLabelsArrayOut = mydata->testLabelsArrayOut;

    vector<int> permutation;
    // srand(seed);
    // POSIX not guarantee thread-safe srand
    for (int i = 0; i < nDim; i++) {
        permutation.push_back(i);
    }
    std::shuffle(permutation.begin(), permutation.end(), default_random_engine(seed));

    for (i = 0; i < nDim; i++) //col major
    {
        for (j = 0; j < nPoints; j++) {
            dataArrayOut[i * nPoints + j] = dataArrayIn[permutation[i] * nPoints + j];
        }
    }

    for (i = 0; i < nDim; i++) {
        for (j = 0; j < nTestPoints; j++) {
            testDataArrayOut[i * nTestPoints + j] = testDataArrayIn[permutation[i] * nTestPoints + j];
        }
    }
    for (i = 0; i < nTestPoints; i++)
        testLabelsArrayOut[i] = testLabelsArrayOut[i + nTestPoints] = testLablesArrayIn[i];

    for (i = 0; i < nPoints; i++) //row major
    {
        labelsArrayOut[i] = labelsArrayOut[i + nPoints] = labelsArrayIn[i];
        int offset = (nDim + 1) / 2;

        for (j = 0; j < offset; j++) //process two new svms together
        {
            transposedDataArrayOut[i * offset + j] = transposedDataArrayIn[i * offset + permutation[2 * j]];
            if (2 * j + 1 < nDim)
                transposedDataArrayOut2[i * offset + j] =
                    transposedDataArrayIn[i * offset + permutation[2 * j + 1]];
        }
    }
    return NULL;
}

void splitfeatures(int numSVM,
                   int numSVMfinal,
                   int folder,
                   int *nPointsArray,
                   int *nPointsArrayOut,
                   int *nDimensionArray,
                   int *nDimensionArrayOut,
                   float **dataArray,
                   float **dataArrayOut,
                   float **transposedDataArray,
                   float **transposedDataArrayOut,
                   float **testDataArray,
                   float **testDataArrayOut,
                   int *testNPointsArray,
                   int *testNPointsArrayOut,
                   float **testLabelsArray,
                   float **testLabelsArrayOut,
                   float **labelsArray,
                   float **labelsArrayOut,
                   Kernel_params **kpArray,
                   Kernel_params **kpArrayOut,
                   float *costArray,
                   float *costArrayOut,
                   SelectionHeuristic *heuristicMethodArray,
                   SelectionHeuristic *heuristicMethodArrayOut,
                   float *epsilonArray,
                   float *epsilonArrayOut,
                   float *toleranceArray,
                   float *toleranceArrayOut) {

    int rc, i, j;
    pthread_attr_t attr;
    void *status;
    pthread_attr_init(&attr);
    vector<pthread_t> threads;
    pthread_t *newThread;
    struct threadData *td;

    // every folder svms have the same randonseeds
    srand(FIRSTSEED);
    unsigned int *secondseeds = (unsigned int *) malloc(sizeof(unsigned int) * (nDimensionArray[0] + 1));
    for (i = 0; i < nDimensionArray[0]; i++)
        secondseeds[i] = rand();

    // only 1 input svm
    for (i = 0; i < folder; i++) //for a cv
    {
        int nPoints = nPointsArray[i];
        int nDim = nDimensionArray[i];
        int nTestPoints = testNPointsArray[i];
        //printf("npoints %d ndim %d\n", nPoints, nDim);
        int offset = nDim * 2;

        int dim1 = (nDim + 1) / 2;

        int dim2 = nDim - dim1;

        printf("nDim %d dim1 %d dim2 %d \n", nDim, dim1, dim2);
        for (j = 0; j < nDim * 2; j += 2) //split nDim times, every time produce 2 svms
        {
            //printf("doing %d seed = %u\n", i*offset+j, secondseeds[j>>1]);

            td = (struct threadData *) malloc(sizeof(struct threadData));

            td->seed = secondseeds[j >> 1];
            td->nPoints = nPoints;
            td->nDim = nDim;
            td->nTestPoints = nTestPoints;

            td->dataArrayIn = dataArray[i];

            td->transposedDataArrayIn = transposedDataArray[i];
            td->labelsArrayIn = labelsArray[i];
            td->testDataArrayIn = testDataArray[i];
            td->testLablesArrayIn = testLabelsArray[i];

            dataArrayOut[i * offset + j] = (float *) malloc(sizeof(float) * nDim * nPoints);
            dataArrayOut[i * offset + j + 1] = &(dataArrayOut[i * offset + j][dim1 * nPoints]);

            transposedDataArrayOut[i * offset + j] = (float *) malloc(sizeof(float) * nDim * nPoints);
            transposedDataArrayOut[i * offset + j + 1] = &(transposedDataArrayOut[i * offset + j][dim1 * nPoints]);

            labelsArrayOut[i * offset + j] = (float *) malloc(sizeof(float) * nPoints * 2);
            labelsArrayOut[i * offset + j + 1] = &(labelsArrayOut[i * offset + j][nPoints]);

            testDataArrayOut[i * offset + j] = (float *) malloc(sizeof(float) * nDim * nTestPoints);
            testDataArrayOut[i * offset + j + 1] = &(testDataArrayOut[i * offset + j][dim1 * nTestPoints]);

            testLabelsArrayOut[i * offset + j] = (float *) malloc(sizeof(float) * nTestPoints * 2);
            testLabelsArrayOut[i * offset + j + 1] = &(testLabelsArrayOut[i * offset + j][nTestPoints]);

            td->dataArrayOut = dataArrayOut[i * offset + j];
            td->transposedDataArrayOut = transposedDataArrayOut[i * offset + j];
            td->transposedDataArrayOut2 = transposedDataArrayOut[i * offset + j + 1];
            td->labelsArrayOut = labelsArrayOut[i * offset + j];
            td->testDataArrayOut = testDataArrayOut[i * offset + j];
            td->testLabelsArrayOut = testLabelsArrayOut[i * offset + j];

            newThread = (pthread_t *) malloc(sizeof(pthread_t));

            // create new thread for spliting
            rc = pthread_create(newThread, &attr, splitThread, (void *) td);
            while (rc) {
                cout << "Error: unable to create thread, " << rc << ", trying again" << endl;
                rc = pthread_create(newThread, &attr, splitThread, (void *) td);

            }
            //printf("thread %lld OK\n", *newThread);
            threads.push_back(*newThread);

            nPointsArrayOut[i * offset + j] = nPoints;
            nPointsArrayOut[i * offset + j + 1] = nPoints;
            nDimensionArrayOut[i * offset + j] = dim1;
            nDimensionArrayOut[i * offset + j + 1] = dim2;

            kpArrayOut[i * offset + j] = (Kernel_params *) malloc(sizeof(Kernel_params) * 2);
            kpArrayOut[i * offset + j + 1] = &kpArrayOut[i * offset + j][1];
            memcpy(kpArrayOut[i * offset + j], kpArray[i], sizeof(Kernel_params));
            memcpy(kpArrayOut[i * offset + j + 1], kpArray[i], sizeof(Kernel_params));


            costArrayOut[i * offset + j] = costArray[i];
            costArrayOut[i * offset + j + 1] = costArray[i];
            heuristicMethodArrayOut[i * offset + j] = heuristicMethodArray[i];
            heuristicMethodArrayOut[i * offset + j + 1] = heuristicMethodArray[i];
            epsilonArrayOut[i * offset + j] = epsilonArray[i];
            epsilonArrayOut[i * offset + j + 1] = epsilonArray[i];
            toleranceArrayOut[i * offset + j] = toleranceArray[i];
            toleranceArrayOut[i * offset + j + 1] = toleranceArray[i];
            testNPointsArrayOut[i * offset + j] = testNPointsArray[i];
            testNPointsArrayOut[i * offset + j + 1] = testNPointsArray[i];
        }

    }

    // free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for (i = 0; i < threads.size(); i++) {
        //printf("join thread %lld \n", threads[i]);
        rc = pthread_join(threads[i], NULL);
        if (rc) {
            cout << "Error: unable to join, " << rc << endl;
            exit(-1);
        }
    }
    //printf("Exit Split\n");

//    for(i=0;i<folder;i++)
//    {
//        free(dataArrayIn[i]);
//        free(transposedDataArrayIn[i]);
//        free(labelsArrayIn[i]);
//        free(nPointsArray[i]);
//        free(nDimensionArray[i]);
//        free(kpArray[i]);
//        free(costArray[i]);
//        free(heuristicMethodArray[i]);
//        free(epsilonArray[i]);
//        free(toleranceArray[i]);
//        free(testDataArray[i]);
//        free(testNPointsArray[i]);
//        free(testLabelsArray[i]);
//    }


    // only free the old pointer pointing to the array of these values, but the values are kept in memory and reuse later
    free(dataArray);
    free(transposedDataArray);
    free(labelsArray);
    free(nPointsArray);
    free(nDimensionArray);
    free(kpArray);
    free(costArray);
    free(heuristicMethodArray);
    free(epsilonArray);
    free(toleranceArray);
    free(testDataArray);
    free(testNPointsArray);
    free(testLabelsArray);
    free(secondseeds);
}
