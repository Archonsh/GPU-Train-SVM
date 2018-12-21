/*
 * example_letter.cu
 *
 *  Add on: Aug 14, 2016
 *  Add by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  create a running example of one-against-all multi-class training
 */

#include "../include/svmTrain.h"
#include "../include/svmCommon.h"
#include "../include/svmClassify.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <sstream>

using namespace std;

int readSvm(const char* filename, float** p_data, float** p_labels, int* p_npoints, int* p_dimension, float** p_transposed_data = 0);
int readModel(const char* filename, float** alpha, float** supportVectors, int* nSVOut, int* nDimOut, Kernel_params* kp, float* p_class1Label, float* p_class2Label);
void printClassification(const char* outputFile, float* result, int nPoints);

int main(int argc, char * argv[]) {
	int i;

	if(argc != 3) {
		cout << "Usage: example_bin train_file test_file" << endl;
		return 0;
	}

	char* dataFileChar = argv[1];
	char* queryFileChar = argv[2];

	int nPoints;
	int nDimension;
	float* data;
	float* transposedData;
	float* labels;
	// read in data from input files
	readSvm(dataFileChar, &data, &labels, &nPoints, &nDimension, &transposedData);

	// category stores all classes for the training set
	vector<float> category;
	for(i = 0; i < nPoints; i++) {
		if(find(category.begin(), category.end(), labels[i]) == category.end()) {
			category.push_back(labels[i]);
		}
	}

	// in this example we test on Gaussian kernel cross validation
	int kernelType = GAUSSIAN;

	// setup variables for cross validation
	int numSVM = category.size();
	float** labelsArray = (float**)malloc(sizeof(float*)*numSVM);
	float** dataArray = (float**) malloc(sizeof(float*)*numSVM);
	float** transposedDataArray = (float**)malloc(sizeof(float*)*numSVM);
	int* nPointsArray = (int*)malloc(sizeof(int)*numSVM);
	int* nDimensionArray = (int*)malloc(sizeof(int)*numSVM);
	Kernel_params** kpArray = (Kernel_params**)malloc(sizeof(Kernel_params*)*numSVM);
	float* costArray = (float*)malloc(sizeof(float)*numSVM);
	SelectionHeuristic* heuristicMethodArray = (SelectionHeuristic*)malloc(sizeof(SelectionHeuristic)*numSVM);
	float* epsilonArray = (float*)malloc(sizeof(float)*numSVM);
	float* toleranceArray = (float*)malloc(sizeof(float)*numSVM);

	//set default value of parameters for training and cross validation
	float cost = 1.0f;
	float tolerance = 1e-3f;
	float epsilon = 1e-5f;
	SelectionHeuristic heuristicMethod = ADAPTIVE;

  	Kernel_params kp;

 	for(i = 0; i < numSVM; i++) {
    	nPointsArray[i] = nPoints;
    	nDimensionArray[i] = nDimension;

    	labelsArray[i] = (float*) malloc(sizeof(float) * nPoints);
    	int j;

    	// set labelsArray for one-against-all cross valiation
    	for(j = 0; j < nPoints; j++) {
    		if(labels[j] == category[i])
    			labelsArray[i][j] = 1;
    		else
    			labelsArray[i][j] = -1;
    	}

    	dataArray[i] = data;
    	transposedDataArray[i] = transposedData;

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
			parameterA = 1.0/nPointsArray[i];
			parameterB = 0.0f;
			if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) {
				printf("Invalid parameters\n");
				exit(1);
			}
			kp.kernel_type = "polynomial";
			kp.gamma = parameterA;
			kp.coef0 = parameterB;
			kp.degree = (int)parameterC;
	    } else if (kernelType == GAUSSIAN) {
			parameterA = 1.0/nDimensionArray[i];
			if (parameterA < 0) {
				printf("Invalid parameters\n");
				exit(1);
			}
			kp.kernel_type = "rbf";
			kp.gamma = parameterA;
	    } else if (kernelType == SIGMOID) {
			parameterA = 1.0/nPointsArray[i];
			parameterB = 0.0f;
			if ((parameterA <= 0) || (parameterB < 0)) {
				printf("Invalid Parameters\n");
				exit(1);
			}
			kp.kernel_type = "sigmoid";
			kp.gamma = parameterA;
			kp.coef0 = parameterB;
	    }
    	memcpy(kpArray[i], &kp, sizeof(Kernel_params));
    }
    
    cout << "Finish Setting Parameters!" << endl;

    //1. Train models and output models
    svmTrain(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, NULL, 0.5);
    
    float** results = (float **) malloc(sizeof(float*)*numSVM);

    float* queryData;
    float* queryLabels;
    int querynPoints;
    int queryDimension;
    
    // Read in query data
    readSvm(queryFileChar, &queryData, &queryLabels, &querynPoints, &queryDimension);

    //2. Load models and make prediction
	for(i = 0; i < numSVM; i++) {
		int nSV;
		int mDimension;		
		float* alpha;		
		float* supportVectors;	

		struct Kernel_params kp;
	  
		float class1Label, class2Label;

		stringstream ss;
		ss << i;
		string filename = "./data/svm" + ss.str() + ".mdl";

		int success = readModel(filename.c_str(), &alpha, &supportVectors, &nSV, &mDimension, &kp, &class1Label, &class2Label);
		if (success == 0) {
			printf("Invalid Model\n");
			exit(1);
		}
	
		performClassification(queryData, querynPoints, supportVectors, nSV, mDimension, alpha, kp, &(results[i]));	
	}

	// classification stores the final classification for each query point
   	float* classification = (float *) malloc(sizeof(float) * querynPoints);
   	for(i = 0; i < querynPoints; i++) {
   		int j;
   		float max = results[0][i];
   		classification[i] = category[0];
   		for(j = 0; j < numSVM; j++) {
   			if(results[j][i] > max) {
   				max = results[j][i];
   				classification[i] = category[j];
   			}
   		}
   	}

   	// calculate accuracy
   	int correct = 0;
   	for(i = 0; i < querynPoints; i++) {
   		if(classification[i] == queryLabels[i]) {
   			correct ++;
   		}
   	}
   	float accuracy = (100 * (float) correct ) / querynPoints;
    free(classification);

    //3. cross validation on parameter set (cost, gamma), modify this to cross validation on different parameter sets
    cost = 1.0;
    float gamma = 1.0/nDimension;
    for(i = 0; i < numSVM; i++) {
    	costArray[i] = cost;
    	kpArray[i]->gamma = gamma;
    }

    int folder = 5;
    cout << "Start cross-validation" << endl;

    // mcvResult stores the result of multi-class cross-validation
    float ** mcvResult;
	svmCrossValidation(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, 1, true, &mcvResult);

	// calculate the accuracy of the corss-validation on parameter set (cost, gamma)
	int* numTestingPointsEachFolder = (int*) malloc(sizeof(int) * folder);
	for(i = 0; i < folder; i++) {
		numTestingPointsEachFolder[i] = nPoints / folder;
		if(i < nPoints % folder)
			numTestingPointsEachFolder[i] += 1;
	}

	// mcvClassification stores the classification for each data point in cross validation
	float* mcvClassification = (float*) malloc(sizeof(float) * nPoints);
	for(i = 0; i < folder; i++) {
		int j;
		for(j = 0; j < numTestingPointsEachFolder[i]; j++) {
			float max = mcvResult[i][j];
			mcvClassification[i+j*folder] = category[0];
			int p;
			for(p = 0; p < numSVM; p++) {
				if(mcvResult[i+p*folder][j] > max) {
					max = mcvResult[i+p*folder][j];
					mcvClassification[i+j*folder] = category[p];
				}
			}
		}
	}

	// calculate cross validation accuracy
	int corrLabel = 0;
	for(i = 0; i < nPoints; i++) {
		if(mcvClassification[i] == labels[i]) {
			corrLabel ++;
		}
	}

	cout << "Accuracy: " << accuracy << "%" << endl;
	cout << "Cross validation accuracy: " << (100.0 * (float) corrLabel) / nPoints << "%" << endl;
	free(mcvClassification);

	cout << "EXIT." << endl;

	// free pointers	
	free(nPointsArray);
	free(nDimensionArray);
	for(i = 0; i < numSVM; i++) {
		free(labelsArray[i]);
		free(kpArray[i]);
	}
	free(dataArray);
	free(labelsArray);
	free(kpArray);
	free(costArray);
	free(heuristicMethodArray);
	free(epsilonArray);
	free(toleranceArray);
	free(transposedDataArray);
}