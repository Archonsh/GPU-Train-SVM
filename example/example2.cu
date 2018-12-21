/*
 * example_letter2.cu
 *
 *  Add on: Dec 1, 2016
 *  Add by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  create a running example of one-against-one multi-class training
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
#include <sys/time.h> 

// include code for data setup
#include "./common/setupData.cu"

using namespace std;

int readSvm(const char* filename, float** p_data, float** p_labels, int* p_npoints, int* p_dimension, float** p_transposed_data = 0);
int readModel(const char* filename, float** alpha, float** supportVectors, int* nSVOut, int* nDimOut, Kernel_params* kp, float* p_class1Label, float* p_class2Label);
void printClassification(const char* outputFile, float* result, int nPoints);

// if outputModel is ture, the function will output the model file
float multiClassTrainAndPredict(int nPoints, int nDimension, float* labels, float* transposedData, int querynPoints, float* queryLabels, float* queryData, float* inCostArray, float* gammaArray, bool outputModel = false) {
	int i;

	// category stores all classes for the training set
	vector<float> category;
	for(i = 0; i < nPoints; i++) {
		if(find(category.begin(), category.end(), labels[i]) == category.end()) {
			category.push_back(labels[i]);
		}
	}

	// classDist stores the class distribution of the data set
	int* classDist = (int*)calloc(category.size(), sizeof(int));
	for(i = 0; i < nPoints; i++) {
		int idx = find(category.begin(), category.end(), labels[i]) - category.begin();
		classDist[idx] ++;
	}

	// in this example we test on Gaussian kernel cross validation
	int kernelType = GAUSSIAN;

	// setup variables for training and cross validation
	int numSVM = category.size()*(category.size()-1)/2;
	float** labelsArray = (float**)malloc(sizeof(float*)*numSVM);
	float** dataArray = (float**)malloc(sizeof(float*)*numSVM);
	float** transposedDataArray = (float**)malloc(sizeof(float*)*numSVM);
	int* nPointsArray = (int*)malloc(sizeof(int)*numSVM);
	int* nDimensionArray = (int*)malloc(sizeof(int)*numSVM);
	Kernel_params** kpArray = (Kernel_params**)malloc(sizeof(Kernel_params*)*numSVM);
	float* costArray = (float*)malloc(sizeof(float)*numSVM);
	SelectionHeuristic* heuristicMethodArray = (SelectionHeuristic*)malloc(sizeof(SelectionHeuristic)*numSVM);
	float* epsilonArray = (float*)malloc(sizeof(float)*numSVM);
	float* toleranceArray = (float*)malloc(sizeof(float)*numSVM);

	//set default value of parameters for cross validation
	// float cost = 1.0f;
	float tolerance = 1e-3f;
	float epsilon = 1e-5f;
	SelectionHeuristic heuristicMethod = ADAPTIVE;

  	Kernel_params kp;
  	int classA = 0;
  	int classB = 1;
 	for(i = 0; i < numSVM; i++) {
    	nPointsArray[i] = classDist[classA] + classDist[classB];
    	nDimensionArray[i] = nDimension;

    	labelsArray[i] = (float*) malloc(sizeof(float) * nPointsArray[i]);
    	int j;

    	// set labelsArray for one-against-one cross valiation
    	for(j = 0; j < nPointsArray[i]; j++) {
    		if(j < classDist[classA])
    			labelsArray[i][j] = 1;
    		else
    			labelsArray[i][j] = -1;
    	}

    	if(classB == category.size()-1) {
    		classA += 1;
    		classB = classA + 1;
    	} else 
    		classB ++;
    	
    	dataArray[i] = (float*) malloc(sizeof(float) * nPointsArray[i] * nDimension);
    	transposedDataArray[i] = (float*) malloc(sizeof(float) * nPointsArray[i] * nDimension);

    	// costArray[i] = cost;
	    costArray[i] = inCostArray[i];
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
			// parameterA = 1.0/nDimensionArray[i];
			parameterA = gammaArray[i];
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

    setupData(numSVM, category, classDist, nPoints, nDimension, transposedData, labels, nPointsArray, dataArray, transposedDataArray);

    cout << "Finish Setting Parameters!" << endl;

    float** results;
    if(!outputModel) {
    	float** testDataArray = (float**)malloc(sizeof(float*)*numSVM);
    	int* testNPointsArray = (int*)malloc(sizeof(int)*numSVM);
    	float** testLabelsArray = (float**)malloc(sizeof(float*)*numSVM);
    	for(i = 0; i < numSVM; i++) {
    		testDataArray[i] = queryData;
    		testNPointsArray[i] = querynPoints;
    		testLabelsArray[i] = queryLabels;
    	}

    	svmPredict(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, testDataArray, testNPointsArray, testLabelsArray, NULL, 0.5, true, &results);
    	
    	// free pointers
    	free(testDataArray);
    	free(testNPointsArray);
    	free(testLabelsArray);
    } else {
    	// Train models and output models
	    svmTrain(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, NULL, 0.5);

	    results = (float **) malloc(sizeof(float*)*numSVM);

	    // Load models and make prediction
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
			cout << "model: " << i << " " << filename << endl;
			if (success == 0) {
				printf("Invalid Model\n");
				exit(1);
			}
		
			performClassification(queryData, querynPoints, supportVectors, nSV, mDimension, alpha, kp, &(results[i]));	
		}
    }

	// classification stores the final classification for each query point
   	float* classification = (float *) malloc(sizeof(float) * querynPoints);
   	int* vote = (int*) malloc(sizeof(int) * category.size());
   	for(i = 0; i < querynPoints; i++) {
   		int j;
   		for(j = 0; j < category.size(); j++)
   			vote[j] = 0;
   		for(j = 0; j < numSVM; j++) {
   			int classA = 0;
			while(j >= (classA+1)*(category.size()-1)-(classA+1)*classA/2)
				classA ++;
			int classB = j - classA*(category.size()-1) + classA*(classA-1)/2 + classA + 1;

			if(results[j][i] > 0)
				vote[classA] ++;
			else
				vote[classB] ++;
   		}

   		int max = vote[0];
   		classification[i] = category[0];
   		for(j = 0; j < category.size(); j++) {
   			if(vote[j] > max) {
   				max = vote[j];
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
    free(vote);
    free(classification);

    // free pointers	
	free(nPointsArray);
	free(nDimensionArray);
	for(i = 0; i < numSVM; i++) {
		free(labelsArray[i]);
		free(transposedDataArray[i]);
		free(kpArray[i]);
		free(dataArray[i]);
	}
	free(dataArray);
	free(labelsArray);
	free(kpArray);
	free(costArray);
	free(heuristicMethodArray);
	free(epsilonArray);
	free(toleranceArray);
	free(transposedDataArray);

    return accuracy;
}

// cross validation on parameter costArray and gammaArray for RBF kernel
float multiClassCrossValidation(int folder, float* costArray, float* gammaArray, int nPoints, int nDimension, float* labels, float* transposedData) {
	int i, j;

	// random split the data set
	int* permutation = (int*)malloc(sizeof(int)*nPoints);
	for(i = 0; i < nPoints; i++)
		permutation[i] = i;
	srand(unsigned(time(0)));
	random_shuffle(permutation, permutation+nPoints);

	int* dataPartitionSize = (int*)malloc(sizeof(int)*folder);
	for(i = 0; i < folder; i++) {
		dataPartitionSize[i] = nPoints / folder;
		if(nPoints % folder != 0 && i < nPoints % folder)
			dataPartitionSize[i] ++;
	}

	float** dataPartitionArray = (float**)malloc(sizeof(float*)*folder);
	float** dataTranPartitionArray = (float**)malloc(sizeof(float*)*folder);
	for(i = 0; i < folder; i++) {
		dataPartitionArray[i] = (float*)malloc(sizeof(float)*dataPartitionSize[i]*nDimension);
		dataTranPartitionArray[i] = (float*)malloc(sizeof(float)*dataPartitionSize[i]*nDimension);
	}

	// randomly partition the data set into folder parts, each part with size dataPartitionSize[i]
	setupDataCV(folder, nPoints, nDimension, transposedData, permutation, dataPartitionArray, dataTranPartitionArray);

	float* results = (float*) malloc(sizeof(float)*folder);

	int nPointsPar;
	float* transposedDataPar = (float*)malloc(sizeof(float)*(nPoints-nPoints/folder)*nDimension);
	float* labelsPar = (float*)malloc(sizeof(float)*(nPoints-nPoints/folder));
	int querynPoints;
	float* queryData;
	float* queryLabels = (float*)malloc(sizeof(float)*(nPoints/folder+1));
	// find the accuracy results of training on (folder-1) parts and testing on the rest part
	for(i = 0; i < folder; i++) {
		nPointsPar = nPoints - dataPartitionSize[i];
		querynPoints = dataPartitionSize[i];

		int qLabelIdx = 0;
		int dLabelIdx = 0;
		for(j = 0; j < folder; j++) {
			int p = j;
			while(p < nPoints) {
				if(j == i) {
					queryLabels[qLabelIdx] = labels[permutation[p]];
					qLabelIdx ++;
				} else {
					labelsPar[dLabelIdx] = labels[permutation[p]];
					dLabelIdx++;
				}
				p += folder;
			}
		}

		int accuPoints = 0;
		for(j = 0; j < folder; j++) {
			if(j == i) {
				queryData = dataPartitionArray[i];
			} else {
				memcpy(transposedDataPar+accuPoints*nDimension, dataTranPartitionArray[j], sizeof(float)*dataPartitionSize[j]*nDimension);
				accuPoints += dataPartitionSize[j];
			}
		}

		results[i] = multiClassTrainAndPredict(nPointsPar, nDimension, labelsPar, transposedDataPar, querynPoints, queryLabels, queryData, costArray, gammaArray);
	}

	float accuracy = 0;
	for(i = 0; i < folder; i++)
		accuracy += results[i] * dataPartitionSize[i];

	accuracy /= nPoints;

	free(dataPartitionSize);
	for(i = 0; i < folder; i++) {
		free(dataPartitionArray[i]);
		free(dataTranPartitionArray[i]);
	}
	free(dataPartitionArray);
	free(dataTranPartitionArray);
	free(transposedDataPar);
	free(labelsPar);
	free(queryLabels);
	free(results);

	return accuracy;
}

int main(int argc, char * argv[]) {
	int i;

	if(argc != 3) {
		cout << "Usage: example_bin2 train_file test_file" << endl;
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

    float* queryData;
    float* queryLabels;
    int querynPoints;
    int queryDimension;
    
    // Read in query data
    readSvm(queryFileChar, &queryData, &queryLabels, &querynPoints, &queryDimension);

    // category stores all classes for the training set
	vector<float> category;
	for(i = 0; i < nPoints; i++) {
		if(find(category.begin(), category.end(), labels[i]) == category.end()) {
			category.push_back(labels[i]);
		}
	}

	int numSVM = category.size() * (category.size() - 1) / 2;
	

	float* costArray = (float*)malloc(sizeof(float)*numSVM);
	float* gammaArray = (float*)malloc(sizeof(float)*numSVM);

	//1. cross validation on parameter set (cost, gamma), in this example we use RBF kernel, modify this to cross validation on different parameter sets
    float cost = 1.0;
    float gamma = 1.0/nDimension;
    for(i = 0; i < numSVM; i++) {
		costArray[i] = cost;
		gammaArray[i] = gamma;
	}

    int folder = 5;
    float cvAccuracy = multiClassCrossValidation(folder, costArray, gammaArray, nPoints, nDimension, labels, transposedData);
    cout << "Cross validation accuracy: " << cvAccuracy << "%" << endl;
    
    //2. Train and prediciton, in this example we use RBF kernel, using default parameter
	for(i = 0; i < numSVM; i++) {
		costArray[i] = 1.0;
		gammaArray[i] = 1.0/nDimension;
	}
    float accuracy = multiClassTrainAndPredict(nPoints, nDimension, labels, transposedData, querynPoints, queryLabels, queryData, costArray, gammaArray, true);
   	
    cout << "Cross validation accuracy: " << cvAccuracy << "%" << endl;
   	cout << "Accuracy: " << accuracy << "%" << endl;

    free(costArray);
    free(gammaArray);

	cout << "EXIT." << endl;
}

