/*
 * grid.cu
 *
 *  Add on: Dec 8, 2016
 *  Add by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  interfaces for finding the optimal parameters for for single SVM (work for both binary and multi-class case)
 */

#include "../include/svmTrain.h"
#include "../include/svmCommon.h"
#include "../include/svmClassify.h"
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>

// include code for data setup
#include "./common/transposeData.h"	
#include "./common/setupData.h"

using namespace std;

void printModel(const char* outputFile, Kernel_params kp, float* alpha, float* labels, float* data, int nPoints, int nDimension, float epsilon);

/**
 * Grid_MC_One selects best parameter set for multi-class SVM via one against one cross validation
 * @param nPoints the number of points in the original data
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the row major store of the original data
 * @param labels store the labels for the orignal data
 * @param folder specifies the folder for cross validation
 * @param kernelType the kernel type for trainging (eg. LINEAR, GAUSSIAN)
 * @param inCost sets the range of the cost parameter for testing, inCost[0] is the start value, inCost[1] is the end value and inCost[2] is the addend. (For example, inCost=[1, 5, 2] will test the value of cost from the set {1, 3, 5})
 * @param inGamma sets the range of the gamma parameter for testing, inGamma[0] is the start value, inGamma[1] is the end value and inGamma[2] is the multiplier. (For example, inGamma=[0.1, 1, 2] will test the value of gamma from the set {0.1, 0.2, 0.4, 0.8})
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 */
void grid_MC_One(int nPoints, int nDimension, float* transposedData, float* labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE) {
	int i, j;

	// set default value for inCost
	bool isCostNull = false;
	if(inCost == NULL) {
		isCostNull = true;
		inCost = (float*)malloc(sizeof(float)*3);
		inCost[0] = 1.0f;
		inCost[1] = 15.0f;
		inCost[2] = 2.0f;
	}

	// set default value for inGamma
	bool isGammaNull = false;
	if(kernelType == GAUSSIAN && inGamma == NULL) {
		isGammaNull = true;
		inGamma = (float*)malloc(sizeof(float)*3);
		inGamma[0] = 0.050f;
		inGamma[1] = 4.0f;
		inGamma[2] = 2.0f;
	}

	// setup parameters for training and cross validation
	int numCost = 0;
	int numGamma = 0;
	while(inCost[0] + inCost[2]*numCost <= inCost[1])
		numCost ++;
	int numParam = numCost;
	if(kernelType == GAUSSIAN) {
		float tmp = inGamma[0];
		while(tmp <= inGamma[1]) {
			tmp *= inGamma[2];
			numGamma ++;
		}
		numParam *= numGamma;
	}

	// find the classes of the data set: category stores all the classes
	vector<float> category;
	for(i = 0; i < nPoints; i++) {
		if(find(category.begin(), category.end(), labels[i]) == category.end()) {
			category.push_back(labels[i]);
		}
	}
	int numClass = category.size();

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
	// assert: for k-folder cross validation, training data contains (k-1)-folder and testing data contains 1-folder
	// make sure that there are numClass in the (k-1)-folder training data
	setupDataCV(folder, nPoints, nDimension, transposedData, permutation, dataPartitionArray, dataTranPartitionArray);

	// setup parameters for cross validation
	int numPair = numClass*(numClass-1)/2;
	int numSVM = numParam * folder * numPair;

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

	float** testDataArray = (float**)malloc(sizeof(float*)*numSVM);
	int* testNPointsArray = (int*)malloc(sizeof(int)*numSVM);
	float** testLabelsArray = (float**)malloc(sizeof(float*)*numSVM);

	Kernel_params kp;

	int nPointsPar;
	float* transposedDataPar = (float*)malloc(sizeof(float)*(nPoints-nPoints/folder)*nDimension);
	float* labelsPar = (float*)malloc(sizeof(float)*(nPoints-nPoints/folder));
	int querynPoints;
	float* queryData;
	float** queryLabelsPar = (float**)malloc(sizeof(float*)*folder);

	// classDist stores the class distribution of the data set
	int* classDist = (int*)malloc(sizeof(int)*numClass);

	int* nPointsParArray = (int*)malloc(sizeof(int)*numPair);
	for(i = 0; i < folder; i++) {
		queryLabelsPar[i] = (float*)malloc(sizeof(float)*(nPoints/folder+1));

		nPointsPar = nPoints - dataPartitionSize[i];
		querynPoints = dataPartitionSize[i];

		int qLabelIdx = 0;
		int dLabelIdx = 0;
		for(j = 0; j < folder; j++) {
			int p = j;
			while(p < nPoints) {
				if(j == i) {
					queryLabelsPar[i][qLabelIdx] = labels[permutation[p]];
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

		vector<float> categoryPar;
		for(j = 0; j < nPointsPar; j++) {
			if(find(categoryPar.begin(), categoryPar.end(), labelsPar[j]) == categoryPar.end()) {
				categoryPar.push_back(labelsPar[j]);
			}
		}
		if(categoryPar.size() < numClass) {
			cout << "Assertion fail: (folder-1) paritition do not contain all classes!" << endl; 
			return ;
		}

  		for(j = 0; j < numClass; j++)
  			classDist[j] = 0;
		for(j = 0; j < nPointsPar; j++) {
			int idx = find(category.begin(), category.end(), labelsPar[j]) - category.begin();
			classDist[idx] ++;
		}

		int classA = 0;
  		int classB = 1;
		for(j = 0; j < numPair; j++) {
			int svmId = i*numPair + j;
			nPointsParArray[j] = classDist[classA] + classDist[classB];
			nPointsArray[svmId] = nPointsParArray[j];

    		nDimensionArray[svmId] = nDimension;

    		labelsArray[svmId] = (float*) malloc(sizeof(float) * nPointsArray[svmId]);

    		int p;
    		// set labelsArray for one-against-one cross valiation
	    	for(p = 0; p < nPointsArray[svmId]; p++) {
	    		if(p < classDist[classA])
	    			labelsArray[svmId][p] = 1;
	    		else
	    			labelsArray[svmId][p] = -1;
	    	}

	    	if(classB == numClass-1) {
	    		classA += 1;
	    		classB = classA + 1;
	    	} else 
	    		classB ++;

	    	dataArray[svmId] = (float*) malloc(sizeof(float) * nPointsArray[svmId] * nDimension);
    		transposedDataArray[svmId] = (float*) malloc(sizeof(float) * nPointsArray[svmId] * nDimension);

    		costArray[svmId] = inCost[0];
		    toleranceArray[svmId] = tolerance;
		    epsilonArray[svmId] = epsilon;
		    heuristicMethodArray[svmId] = heuristicMethod;

		    kpArray[svmId] = (Kernel_params *) malloc(sizeof(Kernel_params));
		    if(kernelType == GAUSSIAN) {
				kp.kernel_type = "rbf";
				kp.gamma = inGamma[0];
			} else {
				kp.kernel_type = "linear";
			}
			memcpy(kpArray[svmId], &kp, sizeof(Kernel_params));

			testDataArray[svmId] = queryData;
    		testNPointsArray[svmId] = querynPoints;
    		testLabelsArray[svmId] = queryLabelsPar[i];
		}

		setupData(numPair, category, classDist, nPointsPar, nDimension, transposedDataPar, labelsPar, nPointsParArray, dataArray+i*numPair, transposedDataArray+i*numPair);
	
		for(j = 0; j < numPair; j++) {
			int p;
			int svmIdInFolder = i*numPair + j;
			for(p = 1; p < numParam; p++) {
				int svmId = (p*folder+i)*numPair + j;
				nPointsArray[svmId] = nPointsArray[svmIdInFolder];
				nDimensionArray[svmId] = nDimension;
				labelsArray[svmId] = labelsArray[svmIdInFolder];
				dataArray[svmId] = dataArray[svmIdInFolder];
				transposedDataArray[svmId] = transposedDataArray[svmIdInFolder];

				toleranceArray[svmId] = tolerance;
			    epsilonArray[svmId] = epsilon;
			    heuristicMethodArray[svmId] = heuristicMethod;

			    kpArray[svmId] = (Kernel_params *) malloc(sizeof(Kernel_params));
			    if(kernelType == GAUSSIAN) {
					int costId = p / numGamma;
					int gammaId = p - numGamma * costId;

					costArray[svmId] = inCost[0] + costId*inCost[2];
					kp.kernel_type = "rbf";
					kp.gamma = inGamma[0];
					int q;
					for(q = 0; q < gammaId; q++)
						kp.gamma *= inGamma[2];
				} else {
					costArray[svmId] = inCost[0] + p*inCost[2];
					kp.kernel_type = "linear";
				}
				memcpy(kpArray[svmId], &kp, sizeof(Kernel_params));

			    testDataArray[svmId] = testDataArray[svmIdInFolder];
	    		testNPointsArray[svmId] = testNPointsArray[svmIdInFolder];
	    		testLabelsArray[svmId] = testLabelsArray[svmIdInFolder];
			}
		}
	}

	cout << "Finish Setting Parameters!" << endl;

	float** results;
	svmPredict(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, testDataArray, testNPointsArray, testLabelsArray, NULL, 1, true, &results);

	float* accuracyResult = (float*)malloc(sizeof(float)*numParam);
	// mcvClassification stores the classification for each data point in cross validation
	float* mcvClassification = (float*)malloc(sizeof(float)*nPoints);
	int* vote = (int*)malloc(sizeof(int)*numClass);
	int* dataPartitionAccuSize = (int*)malloc(sizeof(int)*folder);
	dataPartitionAccuSize[0] = 0;
	for(i = 1; i < folder; i++)
		dataPartitionAccuSize[i] = dataPartitionAccuSize[i-1] + dataPartitionSize[i-1];

	for(i = 0; i < numParam; i++) {
		for(j = 0; j < folder; j++) {
			int p;
			for(p = 0; p < dataPartitionSize[j]; p++) {
		   		int q;
		   		for(q = 0; q < numClass; q++)
		   			vote[q] = 0;
		   		for(q = 0; q < numPair; q++) {
		   			int classA = 0;
		   			while(q >= (classA+1)*(numClass-1)-(classA+1)*classA/2)
		   				classA ++;
		   			int classB = q - classA*(numClass-1) + classA*(classA-1)/2 + classA + 1;

		   			if(results[(i*folder+j)*numPair+q][p] > 0)
		   				vote[classA] ++;
		   			else
		   				vote[classB] ++;
		      	}

		   		int max = vote[0];
		   		mcvClassification[dataPartitionAccuSize[j]+p] = category[0];
		   		for(q = 0; q < numClass; q++) {
		   			if(vote[q] > max) {
		   				max = vote[q];
		   				mcvClassification[dataPartitionAccuSize[j]+p] = category[q];
		   			}
		   		}
		   	}
		}

		// calculate accuracy
	   	int correct = 0;
	   	for(j = 0; j < folder; j++) {
	   		int p;
	   		for(p = 0; p < dataPartitionSize[j]; p++) {
	   			if(mcvClassification[dataPartitionAccuSize[j]+p] == queryLabelsPar[j][p]) {
		   			correct ++;
		   		}
	   		}
	   	}
	   	accuracyResult[i] = (100 * (float) correct ) / nPoints;
	}

	cout << "Accuracy results for different parameter set: " << endl;
	if(kernelType == GAUSSIAN) {
		for(i = 0; i < numCost; i++) {
			for(j = 0; j < numGamma; j++) {
				float gamma = inGamma[0];
				int p;
				for(p = 0; p < j; p++)
					gamma *= inGamma[2];
				cout << "(" << inCost[0]+i*inCost[2] << ", " << gamma << ", " << accuracyResult[i*numGamma+j] << ") "; 
			}
		}
	} else {
		for(i = 0; i < numCost; i++) {
			cout << "(" << inCost[0]+i*inCost[2] << ", " << accuracyResult[i] << ") ";
		}
	}
	cout << endl;
	
	float bestAccuracy = accuracyResult[0];
	int bestParamId = 0;
	for(i = 1; i < numParam; i++) {
		if(accuracyResult[i] > bestAccuracy) {
			bestAccuracy = accuracyResult[i];
			bestParamId = i;
		}
	}

	float bestCost = 0.0;
	float bestGamma = 0.0;
	if(kernelType == GAUSSIAN) {
		int costId = bestParamId / numGamma;
		int gammaId = bestParamId - numGamma * costId;

		bestCost = inCost[0] + costId*inCost[2];
		bestGamma = inGamma[0];
		for(j = 0; j < gammaId; j++)
			bestGamma *= inGamma[2];
		cout << "RBF Kernel: Find the best parameter set (cost, gamma, accuracy): (" << bestCost << ", " << bestGamma << ", " << accuracyResult[bestParamId] << ")." << endl;
	} else {
		bestCost = inCost[0] + bestParamId*inCost[2];
		cout << "Linear Kernel: find the best parameter (cost, accuracy): " << "(" << bestCost << ", " << accuracyResult[bestParamId] << ")." << endl;
	}

	// free pointers
	if(isCostNull)
		free(inCost);
	if(isGammaNull)
		free(inGamma);
	free(permutation);
	free(dataPartitionSize);
	for(i = 0; i < folder; i++) {
		free(dataPartitionArray[i]);
		free(dataTranPartitionArray[i]);
		free(queryLabelsPar[i]);
	}
	free(dataPartitionArray);
	free(dataTranPartitionArray);
	free(queryLabelsPar);
	for(i = 0; i < folder; i++) {
		for(j = 0; j < numPair; j++) {
			free(dataArray[i*numPair+j]);
			free(transposedDataArray[i*numPair+j]);
			free(labelsArray[i*numPair+j]);
		}
	}
	for(i = 0; i < numSVM; i++) {
		free(kpArray[i]);
	}
	free(labelsArray);
	free(dataArray);
	free(transposedDataArray);
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

	free(transposedDataPar);
	free(labelsPar);

	free(classDist);
	free(nPointsParArray);

	free(accuracyResult);
	free(mcvClassification);
	free(vote);
	free(dataPartitionAccuSize);
}

/**
 * Grid_MC_All selects best parameter set for multi-class SVM via one against all cross validation
 * @param nPoints the number of points in the original data
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the row major store of the original data
 * @param labels store the labels for the orignal data
 * @param data the column major store of the original data
 * @param folder specifies the folder for cross validation
 * @param kernelType the kernel type for trainging (eg. LINEAR, GAUSSIAN)
 * @param inCost sets the range of the cost parameter for testing, inCost[0] is the start value, inCost[1] is the end value and inCost[2] is the addend. (For example, inCost=[1, 5, 2] will test the value of cost from the set {1, 3, 5})
 * @param inGamma sets the range of the gamma parameter for testing, inGamma[0] is the start value, inGamma[1] is the end value and inGamma[2] is the multiplier. (For example, inGamma=[0.1, 1, 2] will test the value of gamma from the set {0.1, 0.2, 0.4, 0.8})
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 */
void grid_MC_All(int nPoints, int nDimension, float* transposedData, float* labels, float* data = NULL, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE) {
	int i, j;

	// set default value for inCost
	bool isCostNull = false;
	if(inCost == NULL) {
		isCostNull = true;
		inCost = (float*)malloc(sizeof(float)*3);
		inCost[0] = 1.0f;
		inCost[1] = 15.0f;
		inCost[2] = 2.0f;
	}

	// set default value for inGamma
	bool isGammaNull = false;
	if(kernelType == GAUSSIAN && inGamma == NULL) {
		isGammaNull = true;
		inGamma = (float*)malloc(sizeof(float)*3);
		inGamma[0] = 0.050f;
		inGamma[1] = 4.0f;
		inGamma[2] = 2.0f;
	}

	// find the classes of the data set: category stores all the classes
	vector<float> category;
	for(i = 0; i < nPoints; i++) {
		if(find(category.begin(), category.end(), labels[i]) == category.end()) {
			category.push_back(labels[i]);
		}
	}
	int numClass = category.size();
	
	// set labelsPerClass for each class for one-against-all cross valiation
	float** labelsPerClass = (float**)malloc(sizeof(float*)*numClass);
	for(i = 0; i < numClass; i++) {
		labelsPerClass[i] = (float*) malloc(sizeof(float)*nPoints);
    	for(j = 0; j < nPoints; j++) {
    		if(labels[j] == category[i])
    			labelsPerClass[i][j] = 1.0;
    		else
    			labelsPerClass[i][j] = -1.0;
    	}
	}

	// setup parameters for training and cross validation
	int numCost = 0;
	int numGamma = 0;
	while(inCost[0] + inCost[2]*numCost <= inCost[1])
		numCost ++;
	int numParam = numCost;
	if(kernelType == GAUSSIAN) {
		float tmp = inGamma[0];
		while(tmp <= inGamma[1]) {
			tmp *= inGamma[2];
			numGamma ++;
		}
		numParam *= numGamma;
	}

	int numSVM = numClass * numParam;

	float** labelsArray = (float**)malloc(sizeof(float*)*numSVM);
	float** transposedDataArray = (float**)malloc(sizeof(float*)*numSVM);
	int* nPointsArray = (int*)malloc(sizeof(int)*numSVM);
	int* nDimensionArray = (int*)malloc(sizeof(int)*numSVM);
	Kernel_params** kpArray = (Kernel_params**)malloc(sizeof(Kernel_params*)*numSVM);
	float* costArray = (float*)malloc(sizeof(float)*numSVM);
	SelectionHeuristic* heuristicMethodArray = (SelectionHeuristic*)malloc(sizeof(SelectionHeuristic)*numSVM);
	float* epsilonArray = (float*)malloc(sizeof(float)*numSVM);
	float* toleranceArray = (float*)malloc(sizeof(float)*numSVM);

	Kernel_params kp;
	for(i = 0; i < numParam; i++) {
		for(j = 0; j < numClass; j++) {
			labelsArray[i*numClass+j] = labelsPerClass[j];
			transposedDataArray[i*numClass+j] = transposedData;
			nPointsArray[i*numClass+j] = nPoints;
			nDimensionArray[i*numClass+j] = nDimension;
			heuristicMethodArray[i*numClass+j] = heuristicMethod;
			epsilonArray[i*numClass+j] = epsilon;
			toleranceArray[i*numClass+j] = tolerance;

			kpArray[i*numClass+j] = (Kernel_params *) malloc(sizeof(Kernel_params));
			if(kernelType == GAUSSIAN) {
				int costId = i / numGamma;
				int gammaId = i - numGamma * costId;

				costArray[i*numClass+j] = inCost[0] + costId*inCost[2];
				kp.kernel_type = "rbf";
				kp.gamma = inGamma[0];
				int p;
				for(p = 0; p < gammaId; p++)
					kp.gamma *= inGamma[2];
			} else {
				costArray[i*numClass+j] = inCost[0] + i*inCost[2];
				kp.kernel_type = "linear";
			}
			memcpy(kpArray[i*numClass+j], &kp, sizeof(Kernel_params));
		}
	}
	
	// 1. find the best parameter for SVM
	// mcvResult stores the result of multi-class cross-validation
    float ** mcvResult;
	svmCrossValidation(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, 1, true, &mcvResult);

	int* numTestingPointsEachFolder = (int*) malloc(sizeof(int) * folder);
	for(i = 0; i < folder; i++) {
		numTestingPointsEachFolder[i] = nPoints / folder;
		if(i < nPoints % folder)
			numTestingPointsEachFolder[i] += 1;
	}

	float* accuracyResult = (float*)malloc(sizeof(float)*numParam);
	// mcvClassification stores the classification for each data point in cross validation
	float* mcvClassification = (float*) malloc(sizeof(float) * nPoints);

	// find the accuracy result for each paramter set
	for(i = 0; i < numParam; i++) {
		for(j = 0; j < folder; j++) {
			int p;
			for(p = 0; p < numTestingPointsEachFolder[j]; p++) {
				float max = mcvResult[i*numClass*folder+j][p];
				mcvClassification[j+p*folder] = category[0];
				int q;
				for(q = 0; q < numClass; q++) {
					if(mcvResult[i*numClass*folder+j+q*folder][p] > max) {
						max = mcvResult[i*numClass*folder+j+q*folder][p];
						mcvClassification[j+p*folder] = category[q];
					}
				}
			}
		}

		// calculate cross validation accuracy for each parameter set
		int corrLabel = 0;
		for(j = 0; j < nPoints; j++) {
			if(mcvClassification[j] == labels[j]) {
				corrLabel ++;
			}
		}

		accuracyResult[i] = (100.0 * (float) corrLabel) / nPoints;
	}

	cout << "Accuracy results for different parameter set: " << endl;
	if(kernelType == GAUSSIAN) {
		for(i = 0; i < numCost; i++) {
			for(j = 0; j < numGamma; j++) {
				float gamma = inGamma[0];
				int p;
				for(p = 0; p < j; p++)
					gamma *= inGamma[2];
				cout << "(" << inCost[0]+i*inCost[2] << ", " << gamma << ", " << accuracyResult[i*numGamma+j] << ") "; 
			}
		}
	} else {
		for(i = 0; i < numCost; i++) {
			cout << "(" << inCost[0]+i*inCost[2] << ", " << accuracyResult[i] << ") ";
		}
	}
	cout << endl;
	
	float bestAccuracy = accuracyResult[0];
	int bestParamId = 0;
	for(i = 1; i < numParam; i++) {
		if(accuracyResult[i] > bestAccuracy) {
			bestAccuracy = accuracyResult[i];
			bestParamId = i;
		}
	}

	float bestCost = 0.0;
	float bestGamma = 0.0;
	if(kernelType == GAUSSIAN) {
		int costId = bestParamId / numGamma;
		int gammaId = bestParamId - numGamma * costId;

		bestCost = inCost[0] + costId*inCost[2];
		bestGamma = inGamma[0];
		for(j = 0; j < gammaId; j++)
			bestGamma *= inGamma[2];
		cout << "RBF Kernel: Find the best parameter set (cost, gamma, accuracy): (" << bestCost << ", " << bestGamma << ", " << accuracyResult[bestParamId] << ")." << endl;
	} else {
		bestCost = inCost[0] + bestParamId*inCost[2];
		cout << "Linear Kernel: find the best parameter (cost, accuracy): " << "(" << bestCost << ", " << accuracyResult[bestParamId] << ")." << endl;
	}

	// free pointer
	for(i = 0; i < numSVM; i++) {
		free(kpArray[i]);
	}
	free(numTestingPointsEachFolder);
	free(labelsArray);
	free(accuracyResult);
	free(mcvClassification);


	// 2. Using the best parameter to train a SVM model
	bool isDataNull = false;
	if(data == NULL) {
		isDataNull = true;
		data = (float*) malloc(sizeof(float)*nPoints*nDimension);
		transposeData(nPoints, nDimension, transposedData, data);
	}

	numSVM = numClass;

	float** dataArray = (float**)malloc(sizeof(float*)*numSVM);
	transposedDataArray = (float**)realloc(transposedDataArray, sizeof(float*)*numSVM);
	nPointsArray = (int*)realloc(nPointsArray, sizeof(int)*numSVM);
	nDimensionArray = (int*)realloc(nDimensionArray, sizeof(int)*numSVM);
	kpArray = (Kernel_params**)realloc(kpArray, sizeof(Kernel_params*)*numSVM);
	costArray = (float*)realloc(costArray, sizeof(float)*numSVM);
	heuristicMethodArray = (SelectionHeuristic*)realloc(heuristicMethodArray, sizeof(SelectionHeuristic)*numSVM);
	epsilonArray = (float*)realloc(epsilonArray, sizeof(float)*numSVM);
	toleranceArray = (float*)realloc(toleranceArray, sizeof(float)*numSVM);

	for(i = 0; i < numSVM; i++) {
		dataArray[i] = data;
		transposedDataArray[i] = transposedData;
		nPointsArray[i] = nPoints;
		nDimensionArray[i] = nDimension;
		heuristicMethodArray[i] = heuristicMethod;
		epsilonArray[i] = epsilon;
		toleranceArray[i] = tolerance;

		kpArray[i] = (Kernel_params *) malloc(sizeof(Kernel_params));
		if(kernelType == GAUSSIAN) {
			costArray[i] = bestCost;
			kp.kernel_type = "rbf";
			kp.gamma = bestGamma;
		} else {
			costArray[i] = bestCost;
			kp.kernel_type = "linear";
		}
		memcpy(kpArray[i], &kp, sizeof(Kernel_params));
	}

	svmTrain(numSVM, dataArray, nPointsArray, nDimensionArray, labelsPerClass, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, NULL, 0.5);

	// free pointer
	if(isCostNull)
		free(inCost);
	if(isGammaNull)
		free(inGamma);
	if(isDataNull)
		free(data);
	for(i = 0; i < numSVM; i++) {
		free(labelsPerClass[i]);
		free(kpArray[i]);
	}
	free(nPointsArray);
	free(nDimensionArray);
	free(dataArray);
	free(labelsPerClass);
	free(kpArray);
	free(costArray);
	free(heuristicMethodArray);
	free(epsilonArray);
	free(toleranceArray);
	free(transposedDataArray);
}

/**
 * Grid selects best parameter set for SVM via cross validation
 * @param nPoints the number of points in the original data
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the row major store of the original data
 * @param labels store the labels for the orignal data
 * @param data the column major store of original data 
 * @param folder specifies the folder for cross validation
 * @param kernelType the kernel type for trainging (eg. LINEAR, GAUSSIAN)
 * @param inCost sets the range of the cost parameter for testing, inCost[0] is the start value, inCost[1] is the end value and inCost[2] is the addend. (For example, inCost=[1, 5, 2] will test the value of cost from the set {1, 3, 5})
 * @param inGamma sets the range of the gamma parameter for testing, inGamma[0] is the start value, inGamma[1] is the end value and inGamma[2] is the multiplier. (For example, inGamma=[0.1, 1, 2] will test the value of gamma from the set {0.1, 0.2, 0.4, 0.8})
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 */
void grid(int nPoints, int nDimension, float* transposedData, float* labels, float* data = NULL, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE) {
	int i, j;

	// set default value for inCost
	bool isCostNull = false;
	if(inCost == NULL) {
		isCostNull = true;
		inCost = (float*)malloc(sizeof(float)*3);
		inCost[0] = 1.0f;
		inCost[1] = 15.0f;
		inCost[2] = 2.0f;
	}

	// set default value for inGamma
	bool isGammaNull = false;
	if(kernelType == GAUSSIAN && inGamma == NULL) {
		isGammaNull = true;
		inGamma = (float*)malloc(sizeof(float)*3);
		inGamma[0] = 0.050f;
		inGamma[1] = 4.0f;
		inGamma[2] = 2.0f;
	}

	// setup parameters for training and cross validation
	int numCost = 0;
	int numGamma = 0;
	while(inCost[0] + inCost[2]*numCost <= inCost[1])
		numCost ++;
	int numSVM = numCost;
	if(kernelType == GAUSSIAN) {
		float tmp = inGamma[0];
		while(tmp <= inGamma[1]) {
			tmp *= inGamma[2];
			numGamma ++;
		}
		numSVM *= numGamma;
	}

	float** labelsArray = (float**)malloc(sizeof(float*)*numSVM);
	float** transposedDataArray = (float**)malloc(sizeof(float*)*numSVM);
	int* nPointsArray = (int*)malloc(sizeof(int)*numSVM);
	int* nDimensionArray = (int*)malloc(sizeof(int)*numSVM);
	Kernel_params** kpArray = (Kernel_params**)malloc(sizeof(Kernel_params*)*numSVM);
	float* costArray = (float*)malloc(sizeof(float)*numSVM);
	SelectionHeuristic* heuristicMethodArray = (SelectionHeuristic*)malloc(sizeof(SelectionHeuristic)*numSVM);
	float* epsilonArray = (float*)malloc(sizeof(float)*numSVM);
	float* toleranceArray = (float*)malloc(sizeof(float)*numSVM);

	Kernel_params kp;
	for(i = 0; i < numSVM; i++) {
		labelsArray[i] = labels;
		transposedDataArray[i] = transposedData;
		nPointsArray[i] = nPoints;
		nDimensionArray[i] = nDimension;
		heuristicMethodArray[i] = heuristicMethod;
		epsilonArray[i] = epsilon;
		toleranceArray[i] = tolerance;

		kpArray[i] = (Kernel_params *) malloc(sizeof(Kernel_params));
		if(kernelType == GAUSSIAN) {
			int costId = i / numGamma;
			int gammaId = i - numGamma * costId;

			costArray[i] = inCost[0] + costId*inCost[2];
			kp.kernel_type = "rbf";
			kp.gamma = inGamma[0];
			for(j = 0; j < gammaId; j++)
				kp.gamma *= inGamma[2];
		} else {
			costArray[i] = inCost[0] + i*inCost[2];
			kp.kernel_type = "linear";
		}
		memcpy(kpArray[i], &kp, sizeof(Kernel_params));
	}

	// 1. find the best parameter for SVM
	float* accuracyResult = (float*)malloc(sizeof(float)*numSVM);
	svmCrossValidation(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, 0.5, false, NULL, true, accuracyResult);

	cout << "Accuracy results for different parameter set: " << endl;
	if(kernelType == GAUSSIAN) {
		for(i = 0; i < numCost; i++) {
			for(j = 0; j < numGamma; j++) {
				float gamma = inGamma[0];
				int p;
				for(p = 0; p < j; p++)
					gamma *= inGamma[2];
				cout << "(" << inCost[0]+i*inCost[2] << ", " << gamma << ", " << accuracyResult[i*numGamma+j] << ") "; 
			}
		}
	} else {
		for(i = 0; i < numCost; i++) {
			cout << "(" << inCost[0]+i*inCost[2] << ", " << accuracyResult[i] << ") ";
		}
	}
	cout << endl;
	
	float bestAccuracy = accuracyResult[0];
	int bestParamId = 0;
	for(i = 1; i < numSVM; i++) {
		if(accuracyResult[i] > bestAccuracy) {
			bestAccuracy = accuracyResult[i];
			bestParamId = i;
		}
	}

	float bestCost = 0.0;
	float bestGamma = 0.0;
	if(kernelType == GAUSSIAN) {
		int costId = bestParamId / numGamma;
		int gammaId = bestParamId - numGamma * costId;

		bestCost = inCost[0] + costId*inCost[2];
		bestGamma = inGamma[0];
		for(j = 0; j < gammaId; j++)
			bestGamma *= inGamma[2];
		cout << "RBF Kernel: Find the best parameter set (cost, gamma, accuracy): (" << bestCost << ", " << bestGamma << ", " << accuracyResult[bestParamId] << ")." << endl;
	} else {
		bestCost = inCost[0] + bestParamId*inCost[2];
		cout << "Linear Kernel: find the best parameter (cost, accuracy): " << "(" << bestCost << ", " << accuracyResult[bestParamId] << ")." << endl;
	}


	// 2. Using the best parameter to train a SVM model
	bool isDataNull = false;
	if(data == NULL) {
		isDataNull = true;
		data = (float*) malloc(sizeof(float)*nPoints*nDimension);
		transposeData(nPoints, nDimension, transposedData, data);
	}
		
	float* alpha;
    if(kernelType == GAUSSIAN) {
    	kp.kernel_type = "rbf";
		kp.gamma = bestGamma;
    } else {
    	kp.kernel_type = "linear";
    }
	performTraining(data, nPoints, nDimension, labels, &alpha, &kp, bestCost, heuristicMethod, epsilon, tolerance, transposedData);

	// output the model
	char outputFilename[] = "svm0.mdl";
	printModel(outputFilename, kp, alpha, labels, data, nPoints, nDimension, epsilon);

	// free pointers
	if(isCostNull)
		free(inCost);
	if(isGammaNull)
		free(inGamma);
	if(isDataNull)
		free(data);
	for(i = 0; i < numSVM; i++)
		free(kpArray[i]);
	free(nPointsArray);
	free(nDimensionArray);
	free(labelsArray);
	free(kpArray);
	free(costArray);
	free(heuristicMethodArray);
	free(epsilonArray);
	free(toleranceArray);
	free(transposedDataArray);
	free(accuracyResult);
}