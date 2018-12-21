/*
 * multi_grid.cu
 *
 *  Add on: Dec 27, 2016
 *  Add by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  interfaces for finding the optimal parameters for mutliple SVMs (work for both binary and multi-class case)
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
#include "./common/setupData.h"

using namespace std;

/**
 * MSVM_Grid selects best parameter set for multiple SVMs via cross validation
 * @param nSVM the number of SVMs for finding best parameter set
 * @param nPoints the number of points for each SVM
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the array of row major stores of the original multi SVM data
 * @param labels store the labels for the orignal multi SVM data
 * @param folder specifies the folder for cross validation
 * @param kernelType the kernel type for trainging (eg. LINEAR, GAUSSIAN)
 * @param inCost sets the range of the cost parameter for testing, inCost[0] is the start value, inCost[1] is the end value and inCost[2] is the addend. (For example, inCost=[1, 5, 2] will test the value of cost from the set {1, 3, 5})
 * @param inGamma sets the range of the gamma parameter for testing, inGamma[0] is the start value, inGamma[1] is the end value and inGamma[2] is the multiplier. (For example, inGamma=[0.1, 1, 2] will test the value of gamma from the set {0.1, 0.2, 0.4, 0.8})
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 */
void msvm_grid(int nSVM, int* nPoints, int nDimension, float** transposedData, float** labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE) {
	int i, j, k;

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

	int numSVM = nSVM * numParam;

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
	for(i = 0; i < nSVM; i++) {
		for(j = 0; j < numParam; j++) {
			labelsArray[i*numParam+j] = labels[i];
			transposedDataArray[i*numParam+j] = transposedData[i];
			nPointsArray[i*numParam+j] = nPoints[i];
			nDimensionArray[i*numParam+j] = nDimension;
			heuristicMethodArray[i*numParam+j] = heuristicMethod;
			epsilonArray[i*numParam+j] = epsilon;
			toleranceArray[i*numParam+j] = tolerance;

			kpArray[i*numParam+j] = (Kernel_params *) malloc(sizeof(Kernel_params));
			if(kernelType == GAUSSIAN) {
				int costId = j / numGamma;
				int gammaId = j - numGamma * costId;

				costArray[i*numParam+j] = inCost[0] + costId*inCost[2];
				kp.kernel_type = "rbf";
				kp.gamma = inGamma[0];
				for(k = 0; k < gammaId; k++)
					kp.gamma *= inGamma[2];
			} else {
				costArray[i*numParam+j] = inCost[0] + j*inCost[2];
				kp.kernel_type = "linear";
			}
			memcpy(kpArray[i*numParam+j], &kp, sizeof(Kernel_params));
		}
	}

	// find the best parameter for SVMs
	float* accuracyResult = (float*)malloc(sizeof(float)*numSVM);
	svmCrossValidation(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, 0.5, false, NULL, true, accuracyResult);

	cout << "Accuracy results for different parameter set: " << endl;
	if(kernelType == GAUSSIAN) {
		for(i = 0; i < nSVM; i++) {
			cout << "svm " << i << ": " << endl;
			for(j = 0; j < numCost; j++) {
				for(k = 0; k < numGamma; k++) {
					float gamma = inGamma[0];
					int p;
					for(p = 0; p < k; p++)
						gamma *= inGamma[2];
					cout << "(" << inCost[0]+j*inCost[2] << ", " << gamma << ", " << accuracyResult[i*numParam+j*numGamma+k] << ") "; 
				}
			}
			cout << endl;
		}
	} else {
		for(i = 0; i < nSVM; i++) {
			cout << "svm " << i << ": " << endl;
			for(j = 0; j < numCost; j++) {
				cout << "(" << inCost[0]+j*inCost[2] << ", " << accuracyResult[i*numCost+j] << ") ";
			}
			cout << endl;
		}
	}
	cout << endl;
	
	float* bestAccuracy = (float*) malloc(sizeof(float)*nSVM);
	float* bestParamId = (float*) malloc(sizeof(float)*nSVM);
	for(i = 0; i < nSVM; i++) {
		bestAccuracy[i] = accuracyResult[i*numParam];
		bestParamId[i] = 0;
		for(j = 1; j < numParam; j++) {
			if(accuracyResult[i*numParam+j] > bestAccuracy[i]) {
				bestAccuracy[i] = accuracyResult[i*numParam+j];
				bestParamId[i] = j;
			}
		}
	}

	// if needed, can return (bestCost, bestGamma), however, remember to uncomment free()
	float* bestCost = NULL;
	float* bestGamma = NULL;

	if(kernelType == GAUSSIAN) {
		bestCost = (float*) malloc(sizeof(float)*nSVM);
		bestGamma = (float*) malloc(sizeof(float)*nSVM);
		for(i = 0; i < nSVM; i++) {
			int costId = bestParamId[i] / numGamma;
			int gammaId = bestParamId[i] - numGamma * costId;

			bestCost[i] = inCost[0] + costId*inCost[2];
			bestGamma[i] = inGamma[0];
			for(j = 0; j < gammaId; j++)
				bestGamma[i] *= inGamma[2];
			cout << "svm " << i << ": " << endl;
			cout << "RBF Kernel: Find the best parameter set (cost, gamma, accuracy): (" << bestCost[i] << ", " << bestGamma[i] << ", " << bestAccuracy[i] << ")." << endl;
		}
	} else {
		bestCost = (float*) malloc(sizeof(float)*nSVM);
		for(i = 0; i < nSVM; i++) {
			bestCost[i] = inCost[0] + bestParamId[i]*inCost[2];
			cout << "svm " << i << ": " << endl;
			cout << "Linear Kernel: find the best parameter (cost, accuracy): " << "(" << bestCost[i] << ", " << bestAccuracy[i] << ")." << endl;
		}
	}

	if(isCostNull)
		free(inCost);
	if(isGammaNull)
		free(inGamma);
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
	free(bestAccuracy);
	free(bestParamId);
	free(bestCost);
	free(bestGamma);
}

/**
 * MSVM_Grid_MC_All selects best parameter set for multiple multi-class SVMs via one-against-all cross validation
 * @param nSVM the number of SVMs for finding best parameter set
 * @param nPoints the number of points for each SVM
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the array of row major stores of the original multi SVM data
 * @param labels store the labels for the orignal multi SVM data
 * @param folder specifies the folder for cross validation
 * @param kernelType the kernel type for trainging (eg. LINEAR, GAUSSIAN)
 * @param inCost sets the range of the cost parameter for testing, inCost[0] is the start value, inCost[1] is the end value and inCost[2] is the addend. (For example, inCost=[1, 5, 2] will test the value of cost from the set {1, 3, 5})
 * @param inGamma sets the range of the gamma parameter for testing, inGamma[0] is the start value, inGamma[1] is the end value and inGamma[2] is the multiplier. (For example, inGamma=[0.1, 1, 2] will test the value of gamma from the set {0.1, 0.2, 0.4, 0.8})
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 */
void msvm_grid_MC_All(int nSVM, int* nPoints, int nDimension, float** transposedData, float** labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE) {
	int i, j, k;

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
	vector<vector<float> > category;
	for(i = 0; i < nSVM; i++) {
		vector<float> tmp;
		for(j = 0; j < nPoints[i]; j++) {
			if(find(tmp.begin(), tmp.end(), labels[i][j]) == tmp.end()) {
				tmp.push_back(labels[i][j]);
			}
		}
		category.push_back(tmp);
	}

	int* numClass = (int*) malloc(sizeof(int)*nSVM);
	int totalClass = 0;
	for(i = 0; i < nSVM; i++) {
		numClass[i] = category[i].size();
		totalClass += category[i].size();
	}

	int* accuClass = (int*) malloc(sizeof(int)*nSVM);
	accuClass[0] = 0;
	for(i = 1; i < nSVM; i++) {
		accuClass[i] = accuClass[i-1] + numClass[i-1];
	}

	// set labelsPerClass for each class for one-against-all cross valiation
	float** labelsPerClass = (float**)malloc(sizeof(float*)*totalClass);
	for(i = 0; i < nSVM; i++) {
		for(j = 0; j < numClass[i]; j++) {
			labelsPerClass[accuClass[i]+j] = (float*) malloc(sizeof(float)*nPoints[i]);
			for(k = 0; k < nPoints[i]; k++) {
				if(labels[i][k] == category[i][j])
					labelsPerClass[accuClass[i]+j][k] = 1.0;
				else
					labelsPerClass[accuClass[i]+j][k] = -1.0;
			}
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

	int numSVM = totalClass * numParam;

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
	for(i = 0; i < nSVM; i++) {
		for(j = 0; j < numParam; j++) {
			for(k = 0; k < numClass[i]; k++) {
				int svmId = accuClass[i]*numParam + j*numClass[i] + k;
				labelsArray[svmId] = labelsPerClass[accuClass[i]+k];
				transposedDataArray[svmId] = transposedData[i];
				nPointsArray[svmId] = nPoints[i];
				nDimensionArray[svmId] = nDimension;
				heuristicMethodArray[svmId] = heuristicMethod;
				epsilonArray[svmId] = epsilon;
				toleranceArray[svmId] = tolerance;

				kpArray[svmId] = (Kernel_params *) malloc(sizeof(Kernel_params));
				if(kernelType == GAUSSIAN) {
					int costId = j / numGamma;
					int gammaId = j - numGamma * costId;

					costArray[svmId] = inCost[0] + costId*inCost[2];
					kp.kernel_type = "rbf";
					kp.gamma = inGamma[0];
					int p;
					for(p = 0; p < gammaId; p++)
						kp.gamma *= inGamma[2];
				} else {
					costArray[svmId] = inCost[0] + i*inCost[2];
					kp.kernel_type = "linear";
				}
				memcpy(kpArray[svmId], &kp, sizeof(Kernel_params));
			}
		}
	}

	// find the best parameter for SVM
	// mcvResult stores the result of multi-class cross-validation
    float** mcvResult;
	svmCrossValidation(numSVM, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, folder, 1, true, &mcvResult);

	int* numTestingPointsEachFolder = (int*) malloc(sizeof(int) * folder * nSVM);
	for(i = 0; i < nSVM; i++) {
		for(j = 0; j < folder; j++) {
			numTestingPointsEachFolder[i*folder+j] = nPoints[i] / folder;
			if(j < nPoints[i] % folder)
				numTestingPointsEachFolder[i*folder+j] += 1;
		}
	}

	float* accuracyResult = (float*)malloc(sizeof(float)*numParam*nSVM);
	int maxPoints = 0;
	for(i = 0; i < nSVM; i++) {
		if(nPoints[i] > maxPoints) {
			maxPoints = nPoints[i];
		}
	}
	// mcvClassification stores the classification for each data point in cross validation
	float* mcvClassification = (float*) malloc(sizeof(float) * maxPoints);

	// find the accuracy result for each paramter set, each SVM
	for(i = 0; i < nSVM; i++) {
		for(j = 0; j < numParam; j++) {
			for(k = 0; k < folder; k++) {
				int p;
				for(p = 0; p < numTestingPointsEachFolder[i*folder+k]; p++) {
					int svmId = i*numParam*accuClass[i]*folder+j*numClass[i]*folder+k;
					float max = mcvResult[svmId][p];
					mcvClassification[k+p*folder] = category[i][0];
					int q;
					for(q = 0; q < numClass[i]; q++) {
						if(mcvResult[svmId+q*folder][p] > max) {
							max = mcvResult[svmId+q*folder][p];
							mcvClassification[k+p*folder] = category[i][q];
						}
					}
				}
			}

			// calculate cross validation accuracy for each parameter set
			int corrLabel = 0;
			for(k = 0; k < nPoints[i]; k++) {
				if(mcvClassification[k] == labels[i][k]) {
					corrLabel ++;
				}
			}

			accuracyResult[i*numParam+j] = (100.0 * (float) corrLabel) / nPoints[i];
		}
	}

	cout << "Accuracy results for different parameter set: " << endl;
	if(kernelType == GAUSSIAN) {
		for(i = 0; i < nSVM; i++) {
			cout << "svm " << i << ": " << endl;
			for(j = 0; j < numCost; j++) {
				for(k = 0; k < numGamma; k++) {
					float gamma = inGamma[0];
					int p;
					for(p = 0; p < k; p++)
						gamma *= inGamma[2];
					cout << "(" << inCost[0]+j*inCost[2] << ", " << gamma << ", " << accuracyResult[i*numParam+j*numGamma+k] << ") "; 
				}
			}
			cout << endl;
		}
	} else {
		for(i = 0; i < nSVM; i++) {
			cout << "svm " << i << ": " << endl;
			for(j = 0; j < numCost; j++) {
				cout << "(" << inCost[0]+j*inCost[2] << ", " << accuracyResult[i*numCost+j] << ") ";
			}
			cout << endl;
		}
	}
	cout << endl;
	
	float* bestAccuracy = (float*) malloc(sizeof(float)*nSVM);
	float* bestParamId = (float*) malloc(sizeof(float)*nSVM);
	for(i = 0; i < nSVM; i++) {
		bestAccuracy[i] = accuracyResult[i*numParam];
		bestParamId[i] = 0;
		for(j = 1; j < numParam; j++) {
			if(accuracyResult[i*numParam+j] > bestAccuracy[i]) {
				bestAccuracy[i] = accuracyResult[i*numParam+j];
				bestParamId[i] = j;
			}
		}
	}

	// if needed, can return (bestCost, bestGamma), however, remember to uncomment free()
	float* bestCost = NULL;
	float* bestGamma = NULL;

	if(kernelType == GAUSSIAN) {
		bestCost = (float*) malloc(sizeof(float)*nSVM);
		bestGamma = (float*) malloc(sizeof(float)*nSVM);
		for(i = 0; i < nSVM; i++) {
			int costId = bestParamId[i] / numGamma;
			int gammaId = bestParamId[i] - numGamma * costId;

			bestCost[i] = inCost[0] + costId*inCost[2];
			bestGamma[i] = inGamma[0];
			for(j = 0; j < gammaId; j++)
				bestGamma[i] *= inGamma[2];
			cout << "svm " << i << ": " << endl;
			cout << "RBF Kernel: Find the best parameter set (cost, gamma, accuracy): (" << bestCost[i] << ", " << bestGamma[i] << ", " << bestAccuracy[i] << ")." << endl;
		}
	} else {
		bestCost = (float*) malloc(sizeof(float)*nSVM);
		for(i = 0; i < nSVM; i++) {
			bestCost[i] = inCost[0] + bestParamId[i]*inCost[2];
			cout << "svm " << i << ": " << endl;
			cout << "Linear Kernel: find the best parameter (cost, accuracy): " << "(" << bestCost[i] << ", " << bestAccuracy[i] << ")." << endl;
		}
	}

	// free pointer
	if(isCostNull)
		free(inCost);
	if(isGammaNull)
		free(inGamma);
	free(numClass);
	free(accuClass);

	for(i = 0; i < totalClass; i++) {
		free(labelsPerClass[i]);
	}

	for(i = 0; i < numSVM; i++) {
		free(kpArray[i]);
	}
	free(nPointsArray);
	free(nDimensionArray);
	free(labelsPerClass);
	free(kpArray);
	free(costArray);
	free(heuristicMethodArray);
	free(epsilonArray);
	free(toleranceArray);
	free(transposedDataArray);

	free(numTestingPointsEachFolder);
	free(labelsArray);
	free(accuracyResult);
	free(mcvClassification);
	free(bestAccuracy);
	free(bestParamId);
	free(bestGamma);
	free(bestCost);
}

/**
 * MSVM_Grid_MC_One selects best parameter set for multiple multi-class SVMs via one-against-one cross validation
 * @param nSVM the number of SVMs for finding best parameter set
 * @param nPoints the number of points for each SVM
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the array of row major stores of the original multi SVM data
 * @param labels store the labels for the orignal multi SVM data
 * @param folder specifies the folder for cross validation
 * @param kernelType the kernel type for trainging (eg. LINEAR, GAUSSIAN)
 * @param inCost sets the range of the cost parameter for testing, inCost[0] is the start value, inCost[1] is the end value and inCost[2] is the addend. (For example, inCost=[1, 5, 2] will test the value of cost from the set {1, 3, 5})
 * @param inGamma sets the range of the gamma parameter for testing, inGamma[0] is the start value, inGamma[1] is the end value and inGamma[2] is the multiplier. (For example, inGamma=[0.1, 1, 2] will test the value of gamma from the set {0.1, 0.2, 0.4, 0.8})
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 */
void msvm_grid_MC_One(int nSVM, int* nPoints, int nDimension, float** transposedData, float** labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE) {
	int i, j, k;

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
	vector<vector<float> > category;
	for(i = 0; i < nSVM; i++) {
		vector<float> tmp;
		for(j = 0; j < nPoints[i]; j++) {
			if(find(tmp.begin(), tmp.end(), labels[i][j]) == tmp.end()) {
				tmp.push_back(labels[i][j]);
			}
		}
		category.push_back(tmp);
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

	// random split the data set
	int** permutation = (int**) malloc(sizeof(int*)*nSVM);
	for(i = 0; i < nSVM; i++) {
		permutation[i] = (int*) malloc(sizeof(int)*nPoints[i]);
		for(j = 0; j < nPoints[i]; j++)
			permutation[i][j] = j;
		srand(unsigned(time(0)));
		random_shuffle(permutation[i], permutation[i]+nPoints[i]);
	}

	int** dataPartitionSize = (int**) malloc(sizeof(int*)*nSVM);
	for(i = 0; i < nSVM; i++) {
		dataPartitionSize[i] = (int*) malloc(sizeof(int)*folder);
		for(j = 0; j < folder; j++) {
			dataPartitionSize[i][j] = nPoints[i] / folder;
			if(j < nPoints[i] % folder) 
				dataPartitionSize[i][j] ++;
		}
	}

	float*** dataPartitionArray = (float***) malloc(sizeof(float**)*nSVM);
	float*** dataTranPartitionArray = (float***) malloc(sizeof(float**)*nSVM);

	for(i = 0; i < nSVM; i++) {
		dataPartitionArray[i] = (float**)malloc(sizeof(float*)*folder);
		dataTranPartitionArray[i] = (float**)malloc(sizeof(float*)*folder);
		for(j = 0; j < folder; j++) {
			dataPartitionArray[i][j] = (float*)malloc(sizeof(float)*dataPartitionSize[i][j]*nDimension);
			dataTranPartitionArray[i][j] = (float*)malloc(sizeof(float)*dataPartitionSize[i][j]*nDimension);
		}
	}

	// randomly partition the data set into folder parts, each part with size dataPartitionSize[i]
	// assert: for k-folder cross validation, training data contains (k-1)-folder and testing data contains 1-folder
	// make sure that there are numClass in the (k-1)-folder training data
	for(i = 0; i < nSVM; i++)
		setupDataCV(folder, nPoints[i], nDimension, transposedData[i], permutation[i], dataPartitionArray[i], dataTranPartitionArray[i]);

	int* numClass = (int*) malloc(sizeof(int)*nSVM);
	for(i = 0; i < nSVM; i++)
		numClass[i] = category[i].size();

	int* numPair = (int*) malloc(sizeof(int)*nSVM);
	int totalPair = 0;
	for(i = 0; i < nSVM; i++) {
		numPair[i] = numClass[i]*(numClass[i]-1)/2;
		totalPair += numPair[i];
	}
	int numSVM = numParam * folder * totalPair;

	int* accuPair = (int*) malloc(sizeof(int)*nSVM);
	accuPair[0] = 0;
	for(i = 1; i < nSVM; i++)
		accuPair[i] = accuPair[i-1] + numPair[i-1];

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

	int* nPointsPar = (int*) malloc(sizeof(int)*nSVM);
	float** transposedDataPar = (float**) malloc(sizeof(float*)*nSVM);
	float** labelsPar = (float**) malloc(sizeof(float*)*nSVM);
	int* querynPoints = (int*) malloc(sizeof(int)*nSVM);
	float** queryData = (float**) malloc(sizeof(float*)*nSVM);
	float*** queryLabelsPar = (float***) malloc(sizeof(float**)*nSVM);

	for(i = 0; i < nSVM; i++) {
		transposedDataPar[i] = (float*) malloc(sizeof(float)*(nPoints[i]-nPoints[i]/folder)*nDimension);
		labelsPar[i] = (float*) malloc(sizeof(float)*(nPoints[i]-nPoints[i]/folder));
		queryLabelsPar[i] = (float**) malloc(sizeof(float*)*folder);
	}

	// classDist stores the class distribution of the data set
	int** classDist = (int**) malloc(sizeof(int*)*nSVM);
	int** nPointsParArray = (int**) malloc(sizeof(int*)*nSVM);
	for(i = 0; i < nSVM; i++) {
		classDist[i] = (int*) malloc(sizeof(int)*numClass[i]);
		nPointsParArray[i] = (int*)malloc(sizeof(int)*numPair[i]);
	}

	for(k = 0; k < nSVM; k++) {
		for(i = 0; i < folder; i++) {
			queryLabelsPar[k][i] = (float*)malloc(sizeof(float)*(nPoints[k]/folder+1));

			nPointsPar[k] = nPoints[k] - dataPartitionSize[k][i];
			querynPoints[k] = dataPartitionSize[k][i];

			int qLabelIdx = 0;
			int dLabelIdx = 0;
			for(j = 0; j < folder; j++) {
				int p = j;
				while(p < nPoints[k]) {
					if(j == i) {
						queryLabelsPar[k][i][qLabelIdx] = labels[k][permutation[k][p]];
						qLabelIdx ++;
					} else {
						labelsPar[k][dLabelIdx] = labels[k][permutation[k][p]];
						dLabelIdx++;
					}
					p += folder;
				}
			}

			int accuPoints = 0;
			for(j = 0; j < folder; j++) {
				if(j == i) {
					queryData[k] = dataPartitionArray[k][i];
				} else {
					memcpy(transposedDataPar[k]+accuPoints*nDimension, dataTranPartitionArray[k][j], sizeof(float)*dataPartitionSize[k][j]*nDimension);
					accuPoints += dataPartitionSize[k][j];
				}
			}

			vector<float> categoryPar;
			for(j = 0; j < nPointsPar[k]; j++) {
				if(find(categoryPar.begin(), categoryPar.end(), labelsPar[k][j]) == categoryPar.end()) {
					categoryPar.push_back(labelsPar[k][j]);
				}
			}
			if(categoryPar.size() < numClass[k]) {
				cout << "Assertion fail: (folder-1) paritition do not contain all classes!" << endl; 
				return ;
			}

	  		for(j = 0; j < numClass[k]; j++)
	  			classDist[k][j] = 0;
			for(j = 0; j < nPointsPar[k]; j++) {
				int idx = find(category[k].begin(), category[k].end(), labelsPar[k][j]) - category[k].begin();
				classDist[k][idx] ++;
			}

			int classA = 0;
	  		int classB = 1;
			for(j = 0; j < numPair[k]; j++) {
				int svmId = numParam*accuPair[k]*folder + i*numPair[k] + j;
				nPointsParArray[k][j] = classDist[k][classA] + classDist[k][classB];
				nPointsArray[svmId] = nPointsParArray[k][j];

	    		nDimensionArray[svmId] = nDimension;

	    		labelsArray[svmId] = (float*) malloc(sizeof(float) * nPointsArray[svmId]);

	    		int p;
	    		// set labelsArray for one-against-one cross valiation
		    	for(p = 0; p < nPointsArray[svmId]; p++) {
		    		if(p < classDist[k][classA])
		    			labelsArray[svmId][p] = 1;
		    		else
		    			labelsArray[svmId][p] = -1;
		    	}

		    	if(classB == numClass[k]-1) {
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

				testDataArray[svmId] = queryData[k];
	    		testNPointsArray[svmId] = querynPoints[k];
	    		testLabelsArray[svmId] = queryLabelsPar[k][i];
			}

			setupData(numPair[k], category[k], classDist[k], nPointsPar[k], nDimension, transposedDataPar[k], labelsPar[k], nPointsParArray[k], dataArray+numParam*accuPair[k]*folder+i*numPair[k], transposedDataArray+numParam*accuPair[k]*folder+i*numPair[k]);
		
			for(j = 0; j < numPair[k]; j++) {
				int p;
				int svmIdInFolder = numParam*accuPair[k]*folder + i*numPair[k] + j;
				for(p = 1; p < numParam; p++) {
					int svmId = numParam*accuPair[k]*folder + (p*folder+i)*numPair[k] + j;
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
	}

	cout << "Finish Setting Parameters!" << endl;
	
	float** results;
	svmPredict(numSVM, dataArray, nPointsArray, nDimensionArray, labelsArray, kpArray, costArray, heuristicMethodArray, epsilonArray, toleranceArray, transposedDataArray, testDataArray, testNPointsArray, testLabelsArray, NULL, 1, true, &results);
	
	float* accuracyResult = (float*)malloc(sizeof(float)*numParam*nSVM);
	// mcvClassification stores the classification for each data point in cross validation
	int maxPoints = 0;
	for(i = 0; i < nSVM; i++) {
		if(maxPoints < nPoints[i])
			maxPoints = nPoints[i];
	}
	float* mcvClassification = (float*)malloc(sizeof(float)*maxPoints);
	int maxClass = 0;
	for(i = 0; i < nSVM; i++) {
		if(maxClass < numClass[i]) {
			maxClass = numClass[i];
		}
	}
	int* vote = (int*)malloc(sizeof(int)*maxClass);
	int* dataPartitionAccuSize = (int*)malloc(sizeof(int)*folder);
	
	for(k = 0; k < nSVM; k++) {	

		dataPartitionAccuSize[0] = 0;
		for(i = 1; i < folder; i++)
			dataPartitionAccuSize[i] = dataPartitionAccuSize[i-1] + dataPartitionSize[k][i-1];

		for(i = 0; i < numParam; i++) {
			for(j = 0; j < folder; j++) {
				int p;
				for(p = 0; p < dataPartitionSize[k][j]; p++) {
			   		int q;
			   		for(q = 0; q < numClass[k]; q++)
			   			vote[q] = 0;
			   		for(q = 0; q < numPair[k]; q++) {
			   			int classA = 0;
			   			while(q >= (classA+1)*(numClass[k]-1)-(classA+1)*classA/2)
			   				classA ++;
			   			int classB = q - classA*(numClass[k]-1) + classA*(classA-1)/2 + classA + 1;

			   			int svmId = numParam*accuPair[k]*folder+(i*folder+j)*numPair[k]+q;
			   			if(results[svmId][p] > 0)
			   				vote[classA] ++;
			   			else
			   				vote[classB] ++;
			      	}

			   		int max = vote[0];
			   		mcvClassification[dataPartitionAccuSize[j]+p] = category[k][0];
			   		for(q = 0; q < numClass[k]; q++) {
			   			if(vote[q] > max) {
			   				max = vote[q];
			   				mcvClassification[dataPartitionAccuSize[j]+p] = category[k][q];
			   			}
			   		}
			   	}
			}

			// calculate accuracy
		   	int correct = 0;
		   	for(j = 0; j < folder; j++) {
		   		int p;
		   		for(p = 0; p < dataPartitionSize[k][j]; p++) {
		   			if(mcvClassification[dataPartitionAccuSize[j]+p] == queryLabelsPar[k][j][p]) {
			   			correct ++;
			   		}
		   		}
		   	}
		   	accuracyResult[k*numParam+i] = (100 * (float) correct ) / nPoints[k];
		}
	}

	cout << "Accuracy results for different parameter set: " << endl;
	if(kernelType == GAUSSIAN) {
		for(i = 0; i < nSVM; i++) {
			cout << "svm " << i << ": " << endl;
			for(j = 0; j < numCost; j++) {
				for(k = 0; k < numGamma; k++) {
					float gamma = inGamma[0];
					int p;
					for(p = 0; p < k; p++)
						gamma *= inGamma[2];
					cout << "(" << inCost[0]+j*inCost[2] << ", " << gamma << ", " << accuracyResult[i*numParam+j*numGamma+k] << ") "; 
				}
			}
			cout << endl;
		}
	} else {
		for(i = 0; i < nSVM; i++) {
			cout << "svm " << i << ": " << endl;
			for(j = 0; j < numCost; j++) {
				cout << "(" << inCost[0]+j*inCost[2] << ", " << accuracyResult[i*numCost+j] << ") ";
			}
			cout << endl;
		}
	}
	cout << endl;
	
	float* bestAccuracy = (float*) malloc(sizeof(float)*nSVM);
	float* bestParamId = (float*) malloc(sizeof(float)*nSVM);
	for(i = 0; i < nSVM; i++) {
		bestAccuracy[i] = accuracyResult[i*numParam];
		bestParamId[i] = 0;
		for(j = 1; j < numParam; j++) {
			if(accuracyResult[i*numParam+j] > bestAccuracy[i]) {
				bestAccuracy[i] = accuracyResult[i*numParam+j];
				bestParamId[i] = j;
			}
		}
	}

	// if needed, can return (bestCost, bestGamma), however, remember to uncomment free()
	float* bestCost = NULL;
	float* bestGamma = NULL;

	if(kernelType == GAUSSIAN) {
		bestCost = (float*) malloc(sizeof(float)*nSVM);
		bestGamma = (float*) malloc(sizeof(float)*nSVM);
		for(i = 0; i < nSVM; i++) {
			int costId = bestParamId[i] / numGamma;
			int gammaId = bestParamId[i] - numGamma * costId;

			bestCost[i] = inCost[0] + costId*inCost[2];
			bestGamma[i] = inGamma[0];
			for(j = 0; j < gammaId; j++)
				bestGamma[i] *= inGamma[2];
			cout << "svm " << i << ": " << endl;
			cout << "RBF Kernel: Find the best parameter set (cost, gamma, accuracy): (" << bestCost[i] << ", " << bestGamma[i] << ", " << bestAccuracy[i] << ")." << endl;
		}
	} else {
		bestCost = (float*) malloc(sizeof(float)*nSVM);
		for(i = 0; i < nSVM; i++) {
			bestCost[i] = inCost[0] + bestParamId[i]*inCost[2];
			cout << "svm " << i << ": " << endl;
			cout << "Linear Kernel: find the best parameter (cost, accuracy): " << "(" << bestCost[i] << ", " << bestAccuracy[i] << ")." << endl;
		}
	}

	// free pointer
	if(isCostNull)
		free(inCost);
	if(isGammaNull)
		free(inGamma);

		// pay attention to the order of freeing pointers
	for(k = 0; k < nSVM; k++) {
		for(i = 0; i < folder; i++) {
			free(dataPartitionArray[k][i]);
			free(dataTranPartitionArray[k][i]);
			free(queryLabelsPar[k][i]);
		}
	}
	
	for(i = 0; i < nSVM; i++) {
		free(permutation[i]);
		free(dataPartitionSize[i]);
		free(dataPartitionArray[i]);
		free(dataTranPartitionArray[i]);
		free(queryLabelsPar[i]);
	}

	free(permutation);
	free(dataPartitionSize);
	free(dataPartitionArray);
	free(dataTranPartitionArray);
	free(queryLabelsPar);

	for(k = 0; k < nSVM; k++) {
		for(i = 0; i < folder; i++) {
			for(j = 0; j < numPair[k]; j++) {
				int svmId = numParam*accuPair[k]*folder+i*numPair[k]+j;
				free(dataArray[svmId]);
				free(transposedDataArray[svmId]);
				free(labelsArray[svmId]);
			}
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

	for(i = 0; i < nSVM; i++) {
		free(transposedDataPar[i]);
		free(labelsPar[i]);
		free(classDist[i]);
		free(nPointsParArray[i]);
	}

	free(transposedDataPar);
	free(labelsPar);
	free(queryData);

	free(classDist);
	free(nPointsParArray);

	free(accuracyResult);
	free(mcvClassification);
	free(vote);
	free(dataPartitionAccuSize);

	free(numClass);
	free(numPair);
	free(accuPair);

	free(bestAccuracy);
	free(bestParamId);
	free(bestCost);
	free(bestGamma);

}