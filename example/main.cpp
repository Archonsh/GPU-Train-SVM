/*
 * main.cpp
 *
 *  Add on: Dec 23, 2016
 *  Add by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  create a running example of using grid.cu
 */

#include "./grid.h"
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int readSvm(const char* filename, float** p_data, float** p_labels, int* p_npoints, int* p_dimension, float** p_transposed_data = 0);

int main(int argc, char * argv[]) {
	int i;

	// if(argc != 2) {
	// 	cout << "Usage: grid_bin train_file" << endl;
	// 	return 0;
	// }

	int kernelType = LINEAR;
	if(argc >= 3) {
		kernelType = GAUSSIAN;
	}

	char* dataFilename = argv[1];

	int nPoints;
	int nDimension;
	float* data;
	float* transposedData;
	float* labels;
	// read in data from input files
	readSvm(dataFilename, &data, &labels, &nPoints, &nDimension, &transposedData);

	vector<float> category;
	for(i = 0; i < nPoints; i++) {
		if(find(category.begin(), category.end(), labels[i]) == category.end()) {
			category.push_back(labels[i]);
		}
	}
	int numClass = category.size();

	float inCost[] = {0.5, 1, 0.5};
	float inGamma[] = {0.01, 1, 10};

	if(argc >= 4) {
		if(numClass == 2)
			grid(nPoints, nDimension, transposedData, labels, data, 5, kernelType, inCost, inGamma);
		else if (numClass > 2)
			grid_MC_All(nPoints, nDimension, transposedData, labels, data, 5, kernelType, inCost, inGamma);
			// grid_MC_One(nPoints, nDimension, transposedData, labels, 5, kernelType, inCost, inGamma);
	} else {
		// test grid for multiple SVMs
		int nSVM = 2;

		int* nPointsArray = (int*) malloc(sizeof(int)*nSVM);
		float** transposedDataArray = (float**) malloc(sizeof(float*)*nSVM);
		float** labelsArray = (float**) malloc(sizeof(float*)*nSVM);
		for(i = 0; i < nSVM; i++) {
			nPointsArray[i] = nPoints;
			transposedDataArray[i] = transposedData;
			labelsArray[i] = labels;
		}

		if(numClass == 2)
			msvm_grid(nSVM, nPointsArray, nDimension, transposedDataArray, labelsArray, 5, kernelType, inCost, inGamma);
		else if (numClass > 2) {
			msvm_grid_MC_One(nSVM, nPointsArray, nDimension, transposedDataArray, labelsArray, 5, kernelType, inCost, inGamma);
		}
	}
}