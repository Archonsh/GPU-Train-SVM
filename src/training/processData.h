#ifndef SVMPROCESSDATA
#define SVMPROCESSDATA

/**
 * processData for cross validation 
 * @param numSVM the number of SVMs
 * @param floder the folder for cross validation
 * @param dataArray the training data of all SVMs
 * @param nPointsArray records the number of training points for each SVM
 * @param nDimensionArray records the number of dimensions for each SVM
 * @param permutation random permutation of point indexes for each SVM
 * @param dataArrayOut used to store the output training data in row major manner
 * @param transposedDataArrayOut used to store the output training data in column major manner
 * @param testDataArrayOut used to store the output testing data in row major manner
 * @param testTransposedDataArrayOut used to store the output testing data in column major manner
 */
int* processData(int ngpus, int numSVM, int folder, float** dataArray, int* nPointsArray, int* nDimensionArray, int** permutation, float** dataArrayOut, float** transposedDataArrayOut, float** testDataArrayOut, float** testTransposedDataArrayOut);

#endif