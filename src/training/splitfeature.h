#ifndef GPU_SVM_SPLITFEATURE_H
#define GPU_SVM_SPLITFEATURE_H

/**
 * processData for cross validation
 * @param numSVM the number of SVMs after cross-validation split
 * @param folder the folder for cross validation
 * @param nPointsArray records the number of training points for each SVM
 * @param nDimensionArray records the number of dimensions for each SVM

 * @param dataArrayIn used to store the output training data in column major manner
 * @param transposedDataArray used to store the output training data in row major manner
 * @param testDataArrayIn used to store the output testing data in column major manner

 Others params are the addresses of the source and after-processing arrays
 */

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
                   float *toleranceArrayOut);

#endif //GPU_SVM_SPLITFEATURE_H
