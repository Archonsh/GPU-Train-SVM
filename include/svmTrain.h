#ifndef SVMTRAINH
#define SVMTRAINH

#include "svmCommon.h"
#include "../src/training/kernelType.h"

/**
 * Performs SVM training
 * @param data the training points, stored as a flat column major array.
 * @param nPoints the number of training points
 * @param nDimension the dimensionality of the training points
 * @param labels the labels for the training points (+/- 1.0f)
 * @param alpha a pointer to a float buffer that will be allocated by performTraining and contain the unsigned weights of the classifier after training.
 * @param kp a pointer to a struct containing all the information about the kernel parameters.  The b offset from the training process will be stored in this struct after training is complete
 * @param cost the training cost parameter C
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param transposedData the training points, stored as a flat row major array.  This pointer can be omitted.
 */
void performTraining(float* data, int nPoints, int nDimension, float* labels, float** alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod = ADAPTIVE, float epsilon = 1e-5f, float tolerance = 1e-3f, float* transposedData = 0);

/**
 * Densifies model by collecting Support Vectors from training set.
 * @param trainingPoints the training points, stored as a flat column major array.
 * @param nTrainingPoints the number of training points
 * @param nDimension the dimensionality of the training points
 * @param trainingAlpha the weights learned during the training process
 * @param trainingLabels the labels of the training points (+/- 1.0f)
 * @param p_supportVectors a pointer to a float array, where the Support Vectors will be stored as a flat column major array
 * @param p_nSV a pointer to the number of Support Vectors in this model
 * @param p_alpha a pointer to a float array, where the Support Vector weights will be stored
 * @param epsilon an optional parameter controling the threshold for which points are considered Support Vectors.  Default is 1e-5f.
 */
void formModel(float* trainingPoints, int nTrainingPoints, int nDimension, float* trainingAlpha, float* trainingLabels, float** p_supportVectors, int* p_nSV, float** p_alpha, float epsilon = 1e-5f);

/**
 * Performs Multi-SVM training
 * @param numSVM the number of SVMs
 * @param dataArray the training data of all SVMs
 * @param nPointsArray records the number of training points for each SVM
 * @param nDimensionArray records the number of dimensions for each SVM
 * @param labelsArray records the lables of all training points
 * @param p_alphaArary pointer to an array of float buffer that will be allocated by performMultiTraining and contain the unsigned weights of the classifier after training all SVMs
 * @param kpArray specifies the training paramters for each SVM
 * @param costArray the training cost parameter C for each SVM
 * @param heuristicMethodArray specifies selection heuristic method for each SVM.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param epsilonArray this parameter controls which training points are counted as support vectors for each SVM.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param toleranceArray this parameter controls how close to the optimal solution the optimization process must go for each SVM.  Default is 1e-3f.
 * @param transposedDataArray the transposed dataArray 
 * @param gpuId specifies which gpu to use for training
 */
void performMultiTraining(int numSVM, float** dataArray, int* nPointsArray, int* nDimensionArray, float** labelsArray, float*** p_alphaArray, Kernel_params** kpArray, float* costArray, SelectionHeuristic* heuristicMethodArray, float* epsilonArray, float* toleranceArray, float** transposedDataArray, int gpuId = -1);

/**
 * Uses multiple GPUs to train many SVMs concurrently from input files (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVM for training
 * @param trainingFile the names of the input training files
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param kernelType the kernel type for trainging (eg. LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID)
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param cost the training cost parameter C
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 */
void svmTrainFromFile(int numSVM, char** trainingFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

/**
 * Uses multiple GPUs to predict many SVMs concurrently from input files (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVM for training
 * @param trainingFile the names of the input training files
 * @param testingFile the names of testing files
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param kernelType the kernel type for trainging (eg. LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID)
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param cost the training cost parameter C
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 */
void svmPredictFromFile(int numSVM, char** trainingFile, char** testingFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

/**
 * Uses multiple GPUs to cross validation many SVMs concurrently from input files (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVM for training
 * @param trainingFile the names of the input training files
 * @param folder specifies the folder for cross validation
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param kernelType the kernel type for trainging (eg. LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID)
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param cost the training cost parameter C
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 */
void svmCrossValidationFromFile(int numSVM, char** trainingFile, int folder, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

/**
 * Uses multiple GPUs to train many subset SVMs concurrently from input files (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVM for training
 * @param dataFile the file name that contains training point
 * @param subsetFile the names of the subset files
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param kernelType the kernel type for trainging (eg. LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID)
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param cost the training cost parameter C
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 */
void svmSubsetTrainFromFile(int numSVM, char* dataFile, char** subsetFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

/**
 * Uses multiple GPUs to predict many subset SVMs concurrently from input files (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVM for training
 * @param dataFile the file name that contains training point
 * @param subsetFile the names of the subset files
 * @param testingFile the names of testing files
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param kernelType the kernel type for trainging (eg. LINEAR, POLYNOMIAL, GAUSSIAN, SIGMOID)
 * @param heuristicMethod variable selection heuristic method.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param cost the training cost parameter C
 * @param tolerance this parameter controls how close to the optimal solution the optimization process must go.  Default is 1e-3f.
 * @param epsilon this parameter controls which training points are counted as support vectors.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 */
void svmSubsetPredictFromFile(int numSVM, char* dataFile, char** subsetFile, char** testingFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

/**
 * Uses multiple GPUs to train many SVMs concurrently (Notice: this function uses the default values for each parameter of different kernels)
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
 * @param modelFilename specifies the output file name for training
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 */
void svmTrain(int numSVM, float** dataArray, int* nPointsArray, int* nDimensionArray, float** labelsArray, Kernel_params** kpArray, float* costArray, SelectionHeuristic* heuristicMethodArray, float* epsilonArray, float* toleranceArray, float** transposedDataArray, char** modelFilename = NULL, float ratio = 0.5);

/**
 * Uses multiple GPUs to predict many SVMs concurrently (Notice: this function uses the default values for each parameter of different kernels)
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
 * @param testDataArray the testing data of all SVMs
 * @param testNPointsArray records the number of testing points for each SVM
 * @param testLabelsArray records the labels of all testing point
 * @param predictFilename specifies the output file name for prediction
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 * @param isMultiClass specifies whether the prediction is for multi-class classification (set several variables inside the function to facilitate one-against-one multi-class classification, please view "example_letter2.cu" for reference)
 * @param mPredictResult points to a two dimensional array in which stores the prediction result for multi-class classification (Notice: this variable is set only when isMultiClass is true)
 */
void svmPredict(int numSVM, float** dataArray, int* nPointsArray, int* nDimensionArray, float** labelsArray, Kernel_params** kpArray, float* costArray, SelectionHeuristic* heuristicMethodArray, float* epsilonArray, float* toleranceArray, float** transposedDataArray, float** testDataArray, int* testNPointsArray, float** testLabelsArray, char** predictFilename = NULL, float ratio = 0.5, bool isMultiClass = false, float*** mPredictResult = NULL);

/**
 * Uses multiple GPUs to cross validation many SVMs concurrently (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVMs
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
 * @param isMultiClass specifies whether the cross validation is for multi-class classification (set several variables inside the function to facilitate one-against-all multi-class classification, please view "example_letter.cu" for reference)
 * @param mcvResult points to a two dimensional array in which stores the cross validation result for multi-class cross validation (Notice: this variable is set only when isMultiClass is true)
 * @param outputAccuracy specifies whether should output the accuracy result
 * @param accuracyResult points to an array to store accuracy result (Notice: this variable is set only when outputAccuracy is true)
 */
void svmCrossValidation(int numSVM, int* nPointsArray, int* nDimensionArray, float** labelsArray, Kernel_params** kpArray, float* costArray, SelectionHeuristic* heuristicMethodArray, float* epsilonArray, float* toleranceArray, float** transposedDataArray, int folder = 5, float ratio = 0.5, bool isMultiClass = false, float*** mcvResult = NULL, bool outputAccuracy = false, float* accuracyResult = NULL);

/**
 * Uses multiple GPUs to train many subset SVMs concurrently (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVMs
 * @param nPointsArray records the number of training points for each SVM
 * @param nDimensionArray records the number of dimensions for each SVM
 * @param labelsArray records the lables of all training points
 * @param kpArray specifies the training paramters for each SVM
 * @param costArray the training cost parameter C for each SVM
 * @param heuristicMethodArray specifies selection heuristic method for each SVM.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param epsilonArray this parameter controls which training points are counted as support vectors for each SVM.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param toleranceArray this parameter controls how close to the optimal solution the optimization process must go for each SVM.  Default is 1e-3f.
 * @param transposedDataArray the transposed data array
 * @param subsetIdx records the subset indexes for training or prediction (used when functionality is 3 or 4)
 * @param modelFilename specifies the output file name for training
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 */
void svmSubsetTrain(int numSVM, int* nPointsArray, int* nDimensionArray, float** labelsArray, Kernel_params** kpArray, float* costArray, SelectionHeuristic* heuristicMethodArray, float* epsilonArray, float* toleranceArray, float** transposedDataArray, int** subsetIdx, char** modelFilename = NULL, float ratio = 0.5);

/**
 * Uses multiple GPUs to predict many subset SVMs concurrently (Notice: this function uses the default values for each parameter of different kernels)
 * @param numSVM the number of SVMs
 * @param nPointsArray records the number of training points for each SVM
 * @param nDimensionArray records the number of dimensions for each SVM
 * @param labelsArray records the lables of all training points
 * @param kpArray specifies the training paramters for each SVM
 * @param costArray the training cost parameter C for each SVM
 * @param heuristicMethodArray specifies selection heuristic method for each SVM.  Choices are FIRSTORDER, SECONDORDER, RANDOM and ADAPTIVE.  ADAPTIVE is the default
 * @param epsilonArray this parameter controls which training points are counted as support vectors for each SVM.  Default is 1e-5f.  Making this smaller sometimes prevents convergence.
 * @param toleranceArray this parameter controls how close to the optimal solution the optimization process must go for each SVM.  Default is 1e-3f.
 * @param transposedDataArray the transposed data array
 * @param subsetIdx records the subset indexes for training or prediction (used when functionality is 3 or 4)
 * @param testDataArray the testing data of all SVMs
 * @param testNPointsArray records the number of testing points for each SVM
 * @param predictFilename specifies the output file name for prediction
 * @param testLabelsArray records the labels of all testing point
 * @param ratio the ratio of kernel matrix stored in GPU memory as cache (eg. ratio = 1 means storing the whole kernel matrix in GPU memory, ratio can be any value between 0 and 1, excluding 0; the higher the ratio, the faster the training process and the less number of SVMs can be trained concurrently)
 */
void svmSubsetPredict(int numSVM, int* nPointsArray, int* nDimensionArray, float** labelsArray, Kernel_params** kpArray, float* costArray, SelectionHeuristic* heuristicMethodArray, float* epsilonArray, float* toleranceArray, float** transposedDataArray, int** subsetIdx, float** testDataArray, int* testNPointsArray, float** testLabelsArray, char** predictFilename = NULL, float ratio = 0.5);



#endif
