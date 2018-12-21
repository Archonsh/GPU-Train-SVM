#include "../src/training/kernelType.h"
#include "../include/svmCommon.h"

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
void grid(int nPoints, int nDimension, float* transposedData, float* labels, float* data = NULL, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE);

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
void grid_MC_All(int nPoints, int nDimension, float* transposedData, float* labels, float* data = NULL, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE);

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
void grid_MC_One(int nPoints, int nDimension, float* transposedData, float* labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE);


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
void msvm_grid(int nSVM, int* nPoints, int nDimension, float** transposedData, float** labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE);


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
void msvm_grid_MC_All(int nSVM, int* nPoints, int nDimension, float** transposedData, float** labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE);


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
void msvm_grid_MC_One(int nSVM, int* nPoints, int nDimension, float** transposedData, float** labels, int folder = 5, int kernelType = GAUSSIAN, float* inCost = NULL, float* inGamma = NULL, float tolerance = 1e-3f, float epsilon = 1e-5f, SelectionHeuristic heuristicMethod = ADAPTIVE);