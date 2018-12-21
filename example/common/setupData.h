#include <vector>

using namespace std;

/**
 * Setup both dataArray and transposedDataArray for one-againse-one multi-class training
 * @param numSVM the number of SVMs for training, equals to category.size() * (category.size()-1) / 2
 * @param category stores the labels for different classes
 * @param classDist stores the class distribution for each class, where classDist[0] is the number of points in the original data set which has label category[0]
 * @param nPoints the number of points in the original data
 * @param nDimension the number of dimensions of the data points in the original data
 * @param transposedData the row major store of the original data
 * @param labels store the labels for the orignal data
 * @param nPointsArray the number of points for each SVM, where nPointsArray[svmId] = classDist[i] + classDist[j] and there is a one to one mapping between svmId and (i, j), given by svmId = i * (category.size()-1) - i*(i-1)/2 + j - i - 1. Please refer to createSVMData() for the detailed mapping;  
 * @param dataArray is used to store the output column major data for each SVM
 * @param transposedDataArray is used to store the output row major data for each SVM
 */
void setupData(int numSVM, vector<float> category, int* classDist, int nPoints, int nDimension, float* transposedData, float* labels, int* nPointsArray, float** dataArray, float** transposedDataArray);

void setupDataCV(int folder, int nPoints, int nDimension, float* transposedData, int* permutation, float** dataPartitionArray, float** dataTranPartitionArray);
