/*
 * trainingMain.cu
 *
 *  Modified on: May 27, 2016
 *  Modified by: Zhu Lei
 *  Email: zlheui2@gmail.com
 *
 *  Augmented power: can train, predict and cross-validation multiple SVMs concurrently using multiple GPUs
 *  Please view the README-ZHULEI file for more details
 *
 *  Reference: 
 *    Catanzaro, B., Sundaram, N., & Keutzer, K. (2008). Fast support vector machine training and classification on graphics processors. Proceedings of the 25th International Conference on Machine Learning - ICML '08. doi:10.1145/1390156.1390170
 *    
 */

#include <sys/time.h>   
#include <stdio.h>   
#include <math.h>
#include <string.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <getopt.h>
#include <stdlib.h>
#include <iostream>

#include "svmCommon.h"
#include "../common/svmIO.h"
#include "../common/framework.h"
#include "kernelType.h"

using namespace std;

void printHelp();
void performTraining(float* data, int nPoints, int nDimension, float* labels, float** p_alpha, Kernel_params* kp, float cost, SelectionHeuristic heuristicMethod, float epsilon, float tolerance, float* transposedData);

void svmTrainFromFile(int numSVM, char** trainingFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);
void svmPredictFromFile(int numSVM, char** trainingFile, char** testingFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);
void svmCrossValidationFromFile(int numSVM, char** trainingFile, int folder, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);
void svmSubsetTrainFromFile(int numSVM, char* dataFile, char** subsetFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);
void svmSubsetPredictFromFile(int numSVM, char* dataFile, char** subsetFile, char** testingFile, float ratio = 0.5, int kernelType = GAUSSIAN, SelectionHeuristic heuristicMethod = ADAPTIVE, float cost = 1.0f, float tolerance = 1e-3f, float epsilon = 1e-5f);

static int kType = GAUSSIAN;
int main( const int argc, char** argv)  { 
  int currentOption;
  float parameterA = -0.125f;
  float parameterB = 1.0f;
  float parameterC = 3.0f;

  bool parameterASet = false;
  bool parameterBSet = false;
  bool parameterCSet = false;
  
  
  SelectionHeuristic heuristicMethod = ADAPTIVE;
  float cost = 1.0f;
  
  float tolerance = 1e-3f;
  float epsilon = 1e-5f;
  char* outputFilename = NULL;

  int functionality = -1;
  int folder = 5;
  while (1) {
    static struct option longOptions[] = {
      {"gaussian", no_argument, &kType, GAUSSIAN},
      {"polynomial", no_argument, &kType, POLYNOMIAL},
      {"sigmoid", no_argument, &kType, SIGMOID},
      {"linear", no_argument, &kType, LINEAR},
      {"cost", required_argument, 0, 'c'},
      {"heuristic", required_argument, 0, 'h'},
      {"tolerance", required_argument, 0, 't'},
      {"epsilon", required_argument, 0, 'e'},
      {"output", required_argument, 0, 'o'},
      {"version", no_argument, 0, 'v'},
      {"help", no_argument, 0, 'f'}
    };
    int optionIndex = 0;
    currentOption = getopt_long(argc, (char *const*)argv, "c:h:t:e:o:a:r:d:g:v:f:x:b:", longOptions, &optionIndex);
    if (currentOption == -1) {
      break;
    }
    int method = 3;
    switch (currentOption) {
    case 0:
      break;
    case 'v':
      printf("GPUSVM version %1.1f\n", VERSION);
      return(0);
    case 'f':
      printHelp();
      return(0);
    case 'c':
      sscanf(optarg, "%f", &cost);
      break;
    case 'h':
      sscanf(optarg, "%i", &method);
      switch (method) {
      case 0:
        heuristicMethod = FIRSTORDER;
        break;
      case 1:
        heuristicMethod = SECONDORDER;
        break;
      case 2:
        heuristicMethod = RANDOM;
        break;
      case 3:
        heuristicMethod = ADAPTIVE;
        break;
      }
      break;
    case 't':
      sscanf(optarg, "%f", &tolerance);
      break;
    case 'e':
      sscanf(optarg, "%e", &epsilon);
      break;
    case 'o':
      outputFilename = (char*)malloc(strlen(optarg));
      strcpy(outputFilename, optarg);
      break;
    case 'a':
      sscanf(optarg, "%f", &parameterA);
      parameterASet = true;
      break;
    case 'r':
      sscanf(optarg, "%f", &parameterB);
      parameterBSet = true;
      break;
    case 'd':
      sscanf(optarg, "%f", &parameterC);
      parameterCSet = true;
      break;
    case 'g':
      sscanf(optarg, "%f", &parameterA);
      parameterA = -parameterA;
      parameterASet = true;
      break;
    case 'x':
      sscanf(optarg, "%d", &functionality);
      break;
    case 'b':
      sscanf(optarg, "%d", &folder);
      break;
    case '?':
      break;
    default:
      abort();
      break;
    }
  }

  int numSVM = 1;
  if (optind > argc - 1) {
    printHelp();
    return 0;
  } else {
    numSVM = argc - optind;

    // update numSVM
    switch(functionality) {
      case 1:
        if(numSVM % 2 == 0)
          numSVM = numSVM / 2;
        else {
          printHelp();
          return 0;
        }
        break;
      case 3:
        numSVM -= 1;
        if(numSVM < 1) {
          printHelp();
          return 0;
        }
        break;
      case 4:
        if((numSVM-1) % 2 == 0) {
          numSVM = (numSVM-1) / 2;
          if(numSVM < 1) {
            printHelp();
            return 0;
          }
        } else {
          printHelp();
          return 0;
        }
        break;
      default:
        break;
    }
  }

  if(functionality == -1) {
    const char* trainingFilename = argv[optind];
    
    if (outputFilename == NULL) {
      int inputNameLength = strlen(trainingFilename);
      outputFilename = (char*)malloc(sizeof(char)*(inputNameLength + 5));
      strncpy(outputFilename, trainingFilename, inputNameLength + 4);
      char* period = strrchr(outputFilename, '.');
      if (period == NULL) {
        period = outputFilename + inputNameLength;
      }
      strncpy(period, ".mdl\0", 5);
    }
    
    int nPoints;
    int nDimension;
    float* data;
    float* transposedData;
    float* labels;
    readSvm(trainingFilename, &data, &labels, &nPoints, &nDimension, &transposedData);
    printf("Input data found: %d points, %d dimension\n", nPoints, nDimension);
    
    float* alpha;
    Kernel_params kp;
    if (kType == LINEAR) {
      printf("Linear kernel\n");
      kp.kernel_type = "linear";
    } else if (kType == POLYNOMIAL) {
      if (!(parameterCSet)) {
        parameterC = 3.0f;
      }
      if (!(parameterASet)) {
        parameterA = 1.0/nPoints;
      }
      if (!(parameterBSet)) {
        parameterB = 0.0f;
      }
      //printf("Polynomial kernel: a = %f, r = %f, d = %f\n", parameterA, parameterB, parameterC);
      if ((parameterA <= 0) || (parameterB < 0) || (parameterC < 1.0)) {
        printf("Invalid parameters\n");
        exit(1);
      }
      kp.kernel_type = "polynomial";
      kp.gamma = parameterA;
      kp.coef0 = parameterB;
      kp.degree = (int)parameterC;
    } else if (kType == GAUSSIAN) {
      if (!(parameterASet)) {
        parameterA = 1.0/nDimension;
      } else {
        parameterA = -parameterA;
      }
      //printf("Gaussian kernel: gamma = %f\n", parameterA);
      if (parameterA < 0) {
        printf("Invalid parameters\n");
        exit(1);
      }
      kp.kernel_type = "rbf";
      kp.gamma = parameterA;
    } else if (kType == SIGMOID) {
      if (!(parameterASet)) {
        parameterA = 1.0/nPoints;
      }
      if (!(parameterBSet)) {
        parameterB = 0.0f;
      }
      //printf("Sigmoid kernel: a = %f, r = %f\n", parameterA, parameterB);
      if ((parameterA <= 0) || (parameterB < 0)) {
        printf("Invalid Parameters\n");
        exit(1);
      }
      kp.kernel_type = "sigmoid";
      kp.gamma = parameterA;
      kp.coef0 = parameterB;
    }

    struct timeval start;
    gettimeofday(&start, 0);
    performTraining(data, nPoints, nDimension, labels, &alpha, &kp, cost, heuristicMethod, epsilon, tolerance, transposedData);

    struct timeval finish;
    gettimeofday(&finish, 0);
    float trainingTime = (float)(finish.tv_sec - start.tv_sec) + ((float)(finish.tv_usec - start.tv_usec)) * 1e-6;
    
    printf("Training time : %f seconds\n", trainingTime);
    printModel(outputFilename, kp, alpha, labels, data, nPoints, nDimension, epsilon);

  } else {
    switch(functionality) {
      case 0:
        {
          // train model

          // numSVM = 50000; //set copy of the traning file
          char ** trainingFile = (char **) malloc(sizeof(char *) * numSVM);
          for(int i = 0; i < numSVM; i++) {
            //trainingFile[i] = argv[optind+i];
            trainingFile[i] = argv[optind];
          }
          svmTrainFromFile(numSVM, trainingFile);
        }
        break;
      case 1:
        {
          // predict
          char ** trainingFile = (char **) malloc(sizeof(char *) * numSVM);
          char ** testingFile = (char **) malloc(sizeof(char *) * numSVM);
          for(int i = 0; i < numSVM; i++) {
            trainingFile[i] = argv[optind+i];
            testingFile[i] = argv[optind+numSVM+i];
          }
          svmPredictFromFile(numSVM, trainingFile, testingFile);
        }
        break;
      case 2:
        {
          // cross validation
          char ** trainingFile = (char **) malloc(sizeof(char *) * numSVM);
          for(int i = 0; i < numSVM; i++) {
            trainingFile[i] = argv[optind+i];
          }
          svmCrossValidationFromFile(numSVM, trainingFile, folder, 1.0);
        }
        break;
      case 3:
        {
          // train subset
          char* dataFile = argv[optind];
          char ** subsetFile = (char **) malloc(sizeof(char *) * numSVM);
          for(int i = 0; i < numSVM; i++) {
            subsetFile[i] = argv[optind+i+1];
          }
          svmSubsetTrainFromFile(numSVM, dataFile, subsetFile, 1.0);
        }
        break;
      case 4:
        {
          // predict subset
          char* dataFile = argv[optind];
          char ** subsetFile = (char **) malloc(sizeof(char *) * numSVM);
          char ** testingFile = (char **) malloc(sizeof(char *) * numSVM);
          for(int i = 0; i < numSVM; i++) {
            subsetFile[i] = argv[optind+i+1];
            testingFile[i] = argv[optind+numSVM+i+1];
          }
          svmSubsetPredictFromFile(numSVM, dataFile, subsetFile, testingFile);
        }
        break;
      default:
        printHelp();
        break;
    }

    cout << "Program End." << endl;
  }

  return 0;
}