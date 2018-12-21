################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= svmTrain
CUFILES   := trainingMain.cu svmTrain.cu processData.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= Cache.cpp Controller.cpp splitfeature.cpp
LINKCCFILES := svmIO.cpp
LINKCUFILES := deviceSelect.cu svmClassify.cu svmClassifyKernels.cu

LIB := -lcublas

################################################################################
# Rules and targets

include ../common/common.mk