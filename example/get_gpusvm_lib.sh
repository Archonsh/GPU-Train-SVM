cd ..
make veryclean
make
cd ./obj/release
mv Cache.cpp_o Cache.o
mv Controller.cpp_o Controller.o
mv processData.cu_o processData.o
mv svmClassifyKernels.cu_o svmClassifyKernels.o
mv svmTrain.cu_o svmTrain.o
mv deviceSelect.cu_o deviceSelect.o
mv svmClassify.cu_o svmClassify.o
mv svmIO.cpp_o svmIO.o
ar rvs libgpusvm.a Cache.o Controller.o processData.o svmClassify.o svmTrain.o deviceSelect.o svmClassifyKernels.o svmIO.o
mv libgpusvm.a ../../example/
cd ../..
cd ./example
