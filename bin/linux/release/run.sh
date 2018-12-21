files=""
for i in {1..3000}
do
        files=$files"./mnist/mnist1 "
done
files="-x 0 "$files
./svmTrain $files
