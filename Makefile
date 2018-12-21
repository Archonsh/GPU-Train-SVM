all: executables

# do not change the order of "classifyExec" and "trainExec", because "trainExec" needs the object
# files from "classifyExec" for online classification
executables: classifyExec trainExec

trainExec: common
	make -C src/training

classifyExec: common
	make -C src/classification

common:
	make -C src/common

clean:
	@rm -rf obj

veryclean:
	@make clean
	@rm -rf bin
	@rm -rf lib