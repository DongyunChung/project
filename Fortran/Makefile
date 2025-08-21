# Compiler and complilation flag are included in Makefile.inc
include ./Makefile.inc

all:
	cd PaScaL_TDMA; make lib
	cd examples; mkdir -p obj; make all

lib:
	cd PaScaL_TDMA; make lib

exe:
	cd examples; mkdir -p obj; make all

clean:
	cd PaScaL_TDMA; make clean
	cd examples; make clean
	rm -f run/*.o* run/*.e* $(TARGET) output.txt error.txt

execlean:
	cd examples; make clean
