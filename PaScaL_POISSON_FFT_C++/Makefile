# Compiler and complilation flag are included in Makefile.inc
# Compiler and compilation flags are included in Makefile.inc
include ./Makefile.inc

all:
	cd PaScaL_TDMA; make lib
	cd examples; mkdir -p obj; make all
.PHONY: all lib exe clean execlean

all: lib exe

lib:
	cd PaScaL_TDMA; make lib
	$(MAKE) -C PaScaL_TDMA lib

exe:
	cd examples; mkdir -p obj; make all
	mkdir -p examples/obj
	$(MAKE) -C examples all

clean:
	cd PaScaL_TDMA; make clean
	cd examples; make clean
	$(MAKE) -C PaScaL_TDMA clean
	$(MAKE) -C examples clean
	rm -f run/*.o* run/*.e* $(TARGET) output.txt error.txt

execlean:
	cd examples; make clean
	$(MAKE) -C examples clean