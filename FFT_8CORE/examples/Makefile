HOME = ..
include $(HOME)/Makefile.inc

EXE = core.exe 

OBJDIR = ./obj

global      = $(OBJDIR)/module_global.o
module_mpi  = $(OBJDIR)/module_mpi_topology.o $(OBJDIR)/module_mpi_subdomain.o
solver	    = $(OBJDIR)/module_solve_poisson.o 
analysis    = $(OBJDIR)/module_post.o 
main 	    = $(OBJDIR)/main.o 

OBJS   	    = $(global) $(module_mpi) $(solver) $(analysis) $(main)

MKL_HOME    = /opt/intel/oneapi/mkl/latest
PaScaL_HOME = ../PaScaL_TDMA

PaScaLINC = -I$(PaScaL_HOME)/include
PaScaLLIB = -L$(PaScaL_HOME)/lib -lpascal_tdma

# FFTWINC = -I$(MKL_HOME)/include/fftw

# # FFTWFLAGS 	=  -qmkl=sequential
# # FFTWFLAGS = -I$(FFTWINC) -L$(FFTWLIB) -lfftw3 -lfftw3_threads

# $(OBJDIR)/%.o: %.f90
# 	$(FC) $(FLAG) -c $^ -o $@ -module $(OBJDIR) -I$(OBJDIR) $(FFTWINC) $(FFTWLIB) ${PaScaLINC} $(FFTWFLAGS)

# all:$(OBJS)
# 	$(FC) $(FLAG) $(FFTWINC) $(FFTWLIB) $(OBJS) -o $(EXE) $(PaScaLINC) $(PaScaLLIB) $(FFTWFLAGS)
# 	mv $(EXE) ../run

# clean:
# 	rm -rf $(OBJDIR); rm -f ../run/$(EXE); rm -rf ../run/data 

# ===== FFTW 경로 (환경에 맞게 수정) =====
FFTW_INC_DIR ?= /opt/intel/oneapi/mkl/2023.2.0/include/fftw
FFTW_LIB_DIR ?= /opt/intel/oneapi/mkl/2023.2.0/lib/intel64
INC_FFTW      = -I$(FFTW_INC_DIR)
LIB_FFTW      = -L$(FFTW_LIB_DIR) -lmkl_rt -lpthread -lm -ldl
# 단정밀이면 -lfftw3f -lfftw3f_threads
# ======================================

$(OBJDIR)/%.o: %.f90
	@mkdir -p $(OBJDIR)
	$(FC) $(FLAG) -c $< -o $@ -module $(OBJDIR) -I$(OBJDIR) $(PaScaLINC) $(INC_FFTW)

all: $(OBJS)
	$(FC) $(FLAG) $(OBJS) -o $(EXE) $(PaScaLLIB) $(LIB_FFTW)
	mv $(EXE) ../run

clean:
	rm -rf $(OBJDIR); rm -f ../run/$(EXE); rm -rf ../run/data
