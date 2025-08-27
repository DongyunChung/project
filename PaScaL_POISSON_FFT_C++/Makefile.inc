# Main compiler with MPI. Use mpiifort for intel MPI.
FC = mpiifort
# MPI compiler wrappers. Override by setting FC/CXX in the environment.
FC  ?= mpifort
CXX ?= mpicxx

# Complie flag

# For debugging
#FLAG = -O0 -g -fcheck=all --warn-all
# FLAG = -O0 -g -fPIC -check all -warn all -WB

# For moderate debugging
#FLAG = -O1 -fPIC -fp-model=precise -check bounds

# For optimized run 
#FLAG = -O3 -fPIC -fp-model=precise -xMIC-AVX512
FLAG = -O3 -fPIC -fp-model=precise -xCORE-AVX2 -assume byterecl
# Start with any user-provided flags but drop unsupported variants from some
# toolchains (e.g. nvc++ uses -fPIC, not -fPI).
FFLAGS   := $(filter-out -fPI,$(FFLAGS))
CXXFLAGS := $(filter-out -fPI,$(CXXFLAGS))

# Base optimisation flags
FFLAGS   += -O3 -fPIC
CXXFLAGS += -O3 -fPIC -std=c++17

# Uncomment for extra debugging
#FFLAGS  += -g -fcheck=all -Wall
#CXXFLAGS += -g -Wall -Wextra