#include <iostream>
#include <vector>
#include <cmath>
#include <vector>
#include <fftw3.h>
#include <complex>
#include <mpi.h>
#include "PaScaL_POISSON_FFT.hpp"

void fill_source_array(std::vector<double>& source_array) {
    int global_k = my_floor * N_z_mpi;
    int global_j = my_col * N_y_mpi;

    for (int k = 0; k < N_z_mpi; ++k) {
        for (int j = 0; j < N_y_mpi; ++j) {
            for (int i = 0; i < N_x; ++i) {

                double x = dx * (i + 0.5);
                double y = dy * ((j + global_j) + 0.5);
                double z = dz * ((k + global_k) + 0.5);
                
                source_array[IDX3D(i, j, k)] =  -9 * pi * pi * sin(2 * pi * x) * sin(2 * pi * y) * sin(pi * z);
            }
        }
    }
}

void fill_analytic_solution(std::vector<double>& analytic_solution) {
    int global_k = my_floor * N_z_mpi;
    int global_j = my_col * N_y_mpi;

    for (int k = 0; k < N_z_mpi; ++k) {
        for (int j = 0; j < N_y_mpi; ++j) {
            for (int i = 0; i < N_x; ++i) {

                double x = dx * (i + 0.5);
                double y = dy * ((j + global_j) + 0.5);
                double z = dz * ((k + global_k) + 0.5);
                
                analytic_solution[IDX3D(i, j, k)] =  sin(2 * pi * x) * sin(2 * pi * y) * sin(pi * z);
            }
        }
    }
}