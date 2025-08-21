#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <fftw3.h>
#include <mpi.h>
#include <complex>
#include "PaScaL_POISSON_FFT.hpp"

void write_fft_result_components(const std::vector<fftw_complex>& complex_array,
                                 const std::string& real_filename,
                                 const std::string& imag_filename,
                                 int z_size,
                                 int y_size,
                                 int x_size) {
    std::ofstream realFile(real_filename);
    std::ofstream imagFile(imag_filename);

    if (!realFile || !imagFile) {
        std::cerr << "Error: Cannot open output files.\n";
        return;
    }

    for (int k = 0; k < z_size; ++k) {
        realFile << "# z = " << k << "\n";
        imagFile << "# z = " << k << "\n";

        for (int j = 0; j < (y_size); ++j) {
            for (int i = 0; i < x_size; ++i) {
                int idx = k * (y_size * x_size) + j * (x_size) + i;
                realFile << std::setw(12) << complex_array[idx][0] << " ";
                imagFile << std::setw(12) << complex_array[idx][1] << " ";
            }
            realFile << "\n";
            imagFile << "\n";
        }

        realFile << "\n";
        imagFile << "\n";
    }

    realFile.close();
    imagFile.close();
}

void write_scalar_components(const std::vector<double>& solution,
                                 const std::string& filename,
                                 int z_size,
                                 int y_size,
                                 int x_size) {
    std::ofstream realFile(filename);

    if (!realFile) {
        std::cerr << "Error: Cannot open output files.\n";
        return;
    }

    for (int k = 0; k < z_size; ++k) {
        realFile << "# z = " << k << "\n";

        for (int j = 0; j < (y_size); ++j) {
            for (int i = 0; i < x_size; ++i) {
                int idx = k * (y_size * x_size) + j * (x_size) + i;
                realFile << std::setw(17) << solution[idx] << " ";
            }
            realFile << "\n";
        }

        realFile << "\n";
    }

    realFile.close();
}

void write_fft_result_x(const std::vector<fftw_complex>& fft_result,
                                 const std::string& real_filename,
                                 const std::string& imag_filename) {
    std::ofstream realFile(real_filename);
    std::ofstream imagFile(imag_filename);

    if (!realFile || !imagFile) {
        std::cerr << "Error: Cannot open output files.\n";
        return;
    }

    for (int k = 0; k < N_z_mpi; ++k) {
        realFile << "# z = " << k << "\n";
        imagFile << "# z = " << k << "\n";

        for (int j = 0; j < N_y_mpi; ++j) {
            for (int i = 0; i < (N_kx); ++i) {
                int idx = IDX3D_xfft(i, j, k);
                realFile << std::setw(12) << fft_result[idx][0] << " ";
                imagFile << std::setw(12) << fft_result[idx][1] << " ";
            }
            realFile << "\n";
            imagFile << "\n";
        }

        realFile << "\n";
        imagFile << "\n";
    }

    realFile.close();
    imagFile.close();
}