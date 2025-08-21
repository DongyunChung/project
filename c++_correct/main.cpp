#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <fftw3.h>
#include <omp.h>
#include <mpi.h>
#include <complex>
#include "PaScaL_POISSON_FFT.hpp"

int main(int argc, char **argv) {
    using clock    = std::chrono::high_resolution_clock;
    auto t_global0 = clock::now();
    
    init_mpi(&argc, &argv);
    omp_set_num_threads(2);

    my_floor = rank / p_y;
    my_col = rank % p_y;
    N_x_mpi = (N_x /(2 * p_y)) + (my_col / (p_y - 1));

    MPI_Type_vector(6, 1, (N_y / p_z) * N_x_mpi, MPI_DOUBLE, &recv_type);
    MPI_Type_commit(&recv_type);
    
    //Declare Status
    MPI_Status status;

    // Fill source array and analytic solution
    std::vector<double> source_array(N_y_mpi * N_z_mpi * N_x, 0.0);
    
    fill_source_array(source_array);
    
    // FFT output per slice: N_x x (N_y / 2 + 1)
    std::vector<fftw_complex> fft_result(N_y_mpi * N_z_mpi * (N_x / 2 + 1));
    std::vector<fftw_complex> fft_y_pencil(N_z_mpi * N_x_mpi * N_y);
    std::vector<fftw_complex> fft_final(N_z_mpi * N_x_mpi * N_y);
    
    auto t_fft_0 = clock::now();
    fft_x_direction(source_array.data(), fft_result.data());
    MPI_Request xy_req;
    MPI_Request yx_req;
    transpose_xy_pencil_corrected(
        fft_result.data(),   // send_buf
        fft_y_pencil.data(), // recv_buf
        N_x_mpi,
        my_col,
        MPI_COMM_WORLD,
        &xy_req
    );
    fft_y_direction(fft_y_pencil, fft_final, N_x_mpi);
    auto t_fft_1 = clock::now();
    std::chrono::duration<double> elapsed_fft = t_fft_1 - t_fft_0;
    printf("Rank %8d: FFT %8.15f seconds\n", rank, elapsed_fft.count());
    

    // for (int r = 0; r < size; ++r) {
    //     if (rank == r) {
    //         printf("Rank %8d, ffty over!\n", rank);
    //         fflush(stdout);
    //         std::string real_filename = "RHS_real_tdma_rank" + std::to_string(rank) + ".txt";
    //         std::string imag_filename = "RHS_imag_tdma_rank" + std::to_string(rank) + ".txt";
    //         write_fft_result_x(fft_result, real_filename, imag_filename);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    
    std::vector<fftw_complex> d_star(N_z_mpi * N_x_mpi * N_y);

    // d_star[IDX3D_yfft(N_x_mpi - 1, N_y-1, N_z_mpi - 1)][1] = 0;
    // std::vector<fftw_complex> recv_data_per_node(6 * p_z * N_x_mpi * (N_y / p_z));
    std::vector<double> recv_data_per_node(12 * N_x_mpi * (N_y));
    std::vector<fftw_complex> solution_to_Thomas_algorithm(2 * (p_z) * N_x_mpi * (N_y / p_z));
    std::vector<fftw_complex> c_star_single(2 * p_z);
    std::vector<fftw_complex> d_star_single(2 * p_z);

    auto t_TDMA_0 = clock::now();
    PaScaL_TDMA(fft_final, d_star, recv_data_per_node, solution_to_Thomas_algorithm, c_star_single, d_star_single);
    auto t_TDMA_1 = clock::now();
    std::chrono::duration<double> elapsed_TDMA = t_TDMA_1 - t_TDMA_0;
    printf("Rank %8d: TDMA %8.15f seconds\n", rank, elapsed_TDMA.count());
    // for (int r = 0; r < size; ++r) {
    //     if (rank == r) {
    //         std::string real_filename = "ffty_real_rank" + std::to_string(rank) + ".txt";
    //         std::string imag_filename = "ffty_imag_rank" + std::to_string(rank) + ".txt";
    //         write_fft_result_components(fft_final, real_filename, imag_filename);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // for (int r = 0; r < size; ++r) {
    //     if (rank == r) {
    //         std::string real_filename = "solution_rank" + std::to_string(rank) + ".txt";
    //         write_scalar_components(source_array, real_filename);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    auto t_ifft_0 = clock::now();
    ifft_y_direction(fft_final, fft_final, N_x_mpi);    
    transpose_yx_pencil_corrected(
        fft_final.data(),   // send_buf
        fft_result.data(), // recv_buf
        N_x_mpi,
        my_col,
        MPI_COMM_WORLD,
        &yx_req
    );
    ifft_x_direction(fft_result.data(), source_array.data());
    auto t_ifft_1 = clock::now();
    std::chrono::duration<double> elapsed_ifft = t_ifft_1 - t_ifft_0;
    printf("Rank %8d: IFFT %8.15f seconds\n", rank, elapsed_ifft.count());
    std::vector<double> analytic_solution(N_y_mpi * N_z_mpi * N_x, 0.0);
    fill_analytic_solution(analytic_solution);
    double local_error = 0;
    double global_error = 0;
    for (int k = 0; k < N_z_mpi; k++) {
        for (int j = 0; j < N_y_mpi; j++) {
            for (int i = 0; i < N_x; i++) {
                local_error += (analytic_solution[IDX3D(i,j,k)] - source_array[IDX3D(i,j,k)]) * (analytic_solution[IDX3D(i,j,k)] - source_array[IDX3D(i,j,k)]);
            }
        }
    }
    MPI_Reduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Global L2 error = %8.17f\n", sqrt(global_error/ (N_x * N_y * N_z)) );
    }
    
    MPI_Type_free(&recv_type);
    MPI_Finalize();
    fftw_cleanup();

    auto t_global1 = clock::now();
    std::chrono::duration<double> elapsed = t_global1 - t_global0;


    std::cout << std::fixed << std::setprecision(12)
                  << "Total wall-clock time = " << elapsed.count() << "  seconds\n";

    return 0;
}
