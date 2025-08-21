#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <chrono>
#include <complex>
#include <mpi.h>
#include <iomanip>
#include <omp.h>
#include "PaScaL_POISSON_FFT.hpp"

int rank;
int size;
int my_floor;
int my_col;
int N_x_mpi;

double b_0;
double b_N;
double a_0;
double c_N;
double a;
double b;
double c;
MPI_Datatype recv_type;

 
void init_mpi(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}

void complex_add(fftw_complex a, fftw_complex b, fftw_complex result) {
    result[0] = a[0] + b[0];
    result[1] = a[1] + b[1];
}

void complex_subtract(double areal,double acomp, fftw_complex b, fftw_complex result) {
    result[0] = areal - b[0];
    result[1] = acomp - b[1];
}

void complex_mult(double areal,double acomp, fftw_complex b, fftw_complex result) {
    result[0] = areal*b[0] - acomp*b[1];
    result[1] = areal*b[1] + acomp*b[0];
}

void complex_divide(fftw_complex a, fftw_complex b, fftw_complex result) {
    double denominator = b[0]*b[0] + b[1]*b[1];
    result[0] = (a[0]*b[0] + a[1]*b[1]) / denominator;
    result[1] = (a[1]*b[0] - a[0]*b[1]) / denominator;
}

void thomas_algorithm(std::vector<double>& recv_data_per_node, std::vector<fftw_complex>& solution_to_Thomas_algorithm, int ix, int recv_data_j, std::vector<fftw_complex>& c_star, std::vector<fftw_complex>& d_star) {

    fftw_complex temp_num, num, temp_denom, denom;
    
    c_star[0][0] = recv_data_per_node[IDX_recv(4, recv_data_j, ix)];
    c_star[0][1] = recv_data_per_node[IDX_recv(5, recv_data_j, ix)];


    for (int floor = 0; floor < p_z - 1; floor++) {
        int i = floor * 2 + 1;
        // c_star[i] = c[i] / (b[i] - a[i] * c_star[i - 1])
        c_star[i][0] = recv_data_per_node[IDX_recv((floor * 12 + 6), recv_data_j, ix)]
                       / (1 - recv_data_per_node[IDX_recv((floor * 12 + 2), recv_data_j, ix)] * c_star[i - 1][0]);
        c_star[i][1] = 0.0;

        c_star[i + 1][0] = recv_data_per_node[IDX_recv(((floor + 1) * 12 + 4), recv_data_j, ix)]
                       / (1 - recv_data_per_node[IDX_recv(((floor + 1) * 12), recv_data_j, ix)] * c_star[i][0]);
        c_star[i + 1][1] = 0.0;

    }
    d_star[0][0] = recv_data_per_node[IDX_recv(8, recv_data_j, ix)];
    d_star[0][1] = recv_data_per_node[IDX_recv(9, recv_data_j, ix)];

    for (int floor = 0; floor < p_z - 1; floor ++) {
        int i = floor * 2 + 1;
        
        // i
        // d_star[i] = (d[i] - a[i] * d_star[i - 1]) / (b[i] - a[i] * c_star[i - 1])

        // NUMERATOR : (d[i] - a[i] * d_star[i - 1])
        // Calculate a[i] * d_star[i - 1] =: temp_num
        complex_mult(recv_data_per_node[IDX_recv((floor * 12 + 2), recv_data_j, ix)], // real part of a[i]
                     recv_data_per_node[IDX_recv((floor * 12 + 3), recv_data_j, ix)], // imag part of a[i]
                     d_star[i - 1], temp_num);

        // Calculate d[i] - temp_num =: num
        complex_subtract(recv_data_per_node[IDX_recv((floor * 12 + 10), recv_data_j, ix)], // real part of d[i]
                         recv_data_per_node[IDX_recv((floor * 12 + 11), recv_data_j, ix)], // imag part of d[i]
                         temp_num, num);

        // Calculate a[i] * c_star[i - 1] =: temp_denom
        complex_mult(recv_data_per_node[IDX_recv((floor * 12 + 2), recv_data_j, ix)], // real part of a[i]
                     recv_data_per_node[IDX_recv((floor * 12 + 3), recv_data_j, ix)], // imag part of a[i]
                     c_star[i - 1], temp_denom);
        
        denom[0] = 1 - temp_denom[0];
        denom[1] = -temp_denom[1];

        complex_divide(num, denom, d_star[i]);

        // i + 1
        complex_mult(recv_data_per_node[IDX_recv(((floor + 1) * 12 + 0), recv_data_j, ix)], // real part of a[i + 1]
                     recv_data_per_node[IDX_recv(((floor + 1) * 12 + 1), recv_data_j, ix)], // imag part of a[i + 1]
                     d_star[i], temp_num);

        // Calculate d[i] - temp_num =: num
        complex_subtract(recv_data_per_node[IDX_recv(((floor + 1) * 12 + 8), recv_data_j, ix)], // real part of d[i]
                         recv_data_per_node[IDX_recv(((floor + 1) * 12 + 9), recv_data_j, ix)], // imag part of d[i]
                         temp_num, num);
        
        // Calculate a[i] * c_star[i - 1] =: temp_denom
        complex_mult(recv_data_per_node[IDX_recv(((floor + 1) * 12 + 0), recv_data_j, ix)], // real part of a[i]
                     recv_data_per_node[IDX_recv(((floor + 1) * 12 + 1), recv_data_j, ix)], // imag part of a[i]
                     c_star[i], temp_denom);
        
        denom[0] = 1 - temp_denom[0];
        denom[1] = -temp_denom[1];

        complex_divide(num, denom, d_star[i + 1]);

    }

    // Last index of d_star
    int floor = p_z - 1;
    int i = 2 * p_z - 1;

    // d_star[i] = (d[i] - a[i] * d_star[i - 1]) / (b[i] - a[i] * c_star[i - 1])
    // NUMERATOR : (d[i] - a[i] * d_star[i - 1])
    // Calculate a[i] * d_star[i - 1] =: temp_num
    complex_mult(recv_data_per_node[IDX_recv(((floor) * 12 + 2), recv_data_j, ix)], // real part of a[i]
                 recv_data_per_node[IDX_recv(((floor) * 12 + 3), recv_data_j, ix)], // imag part of a[i]
                 d_star[2 * p_z - 2], temp_num);
    
    // Calculate d[i] - temp_num =: num
    complex_subtract(recv_data_per_node[IDX_recv((floor * 12 + 10), recv_data_j, ix)], // real part of d[i]
                     recv_data_per_node[IDX_recv((floor * 12 + 11), recv_data_j, ix)], // imag part of d[i]
                     temp_num, num);

    // Calculate a[i] * c_star[i - 1] =: temp_denom
    complex_mult(recv_data_per_node[IDX_recv(((floor) * 12 + 2), recv_data_j, ix)], // real part of a[i]
                 recv_data_per_node[IDX_recv(((floor) * 12 + 3), recv_data_j, ix)], // imag part of a[i]
                 c_star[i - 1], temp_denom);

    denom[0] = 1 - temp_denom[0];
    denom[1] = -temp_denom[1];
    
    complex_divide(num, denom, solution_to_Thomas_algorithm[IDX_TDMA((2 * p_z - 1), ix, recv_data_j)]);

    for (int i = 2 * p_z - 2; i >= 0; i--) {
        complex_mult(c_star[i][0], c_star[i][1], solution_to_Thomas_algorithm[IDX_TDMA((i + 1), ix, recv_data_j)], temp_num);
        complex_subtract(d_star[i][0], d_star[i][1], temp_num, solution_to_Thomas_algorithm[IDX_TDMA(i, ix, recv_data_j)]);
    }
}

void make_modified_TDM(int ix, 
                       int global_ix, 
                       int jy, int recv_floor, 
                       int recv_rank, 
                       int recv_data_j,
                       std::vector<fftw_complex>& fft_final,
                       std::vector<fftw_complex>& d_star, 
                       std::vector<double>& c_star, 
                       std::vector<double>& a_star) {
    
    // printf("ix=%d jy=%d thread=%d\n", ix, jy, omp_get_thread_num());

    b = -2.0 / (dz * dz) + (2 * (cos(2 * pi * static_cast<double>(jy) / N_y) + cos(2 * pi * static_cast<double>(global_ix) / N_x) - 2)) / (dx * dx);

    // Define b_0 and b_N
    if (my_floor == 0) {
        b_0 = -3.0 / (dz * dz) + (2 * (cos(2 * pi * static_cast<double>(jy) / N_y) + cos(2 * pi * static_cast<double>(global_ix) / N_x) - 2)) / (dx * dx);
        a_0 = 0.0;
    } else {
        b_0 = -2.0 / (dz * dz) + (2 * (cos(2 * pi * static_cast<double>(jy) / N_y) + cos(2 * pi * static_cast<double>(global_ix) / N_x) - 2)) / (dx * dx);
        a_0 = 1.0 / (dz * dz);
    }

    if (my_floor == p_z - 1) {
        b_N = -3.0 / (dz * dz) + (2 * (cos(2 * pi * static_cast<double>(jy) / N_y) + cos(2 * pi * static_cast<double>(global_ix) / N_x) - 2)) / (dx * dx);
        c_N = 0.0;
    } else {
        b_N = -2.0 / (dz * dz) + (2 * (cos(2 * pi * static_cast<double>(jy) / N_y) + cos(2 * pi * static_cast<double>(global_ix) / N_x) - 2)) / (dx * dx);
        c_N = 1.0 / (dz * dz);
    }

    double r = 0.0;

    d_star[IDX3D_yfft(ix, jy, 0)][0] = fft_final[IDX3D_yfft(ix, jy, 0)][0] / b_0;
    d_star[IDX3D_yfft(ix, jy, 0)][1] = fft_final[IDX3D_yfft(ix, jy, 0)][1] / b_0;
    c_star[IDX3D_yfft(ix, jy, 0)] = c / b_0;
    a_star[IDX3D_yfft(ix, jy, 0)] = a_0 / b_0;
            
    d_star[IDX3D_yfft(ix, jy, 1)][0] = fft_final[IDX3D_yfft(ix, jy, 1)][0] / b;
    d_star[IDX3D_yfft(ix, jy, 1)][1] = fft_final[IDX3D_yfft(ix, jy, 1)][1] / b;
    c_star[IDX3D_yfft(ix, jy, 1)] = c / b;
    a_star[IDX3D_yfft(ix, jy, 1)] = a / b;

    for (int i = 2; i < N_z_mpi - 1; i++) {
        r = 1.0 / (b - a * c_star[IDX3D_yfft(ix, jy, (i-1))]);
        
        d_star[IDX3D_yfft(ix, jy, i)][0] = r * (fft_final[IDX3D_yfft(ix, jy, i)][0] - a * d_star[IDX3D_yfft(ix, jy, (i-1))][0]);
        d_star[IDX3D_yfft(ix, jy, i)][1] = r * (fft_final[IDX3D_yfft(ix, jy, i)][1] - a * d_star[IDX3D_yfft(ix, jy, (i-1))][1]);
        c_star[IDX3D_yfft(ix, jy, i)] = r * a;
        a_star[IDX3D_yfft(ix, jy, i)] = -r * a * a_star[IDX3D_yfft(ix, jy, (i - 1))];
    }

    // i = N_z_mpi - 1
    r = 1.0 / (b_N - a * c_star[IDX3D_yfft(ix, jy, (N_z_mpi - 2))]);
    d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][0] = r * (fft_final[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][0] - a * d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 2))][0]);
    d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][1] = r * (fft_final[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][1] - a * d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 2))][1]);
    c_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))] = r * c_N;
    a_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))] = - r * a * a_star[IDX3D_yfft(ix, jy, (N_z_mpi - 2))];

    for (int i = N_z_mpi - 3; i >= 1; i--) {
        d_star[IDX3D_yfft(ix, jy, i)][0] = d_star[IDX3D_yfft(ix, jy, i)][0] - c_star[IDX3D_yfft(ix, jy, i)] * d_star[IDX3D_yfft(ix, jy, (i + 1))][0];
        d_star[IDX3D_yfft(ix, jy, i)][1] = d_star[IDX3D_yfft(ix, jy, i)][1] - c_star[IDX3D_yfft(ix, jy, i)] * d_star[IDX3D_yfft(ix, jy, (i + 1))][1];
        a_star[IDX3D_yfft(ix, jy, i)] = a_star[IDX3D_yfft(ix, jy, i)] - c_star[IDX3D_yfft(ix, jy, i)] * a_star[IDX3D_yfft(ix, jy, (i + 1))];
        c_star[IDX3D_yfft(ix, jy, i)] = -(c_star[IDX3D_yfft(ix, jy, i)] * c_star[IDX3D_yfft(ix, jy, (i + 1))]);

    }
            
    r = 1 / (1.0 - a_star[IDX3D_yfft(ix, jy, 1)] * c_star[IDX3D_yfft(ix, jy, 0)]);

    d_star[IDX3D_yfft(ix, jy, 0)][0] = r * (d_star[IDX3D_yfft(ix, jy, 0)][0] - c_star[IDX3D_yfft(ix, jy, 0)] * d_star[IDX3D_yfft(ix, jy, 1)][0]);
    d_star[IDX3D_yfft(ix, jy, 0)][1] = r * (d_star[IDX3D_yfft(ix, jy, 0)][1] - c_star[IDX3D_yfft(ix, jy, 0)] * d_star[IDX3D_yfft(ix, jy, 1)][1]);

    c_star[IDX3D_yfft(ix, jy, 0)] = -r * c_star[IDX3D_yfft(ix, jy, 0)] * c_star[IDX3D_yfft(ix, jy, 1)];
    a_star[IDX3D_yfft(ix, jy, 0)] = r * a_star[IDX3D_yfft(ix, jy, 0)];
}

void make_send_data(int ix, int jy, int send_data_j, std::vector<double>& send_data, std::vector<fftw_complex>& d_star, std::vector<double>& c_star, std::vector<double>& a_star) {
    send_data[IDX_send(0, send_data_j, ix)] = a_star[IDX3D_yfft(ix, jy, 0)];
    send_data[IDX_send(1, send_data_j, ix)] = 0.0;

    send_data[IDX_send(2, send_data_j, ix)] = a_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))];
    send_data[IDX_send(3, send_data_j, ix)] = 0.0;

    send_data[IDX_send(4, send_data_j, ix)] = c_star[IDX3D_yfft(ix, jy, 0)];
    send_data[IDX_send(5, send_data_j, ix)] = 0.0;

    send_data[IDX_send(6, send_data_j, ix)] = c_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))];
    send_data[IDX_send(7, send_data_j, ix)] = 0.0;

    send_data[IDX_send(8, send_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, 0)][0];
    send_data[IDX_send(9, send_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, 0)][1];

    send_data[IDX_send(10, send_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][0];
    send_data[IDX_send(11, send_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][1];
}

void make_recv_data(int ix, int jy, int recv_data_j, std::vector<double>& recv_data_per_node, std::vector<fftw_complex>& d_star, std::vector<double>& c_star, std::vector<double>& a_star) {
    recv_data_per_node[IDX_recv((12 * my_floor + 0), recv_data_j, ix)] = a_star[IDX3D_yfft(ix, jy, 0)];
    recv_data_per_node[IDX_recv((12 * my_floor + 1), recv_data_j, ix)] = 0.0;
    
    recv_data_per_node[IDX_recv((12 * my_floor + 2), recv_data_j, ix)] = a_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))];
    recv_data_per_node[IDX_recv((12 * my_floor + 3), recv_data_j, ix)] = 0.0;
    
    recv_data_per_node[IDX_recv((12 * my_floor + 4), recv_data_j, ix)] = c_star[IDX3D_yfft(ix, jy, 0)];
    recv_data_per_node[IDX_recv((12 * my_floor + 5), recv_data_j, ix)] = 0.0;

    recv_data_per_node[IDX_recv((12 * my_floor + 6), recv_data_j, ix)] = c_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))];
    recv_data_per_node[IDX_recv((12 * my_floor + 7), recv_data_j, ix)] = 0.0;

    recv_data_per_node[IDX_recv((12 * my_floor + 8), recv_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, 0)][0];
    recv_data_per_node[IDX_recv((12 * my_floor + 9), recv_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, 0)][1];

    recv_data_per_node[IDX_recv((12 * my_floor + 10), recv_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][0];
    recv_data_per_node[IDX_recv((12 * my_floor + 11), recv_data_j, ix)] = d_star[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][1];
    
}

void PaScaL_TDMA(std::vector<fftw_complex>& fft_final, std::vector<fftw_complex>& d_star, std::vector<double>& recv_data_per_node, std::vector<fftw_complex>& solution_to_Thomas_algorithm, std::vector<fftw_complex>& c_star_single, std::vector<fftw_complex>& d_star_single) {
    
    MPI_Comm new_comm;
    int color = rank % p_y;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &new_comm);
    
    std::vector<double> send_data(12 * N_x_mpi * (N_y / p_z) * (p_z - 1));
    std::vector<double> c_star(N_z_mpi * N_x_mpi * N_y, 0.0);
    std::vector<double> a_star(N_z_mpi * N_x_mpi * N_y, 0.0);
    a = 1.0 / (dz * dz);
    c = 1.0 / (dz * dz);

    int jy_start = (my_floor) * (N_y / p_z);
    int jy_end = (my_floor + 1) * (N_y / p_z);

    using clock    = std::chrono::high_resolution_clock;
    auto t_global0 = clock::now();
    
    // Make Send / Recv data
    #pragma omp parallel for collapse(2)
    for (int ix = 0; ix < N_x_mpi; ix++) {
        // #pragma omp parallel for
        for (int jy = 0; jy < N_y; jy++) {
            make_modified_TDM(ix,
                              ix + original_N_x_mpi * my_col, 
                              jy, 
                              (int) ((jy) / (N_y / p_z)), 
                              (int) ((jy) / (N_y / p_z)) * p_y + my_col, 
                              jy % (N_y / p_z), fft_final, 
                              d_star, c_star, a_star);
            printf("ix=%d jy=%d thread=%d\n", ix, jy, omp_get_thread_num());
            if (jy < jy_start) {
                make_send_data(ix, jy, jy, send_data, d_star, c_star, a_star);
            }
            else if (jy < jy_end) {
                make_recv_data(ix, jy, jy % (N_y / p_z), recv_data_per_node, d_star, c_star, a_star);
            }
            else {
                make_send_data(ix, jy, jy - (N_y / p_z), send_data, d_star, c_star, a_star);
            }
        }
    }
    // auto t_make_0 = clock::now();
    // std::chrono::duration<double> elapsed_make_0 = t_make_0 - t_global0;
    // printf("Rank %8d: makedata 1: %8.15f seconds\n", rank, elapsed_make_0.count());
    // for (int r = 0; r < size; ++r) {
    //     if (rank == r) {
    //         std::string real_filename = "omprecv_rank" + std::to_string(rank) + ".txt";
    //         write_scalar_components(recv_data_per_node, real_filename, 12 * p_z, N_y / p_z, N_x_mpi);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
    // for (int r = 0; r < size; ++r) {
    //     if (rank == r) {
    //         std::string real_filename = "ompsend_rank" + std::to_string(rank) + ".txt";
    //         write_scalar_components(send_data, real_filename, 12, (N_y / p_z) * (p_z - 1), N_x_mpi);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // for (int ix = 0; ix < N_x_mpi; ix++) {
    //     for (int jy = 0; jy < jy_start; jy ++) {
    //         int global_ix = ix + original_N_x_mpi * my_col;
    //         int recv_floor = (int) ((jy) / (N_y / p_z));
    //         int recv_rank = recv_floor * p_y + my_col;
    //         int recv_data_j = jy % (N_y / p_z);
    //         make_modified_TDM(ix, global_ix, jy, recv_floor, recv_rank, recv_data_j, fft_final, d_star,c_star,a_star);
    //         make_send_data(ix, jy, jy, send_data, d_star, c_star, a_star);
    //     }
        
    //     for (int jy = jy_start; jy < jy_end; jy++) {
    //         int global_ix = ix + original_N_x_mpi * my_col;
    //         int recv_floor = (int) ((jy) / (N_y / p_z));
    //         int recv_rank = recv_floor * p_y + my_col;
    //         int recv_data_j = jy % (N_y / p_z);
    //         make_modified_TDM(ix, global_ix, jy, recv_floor, recv_rank, recv_data_j, fft_final, d_star,c_star,a_star);
    //         make_recv_data(ix, jy, recv_data_j, recv_data_per_node, d_star, c_star, a_star);

    //     }
    //     for (int jy = jy_end; jy < N_y; jy++) {
    //         int global_ix = ix + original_N_x_mpi * my_col;
    //         int recv_floor = (int) ((jy) / (N_y / p_z));
    //         int recv_rank = recv_floor * p_y + my_col;
    //         int recv_data_j = jy % (N_y / p_z);
    //         make_modified_TDM(ix, global_ix, jy, recv_floor, recv_rank, recv_data_j, fft_final, d_star,c_star,a_star);
    //         make_send_data(ix, jy, jy - (N_y / p_z), send_data, d_star, c_star, a_star);
    //     }
    // }
    
    std::vector<MPI_Datatype> send_types(p_z, MPI_DATATYPE_NULL);
    std::vector<MPI_Datatype> recv_types(p_z, MPI_DATATYPE_NULL);

    int send_sizes[3] = {12, (N_y / p_z) * (p_z - 1), N_x_mpi};
    int recv_sizes[3] = {12 * p_z, N_y / p_z, N_x_mpi};
    int subsize[3] = {12, N_y / p_z, N_x_mpi};
    int starts_send[3] = {0, 0, 0};
    int starts_recv[3] = {0, 0, 0};

    
    for (int floor = 0; floor < p_z; floor ++) {
        if (floor == my_floor) {
            continue;
        }

        starts_send[1] = (floor < my_floor ? floor : floor - 1) * (N_y / p_z);
        MPI_Type_create_subarray(
            3,                            // 차원 수
            send_sizes,                   // 전체 배열 크기 (전역 배열)
            subsize,                      // 내가 보내거나 받을 블럭의 크기
            starts_send,                  // 내가 보내거나 받을 블럭의 시작 위치 (원래 배열 기준)
            MPI_ORDER_C,                  // 메모리 순서 (MPI_ORDER_C = C 배열)
            MPI_DOUBLE,                   // 원소 타입 
            &send_types[floor]);          // 결과 타입 (이 안에 서브배열 구조가 저장됨)
        
        MPI_Type_commit(&send_types[floor]);

        int recv_starts[3] = {12 * floor, 0, 0};

        MPI_Type_create_subarray(
            3,                            // 차원 수
            recv_sizes,                   // 전체 배열 크기 (전역 배열)
            subsize,                      // 내가 보내거나 받을 블럭의 크기
            recv_starts,                  // 내가 보내거나 받을 블럭의 시작 위치 (원래 배열 기준)
            MPI_ORDER_C,                  // 메모리 순서 (MPI_ORDER_C = C 배열)
            MPI_DOUBLE,                   // 원소 타입 
            &recv_types[floor]);          // 결과 타입 (이 안에 서브배열 구조가 저장됨)
        
        MPI_Type_commit(&recv_types[floor]);
    }

    MPI_Request requests_coeff[2 * (p_z - 1)];
    int count_coeff = 0;

    auto t_make_1 = clock::now();
    std::chrono::duration<double> elapsed_make = t_make_1 - t_global0;
    printf("Rank %8d: makedata: %8.15f seconds\n", rank, elapsed_make.count());

    for (int other_rank = 0; other_rank < p_z; ++other_rank) {
        if (other_rank == my_floor) continue;

        MPI_Isend(send_data.data(), 1, send_types[other_rank], other_rank, my_floor, new_comm, &requests_coeff[count_coeff++]);
        MPI_Irecv(recv_data_per_node.data(), 1, recv_types[other_rank], other_rank, other_rank, new_comm, &requests_coeff[count_coeff++]);
    }
    MPI_Waitall(count_coeff, requests_coeff, MPI_STATUSES_IGNORE);

    
    
    for (int other_rank = 0; other_rank < p_z; ++other_rank) {
        if (send_types[other_rank] != MPI_DATATYPE_NULL) MPI_Type_free(&send_types[other_rank]);
        if (recv_types[other_rank] != MPI_DATATYPE_NULL) MPI_Type_free(&recv_types[other_rank]);
    }
    auto t_comm = clock::now();
    std::chrono::duration<double> elapsed_comm = t_comm - t_make_1;
    printf("Rank %8d: first comm %8.15f seconds\n", rank, elapsed_comm.count());
    
    auto t_comm2_start = clock::now();

    // Solve single Thomas algorithm
    for (int ix = 0; ix < N_x_mpi; ix++) {
        for (int recv_data_j = 0; recv_data_j < (N_y / p_z); recv_data_j++) {
            thomas_algorithm(recv_data_per_node, solution_to_Thomas_algorithm, ix, recv_data_j, c_star_single, d_star_single);
        }
    }

    std::vector<double> lower_floor(p_z * 2 * N_x_mpi * (N_y / p_z));
    std::vector<double> upper_floor(p_z * 2 * N_x_mpi * (N_y / p_z));
    std::vector<double> recv_lower_floor(p_z * 2 * N_x_mpi * (N_y / p_z));
    std::vector<double> recv_upper_floor(p_z * 2 * N_x_mpi * (N_y / p_z));

    int req_count = 0;
    MPI_Request requests[8 * p_z];
    
    for (int floor = 0; floor < p_z; floor++) {
        int z_index = 2 * my_floor;
        if (floor == my_floor) {
            for (int i = 0; i < N_x_mpi; i++) {
                for (int j = 0; j < (N_y / p_z); j++) {
                    int original_j = j + (N_y / p_z) * my_floor;

                    fft_final[IDX3D_yfft(i, original_j, 0)][0] = solution_to_Thomas_algorithm[IDX_TDMA((floor * 2), i, j)][0];
                    fft_final[IDX3D_yfft(i, original_j, 0)][1] = solution_to_Thomas_algorithm[IDX_TDMA((floor * 2), i, j)][1];
                    fft_final[IDX3D_yfft(i, original_j, (N_z_mpi - 1))][0] = solution_to_Thomas_algorithm[IDX_TDMA((floor * 2 + 1), i, j)][0];
                    fft_final[IDX3D_yfft(i, original_j, (N_z_mpi - 1))][1] = solution_to_Thomas_algorithm[IDX_TDMA((floor * 2 + 1), i, j)][1];
                }
            }
        } else {
            int dest = floor * p_y + my_col;
            int global_idx = floor * 2 * N_x_mpi * (N_y / p_z);
            int their_rank = floor * p_y + my_col;

            for (int i = 0; i < N_x_mpi * (N_y / p_z); i++) {
                lower_floor[global_idx + i * 2] = solution_to_Thomas_algorithm[floor * 2 * (N_x_mpi * (N_y / p_z)) + i][0];
                lower_floor[global_idx + i * 2 + 1] = solution_to_Thomas_algorithm[floor * 2 * (N_x_mpi * (N_y / p_z)) + i][1];
            }
            // 윗층, tag = plus
            for (int i = 0; i < N_x_mpi * (N_y / p_z); i++) {
                upper_floor[global_idx + i * 2] = solution_to_Thomas_algorithm[(1 + floor * 2) * (N_x_mpi * (N_y / p_z)) + i][0];
                upper_floor[global_idx + i * 2 + 1] = solution_to_Thomas_algorithm[(1 + floor * 2) * (N_x_mpi * (N_y / p_z)) + i][1];
            }
            MPI_Isend(&lower_floor[global_idx], 2 * N_x_mpi * (N_y / p_z), MPI_DOUBLE, dest / p_y, size + rank, new_comm, &requests[req_count++]);
            MPI_Isend(&upper_floor[global_idx], 2 * N_x_mpi * (N_y / p_z), MPI_DOUBLE, dest / p_y, rank, new_comm, &requests[req_count++]);
            MPI_Irecv(&recv_lower_floor[global_idx], 2 * N_x_mpi * (N_y / p_z), MPI_DOUBLE, their_rank / p_y, size + their_rank, new_comm, &requests[req_count++]);
            MPI_Irecv(&recv_upper_floor[global_idx], 2 * N_x_mpi * (N_y / p_z), MPI_DOUBLE, their_rank / p_y, their_rank, new_comm, &requests[req_count++]);
        }
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    auto t_comm2_end = clock::now();
    std::chrono::duration<double> elapsed_comm2 = t_comm2_end - t_comm2_start;
    printf("Rank %8d: Second comm %8.15f seconds\n", rank, elapsed_comm2.count());
    

    for (int floor = 0; floor < p_z; floor++) {
        if (floor != my_floor) {
            int global_idx = floor * 2 * N_x_mpi * (N_y / p_z);
            for (int i = 0; i < N_x_mpi * (N_y / p_z); i++) {
                fft_final[IDX3D_yfft(0, (floor * (N_y / p_z)), 0) + i][0] = recv_lower_floor[global_idx + i * 2];
                fft_final[IDX3D_yfft(0, (floor * (N_y / p_z)), 0) + i][1] = recv_lower_floor[global_idx + i * 2 + 1];
            }
            
            // MPI_Irecv(&recv_upper_floor[global_idx], 2 * N_x_mpi * (N_y / p_z), MPI_DOUBLE, their_rank, their_rank, MPI_COMM_WORLD, &requests[req_count++]);
            for (int i = 0; i < N_x_mpi * (N_y / p_z); i++) {
                fft_final[IDX3D_yfft(0, (floor * (N_y / p_z)), (N_z_mpi - 1)) + i][0] = recv_upper_floor[global_idx + i * 2];
                fft_final[IDX3D_yfft(0, (floor * (N_y / p_z)), (N_z_mpi - 1)) + i][1] = recv_upper_floor[global_idx + i * 2 + 1];
            }
        }
    }

    //************************************************************************************************************************************************************************//
    for (int ix = 0; ix < N_x_mpi; ix++) {
        for (int jy = 0; jy < N_y; jy ++) {
            for (int k = 1; k < N_z_mpi - 1; k++) {
                int idx = IDX3D_yfft(ix, jy, k);
                // solution[(i + 1) * (N_y_mpi + 2) + j + 1] = d_star[i] - a_star[i] * recv_update[0] - c_star[i] * recv_update[1];
                fft_final[idx][0] = d_star[IDX3D_yfft(ix, jy, k)][0] - a_star[IDX3D_yfft(ix, jy, k)] * fft_final[IDX3D_yfft(ix, jy, 0)][0] - c_star[IDX3D_yfft(ix, jy, k)] * fft_final[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][0];
                fft_final[idx][1] = d_star[IDX3D_yfft(ix, jy, k)][1] - a_star[IDX3D_yfft(ix, jy, k)] * fft_final[IDX3D_yfft(ix, jy, 0)][1] - c_star[IDX3D_yfft(ix, jy, k)] * fft_final[IDX3D_yfft(ix, jy, (N_z_mpi - 1))][1];
            }
        }
    }
}