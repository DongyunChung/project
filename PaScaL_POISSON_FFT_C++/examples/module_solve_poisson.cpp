//======================================================================================================================
//> @file        module_solve_pressure.f90
//> @brief       This file contains a module of pressure solver for PaScaL_TCS.
//>              MPI communication.
//> @author      
//>              - Kiha Kim (k-kiha@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
//>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
//>              - Jung-Il Choi (jic@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
//>
//> @date        October 2022
//> @version     1.0
//> @par         Copyright
//>              Copyright (c) 2022 Kiha Kim and Jung-Il choi, Yonsei University and 
//>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
//> @par         License     
//>              This project is release under the terms of the MIT License (see LICENSE in )
//======================================================================================================================

//>
//> @brief       Module for pressure solver
//> @details     This module solves (incremental) pressure Poisson equation and update pressure fileds.
//>              It uses 2D FFT in x-z directions to decouple the discretized equation and and 
//>              solve the remaining equation using 1D PaScaL_TDMA.
//>
#include "modules.hpp"
#include "pascal_tdma.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <fftw3.h>

namespace mpi_poisson {

using namespace mpi_subdomain;
using namespace global;

std::vector<double> P;
std::vector<double> PRHS;
std::vector<double> exact_sol;
std::vector<double> dzk, dxk1, dxk2;

namespace {
inline std::size_t idx_p(int i, int j, int k) {
    return (static_cast<std::size_t>(k + 1) * (n2msub + 2) + (j + 1)) * (n1msub + 2) + (i + 1);
}
inline std::size_t idx_rhs(int i, int j, int k) {
    return (static_cast<std::size_t>(k) * n2msub + j) * n1msub + i;
}
}

void mpi_poisson_allocation() {
    P.assign((n1sub + 1) * (n2sub + 1) * (n3sub + 1), 0.0);
    exact_sol.assign((n1sub + 1) * (n2sub + 1) * (n3sub + 1), 0.0);
}

void mpi_poisson_clean() {
    std::vector<double>().swap(P);
    std::vector<double>().swap(PRHS);
    std::vector<double>().swap(exact_sol);
    std::vector<double>().swap(dzk);
    std::vector<double>().swap(dxk1);
    std::vector<double>().swap(dxk2);
}

void mpi_poisson_wave_number() {
    double ddx = L1 / static_cast<double>(n1m);
    double ddz = L3 / static_cast<double>(n3m);

    dzk.resize(n3m);
    dxk1.resize(h1pKsub);
    dxk2.resize(h1psub_Ksub);


    // printf("Rank %8d:\n", mpi_topology::myrank);
    // printf("h1psub_Ksub: %8d\n", h1psub_Ksub);
    // printf("h1psub_Ksub: %8d\n", h1psub_Ksub);
    // printf("h1pKsub_ista: %8d\n", h1pKsub_ista);
    for (int k = 0; k < n3m; ++k) {
        int km = (k < h3p) ? k : n3m - k;
        dzk[k] = 2.0 * (1.0 - std::cos(2.0 * PI * static_cast<double>(km) * ddz / L3)) / (ddz * ddz);
        // printf("Rank %8d: km = %8d, dzk[%8d] = %8.15f\n", mpi_topology::myrank, km, k, dzk[k]);
        // printf("ddz / L3: %8.15f\n", ddz / L3);
    }
    for (int i = 0; i < h1pKsub; ++i) {
        int im = h1pKsub_ista + i - 1;
        dxk1[i] = 2.0 * (1.0 - std::cos(2.0 * PI * static_cast<double>(im) * ddx / L1)) / (ddx * ddx);
    }
    for (int i = 0; i < h1psub_Ksub; ++i) {
        int im = h1psub_Ksub_ista + i - 1;
        dxk2[i] = 2.0 * (1.0 - std::cos(2.0 * PI * static_cast<double>(im) * ddx / L1)) / (ddx * ddx);
        // printf("Rank %8d: h1psub_Ksub_ista: %8d, im: %8d, dxk2[%8d] = %8.15f\n", mpi_topology::myrank, h1psub_Ksub_ista, im, i, dxk2[i]);
        // printf("ddx / L1: %8.15f\n", ddx/ L1);
    }
}

// void mpi_poisson_RHS() {
//     const auto &x1 = x1_sub;
//     const auto &x2 = x2_sub;
//     const auto &x3 = x3_sub;

//     PRHS.assign(n1msub * n2msub * n3msub, 0.0);
//     for (int k = 0; k < n3msub; ++k) {
//         for (int j = 0; j < n2msub; ++j) {
//             for (int i = 0; i < n1msub; ++i) {
//                 double val = -std::cos(0.5 * (x1[i] + x1[i + 1]) * PI) *
//                              std::cos(0.5 * (x2[j] + x2[j + 1]) * PI) *
//                              std::cos(0.5 * (x3[k] + x3[k + 1]) * PI) *
//                              3.0 * PI * PI;
//                 PRHS[idx_rhs(i, j, k)] = val;
//             }
//         }
//     }
// }

void mpi_poisson_RHS() {
    const auto &x1 = x1_sub;
    const auto &x2 = x2_sub;
    const auto &x3 = x3_sub;

    PRHS.assign(n1msub * n2msub * n3msub, 0.0);
    for (int k = 1; k < n3msub + 1; ++k) {
        for (int j = 1; j < n2msub + 1; ++j) {
            for (int i = 1; i < n1msub + 1; ++i) {
                double x = 0.5 * (x1[i] + x1[i + 1]);
                double y = 0.5 * (x2[j] + x2[j + 1]);
                double z = 0.5 * (x3[k] + x3[k + 1]);
                double val = -9.0 * PI * PI *
                             std::sin(2.0 * PI * x / L1) *
                             std::sin(2.0 * PI * z / L3) *
                             std::sin(PI * y / L2);
                // printf("i: %8d, j: %8d, k: %8d, x: %8.15f, y: %8.15f, z: %8.15f, val: %8.15f\n", i, j, k, x, y, z, val);
                PRHS[idx_rhs(i - 1, j - 1, k - 1)] = val;
            }
        }
    }
}

void mpi_poisson_exact_sol() {
    const auto &x1 = x1_sub;
    const auto &x2 = x2_sub;
    const auto &x3 = x3_sub;

    for (int k = 1; k < n3msub + 1; ++k) {
        for (int j = 1; j < n2msub + 1; ++j) {
            for (int i = 1; i < n1msub + 1; ++i) {
                double x = 0.5 * (x1[i] + x1[i + 1]);
                double y = 0.5 * (x2[j] + x2[j + 1]);
                double z = 0.5 * (x3[k] + x3[k + 1]);
                exact_sol[idx_p(i - 1, j - 1, k - 1)] =
                    std::sin(2.0 * PI * x / L1) *
                    std::sin(2.0 * PI * z / L3) *
                    std::sin(PI * y / L2);
            }
        }
    }
}

// void mpi_poisson_exact_sol() {
//     const auto &x1 = x1_sub;
//     const auto &x2 = x2_sub;
//     const auto &x3 = x3_sub;

//     for (int k = 0; k < n3sub; ++k) {
//         for (int j = 0; j < n2sub; ++j) {
//             for (int i = 0; i < n1sub; ++i) {
//                 exact_sol[idx_p(i, j, k)] = std::cos(0.5 * (x1[i] + x1[i + 1]) * PI) *
//                                             std::cos(0.5 * (x2[j] + x2[j + 1]) * PI) *
//                                             std::cos(0.5 * (x3[k] + x3[k + 1]) * PI);
//             }
//         }
//     }
// }

//>
//> @brief       Main poisson solver with transpose scheme 1
//> @param       dx2         Grid spacing in y-direction for the example problem
//> @param       dmx2        Mesh length in y-direction for the example problem
//>
void mpi_Poisson_FFT1(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2) {
    using namespace mpi_topology;

    //> @{ Pointer of grid information
    double* dmx1 = dmx1_sub.data();
    double* dmx3 = dmx3_sub.data();
    //> @}

    int buffer_dp_size = std::max(n1m * n2mIsub * n3msub,
                                  n1msub * n2msub * n3msub);
    int buffer_cd_size = std::max(h1pKsub * n2mKsub * n3m,
                                  h1p * n2mIsub * n3msub);

    std::vector<double> buffer_dp1(buffer_dp_size);
    std::vector<double> buffer_dp2(buffer_dp_size);
    std::vector<std::complex<double>> buffer_cd1(buffer_cd_size);
    std::vector<std::complex<double>> buffer_cd2(buffer_cd_size);

    const int stamp_fft = 2;
    double* RHS_Iline = buffer_dp1.data();

    // Alltoall C to I
    timer::timer_stamp0(stamp_fft);
    MPI_Alltoallw(PRHS.data(), countsendI.data(), countdistI.data(),
                  ddtype_dble_C_in_C2I.data(),
                  RHS_Iline, countsendI.data(), countdistI.data(),
                  ddtype_dble_I_in_C2I.data(),
                  comm_1d_x1.mpi_comm);
    timer::timer_stamp(6, stamp_fft);
    std::vector<double>().swap(PRHS);
    
    // Forward FFT in x-direction
    std::complex<double>* RHSIhat_Iline = buffer_cd1.data();
    timer::timer_stamp(8, stamp_fft);
    int n[1] = {n1m};
    fftw_plan plan1 = fftw_plan_many_dft_r2c(1, n,
                       n2mIsub * n3msub,
                       RHS_Iline, n, 1, n1m,
                       reinterpret_cast<fftw_complex*>(buffer_cd1.data()),
                       n, 1, h1p,
                       FFTW_ESTIMATE);
    fftw_execute_dft_r2c(plan1, RHS_Iline,
                         reinterpret_cast<fftw_complex*>(buffer_cd1.data()));
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);
    
    // Alltoall I to K
    std::complex<double>* RHSIhat_Kline = buffer_cd2.data();
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(RHSIhat_Iline, countsendK.data(), countdistK.data(),
                  ddtype_cplx_I_in_I2K.data(),
                  RHSIhat_Kline, countsendK.data(), countdistK.data(),
                  ddtype_cplx_K_in_I2K.data(),
                  comm_1d_x3.mpi_comm);
    timer::timer_stamp(6, stamp_fft);
    
    // Forward FFT in z-direction
    std::vector<std::complex<double>> RHSIKhat_Kline(n3m * h1pKsub * n2mKsub);
    int nz[1] = {n3m};
    timer::timer_stamp(8, stamp_fft);
    plan1 = fftw_plan_many_dft(1, nz, h1pKsub * n2mKsub,
                               reinterpret_cast<fftw_complex*>(RHSIhat_Kline),
                               nz, h1pKsub * n2mKsub, 1,
                               reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()),
                               nz, 1, n3m,
                               FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute_dft(plan1,
                     reinterpret_cast<fftw_complex*>(RHSIhat_Kline),
                     reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()));
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);
    
    // Allocate arrays for matrix coefficients
    std::vector<double> Am_r(n3m * h1pKsub * n2mKsub),
        Ac_r(n3m * h1pKsub * n2mKsub),
        Ap_r(n3m * h1pKsub * n2mKsub),
        Be_r(n3m * h1pKsub * n2mKsub);
    std::vector<double> Am_c(n3m * h1pKsub * n2mKsub),
        Ac_c(n3m * h1pKsub * n2mKsub),
        Ap_c(n3m * h1pKsub * n2mKsub),
        Be_c(n3m * h1pKsub * n2mKsub);

    // Fortran wrong indexing
    // auto idxM = [=](int k, int i, int j) {
    //     return (j * h1pKsub + i) * n3m + k;
    // };

    auto idxM = [=](int k, int i, int j) {
        return ((i * n3m + k) * n2mKsub) + j;
    };

    // Build matrix coefficients for direct TDMA solver in y-direction
    // Calculate real and imaginary coefficients seperately.
    for (int j = 0; j < n2mKsub; ++j) {
        int jp = j + 1;
        double fft_amj = 1.0 / dx2[j + n2mKsub_jsta - 1] /
                          dmx2[j + n2mKsub_jsta - 1];
        double fft_apj = 1.0 / dx2[j + n2mKsub_jsta - 1] /
                          dmx2[jp + n2mKsub_jsta - 1];
        // Lower boundary in y-direction
        if (comm_1d_x1n2.myrank == 0 && j == 0)
            fft_amj = 0.0;
        // Upper boundary in y-direction
        if (comm_1d_x1n2.myrank == comm_1d_x1n2.nprocs - 1 && j == n2mKsub - 1)
            fft_apj = 0.0;

        double fft_acj = -fft_amj - fft_apj;

        for (int i = 0; i < h1pKsub; ++i) {
            for (int k = 0; k < n3m; ++k) {
                std::size_t id = idxM(k, i, j);
                auto val = RHSIKhat_Kline[id];

                // Define the RHS for both 'real' and 'complex'
                Be_r[id] = val.real();
                Be_c[id] = val.imag();

                Am_r[id] = fft_amj;
                Ac_r[id] = fft_acj - dxk1[i] - dzk[k];
                Ap_r[id] = fft_apj;

                Am_c[id] = fft_amj;
                Ac_c[id] = fft_acj - dxk1[i] - dzk[k];
                Ap_c[id] = fft_apj;
            }
        }
    }

    // Special treatment for global 1st element at (1,1,1)
    if (comm_1d_x1n2.myrank == 0 && h1pKsub_ista == 1 &&
        comm_1d_x3.myrank == 0) {
        std::size_t id = idxM(0, 0, 0);
        Am_r[id] = 0.0;
        Ac_r[id] = 1.0;
        Ap_r[id] = 0.0;
        Be_r[id] = 0.0;
    }

    timer::timer_stamp(9, stamp_fft);
    // Solve TDMA in y-direction : Obtain solutions for real and imaginary part separately.
    ptdma_plan_many plan_tdma;
    PaScaL_TDMA_plan_many_create(&plan_tdma, n3m * h1pKsub,
                                 comm_1d_x1n2.myrank,
                                 comm_1d_x1n2.nprocs,
                                 comm_1d_x1n2.mpi_comm);
    PaScaL_TDMA_many_solve(&plan_tdma, Am_r.data(), Ac_r.data(), Ap_r.data(),
                           Be_r.data(), n3m * h1pKsub, n2mKsub);
    PaScaL_TDMA_many_solve(&plan_tdma, Am_c.data(), Ac_c.data(), Ap_c.data(),
                           Be_c.data(), n3m * h1pKsub, n2mKsub);
    PaScaL_TDMA_plan_many_destroy(&plan_tdma, comm_1d_x1n2.nprocs);
    timer::timer_stamp(4, stamp_fft);
    
    // Build complex array from the TDM solution
    for (int j = 0; j < n2mKsub; ++j) {
        if (h1pKsub_ista == 1) {
            Be_r[idxM(0, 0, j)] = 0.0;
            Be_c[idxM(0, 0, j)] = 0.0;
            Be_c[idxM(h3p - 1, 0, j)] = 0.0;
        } else if (h1pKsub_iend == h1p) {
            Be_c[idxM(0, h1pKsub - 1, j)] = 0.0;
            Be_c[idxM(h3p - 1, h1pKsub - 1, j)] = 0.0;
        }
        for (int i = 0; i < h1pKsub; ++i) {
            for (int k = 0; k < n3m; ++k) {
                std::size_t id = idxM(k, i, j);
                RHSIKhat_Kline[id] = {Be_r[id], Be_c[id]};
            }
        }
    }

    // Backward FFT in z-direction
    std::vector<std::complex<double>> RHSIhat_Kline2(h1pKsub * n2mKsub * n3m);
    timer::timer_stamp(8, stamp_fft);
    plan1 = fftw_plan_many_dft(1, nz, h1pKsub * n2mKsub,
                               reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()),
                               nz, 1, n3m,
                               reinterpret_cast<fftw_complex*>(RHSIhat_Kline2.data()),
                               nz, h1pKsub * n2mKsub, 1,
                               FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute_dft(plan1,
                     reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()),
                     reinterpret_cast<fftw_complex*>(RHSIhat_Kline2.data()));
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);

    timer::timer_stamp(8, stamp_fft);

    // Alltoall K to I
    MPI_Alltoallw(RHSIhat_Kline2.data(), countsendK.data(), countdistK.data(),
                  ddtype_cplx_K_in_I2K.data(),
                  buffer_cd1.data(), countsendK.data(), countdistK.data(),
                  ddtype_cplx_I_in_I2K.data(),
                  comm_1d_x3.mpi_comm);
    timer::timer_stamp(6, stamp_fft);
    
    // Backward FFT in x-direction
    timer::timer_stamp(8, stamp_fft);
    plan1 = fftw_plan_many_dft_c2r(1, n,
                                   n2mIsub * n3msub,
                                   reinterpret_cast<fftw_complex*>(buffer_cd1.data()),
                                   n, 1, h1p,
                                   buffer_dp1.data(), n, 1, n1m,
                                   FFTW_ESTIMATE);
    fftw_execute_dft_c2r(plan1,
                         reinterpret_cast<fftw_complex*>(buffer_cd1.data()),
                         buffer_dp1.data());
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);

    // Alltoall I to C
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(buffer_dp1.data(), countsendI.data(), countdistI.data(),
                  ddtype_dble_I_in_C2I.data(),
                  buffer_dp2.data(), countsendI.data(), countdistI.data(),
                  ddtype_dble_C_in_C2I.data(),
                  comm_1d_x1.mpi_comm);
    timer::timer_stamp(6, stamp_fft);

    // Update the temporary solution
    timer::timer_stamp(8, stamp_fft);
    double factor = 1.0 / static_cast<double>(n1m) / static_cast<double>(n3m);
    double* tmp = buffer_dp2.data();
    for (int k = 0; k < n3msub; ++k)
        for (int j = 0; j < n2msub; ++j)
            for (int i = 0; i < n1msub; ++i)
                tmp[(k * n2msub + j) * n1msub + i] *= factor;

    // Calculate the average of obtained incremental pressure (dP)
    double AVERsub = 0.0;
    for (int k = 0; k < n3msub; ++k)
        for (int j = 0; j < n2msub; ++j)
            for (int i = 0; i < n1msub; ++i)
                AVERsub += tmp[(k * n2msub + j) * n1msub + i];

    double AVERmpi = 0.0;
    MPI_Allreduce(&AVERsub, &AVERmpi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    AVERmpi /= static_cast<double>(n1m) * static_cast<double>(n2m) *
               static_cast<double>(n3m);

    // Remove the average
    for (int k = 0; k < n3msub; ++k)
        for (int j = 0; j < n2msub; ++j)
            for (int i = 0; i < n1msub; ++i)
                P[idx_p(i, j, k)] =
                    tmp[(k * n2msub + j) * n1msub + i] - AVERmpi;

    // Boundary condition treatment in y-direction for the example problem
    // Lower wall
    if (comm_1d_x2.myrank == 0) {
        double Pbc_a = std::pow(dmx2[1] + dmx2[2], 2.0) /
                       (std::pow(dmx2[1] + dmx2[2], 2.0) -
                        std::pow(dmx2[1], 2.0));
        double Pbc_b = std::pow(dmx2[1], 2.0) /
                       (std::pow(dmx2[1] + dmx2[2], 2.0) -
                        std::pow(dmx2[1], 2.0));
        for (int k = 0; k < n3sub; ++k)
            for (int i = 0; i < n1msub; ++i)
                P[idx_p(i, 0, k)] = Pbc_a * P[idx_p(i, 1, k)] -
                                     Pbc_b * P[idx_p(i, 2, k)];
    }

    // Upper wall
    if (comm_1d_x2.myrank == comm_1d_x2.nprocs - 1) {
        double Pbc_a =
            std::pow(dmx2[n2sub] + dmx2[n2msub], 2.0) /
            (std::pow(dmx2[n2sub] + dmx2[n2msub], 2.0) -
             std::pow(dmx2[n2sub], 2.0));
        double Pbc_b = std::pow(dmx2[n2sub], 2.0) /
                       (std::pow(dmx2[n2sub] + dmx2[n2msub], 2.0) -
                        std::pow(dmx2[n2sub], 2.0));
        for (int k = 0; k < n3sub; ++k)
            for (int i = 0; i < n1msub; ++i)
                P[idx_p(i, n2sub, k)] =
                    Pbc_a * P[idx_p(i, n2msub, k)] -
                    Pbc_b * P[idx_p(i, n2sub - 2, k)];
    }
    timer::timer_stamp(7, stamp_fft);

    std::vector<double>().swap(buffer_dp1);
    std::vector<double>().swap(buffer_dp2);
    std::vector<std::complex<double>>().swap(buffer_cd1);
    std::vector<std::complex<double>>().swap(buffer_cd2);
    timer::timer_stamp(8, stamp_fft);
}

//>
//> @brief       Main poisson solver with transpose scheme 2
//> @param       dx2         Grid spacing in y-direction for the example problem
//> @param       dmx2        Mesh length in y-direction for the example problem
//>
void mpi_Poisson_FFT2(const std::vector<double>& dx2,
                      const std::vector<double>& dmx2) {
    using namespace mpi_topology;

    // Pointer of grid information
    double* dmx1 = dmx1_sub.data();
    double* dmx3 = dmx3_sub.data();
    (void)dmx1; (void)dmx3;

    int buffer_dp_size =
        std::max(n1m * n2msub * n3msub_Isub, n1msub * n2msub * n3msub);
    // h1p = (n1m / 2) + 1
    // h1psub = h1p / (processors along x)
    // h1psub_Ksub = h1psub / (processors along z)
    int buffer_cd_size = std::max({h1p * n2msub * n3msub_Isub,
                                   h1psub * n2msub * n3msub,
                                   h1psub_Ksub * n2msub * n3m});

    std::vector<double> buffer_dp1(buffer_dp_size);
    std::vector<double> buffer_dp2(buffer_dp_size);
    std::vector<std::complex<double>> buffer_cd1(buffer_cd_size);
    std::vector<std::complex<double>> buffer_cd2(buffer_cd_size);

    const int stamp_fft = 2;

    double* RHS_Iline = buffer_dp1.data();
    //== Alltoall C to I
    timer::timer_stamp0(stamp_fft);
    MPI_Alltoallw(PRHS.data(), countsendI.data(), countdistI.data(),
                  ddtype_dble_C_in_C2I.data(),
                  RHS_Iline, countsendI.data(), countdistI.data(),
                  ddtype_dble_I_in_C2I.data(),
                  comm_1d_x1.mpi_comm);
    timer::timer_stamp(6, stamp_fft);
    std::vector<double>().swap(PRHS);

    // Forward FFT in x-direction
    std::complex<double>* RHSIhat_Iline = buffer_cd1.data();
    timer::timer_stamp(8, stamp_fft);
    int n[1] = {n1m};
    fftw_plan plan1 = fftw_plan_many_dft_r2c(
        1, n, n2msub * n3msub_Isub, RHS_Iline, n, 1, n1m,
        reinterpret_cast<fftw_complex*>(RHSIhat_Iline), n, 1, h1p,
        FFTW_ESTIMATE);
    fftw_execute_dft_r2c(plan1, RHS_Iline,
                         reinterpret_cast<fftw_complex*>(RHSIhat_Iline));
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);

    // Alltoall I to C
    std::complex<double>* RHSIhat = buffer_cd2.data();
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(RHSIhat_Iline, countsendI.data(), countdistI.data(),
                  ddtype_cplx_I_in_C2I.data(),
                  RHSIhat, countsendI.data(), countdistI.data(),
                  ddtype_cplx_C_in_C2I.data(),
                  comm_1d_x1.mpi_comm);
    timer::timer_stamp(6, stamp_fft);

    // Alltoall C to K
    std::complex<double>* RHSIhat_Kline = buffer_cd1.data();
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(RHSIhat, countsendK.data(), countdistK.data(),
                  ddtype_cplx_C_in_C2K.data(),
                  RHSIhat_Kline, countsendK.data(), countdistK.data(),
                  ddtype_cplx_K_in_C2K.data(),
                  comm_1d_x3.mpi_comm);
    timer::timer_stamp(6, stamp_fft);

    // std::string fname_fft_real =
    //     "fft_real_rank" + std::to_string(mpi_topology::myrank) + ".txt";
    // std::string fname_fft_imag =
    //     "fft_imag_rank" + std::to_string(mpi_topology::myrank) + ".txt";

    // mpi_post::write_fft_result_components(RHSIhat_Kline,
    //                                       fname_fft_real, 
    //                                       fname_fft_imag, 
    //                                       h1psub_Ksub, 
    //                                       n2msub, 
    //                                       n3m, 
    //                                       0, 
    //                                       h1psub_Ksub - 1, 
    //                                       0, 
    //                                       n2msub - 1, 
    //                                       0, 
    //                                       n3m - 1);


    // Forward FFT in z-direction
    std::vector<std::complex<double>> RHSIKhat_Kline(
        n3m * h1psub_Ksub * n2msub);
    int nz[1] = {n3m};
    timer::timer_stamp(8, stamp_fft);
    plan1 = fftw_plan_many_dft(
        1, nz, h1psub_Ksub * n2msub,
        reinterpret_cast<fftw_complex*>(RHSIhat_Kline), nz,
        h1psub_Ksub * n2msub, 1,
        reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()), nz, 1, n3m,
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute_dft(plan1,
                     reinterpret_cast<fftw_complex*>(RHSIhat_Kline),
                     reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()));
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);
    
    // std::string fname_fft_real_z =
    //     "fftz_real_rank" + std::to_string(mpi_topology::myrank) + ".txt";
    // std::string fname_fft_imag_z =
    //     "fftz_imag_rank" + std::to_string(mpi_topology::myrank) + ".txt";

    // mpi_post::write_fftz_result_components(RHSIKhat_Kline.data(),
    //                                       fname_fft_real_z, 
    //                                       fname_fft_imag_z, 
    //                                       h1psub_Ksub, 
    //                                       n2msub, 
    //                                       n3m, 
    //                                       0, 
    //                                       h1psub_Ksub - 1, 
    //                                       0, 
    //                                       n2msub - 1, 
    //                                       0, 
    //                                       n3m - 1);

    // h1p = (n1m / 2) + 1
    // h1psub = h1p / (processors along x)
    // h1psub_Ksub = h1psub / (processors along z)
    // h1psub_Ksub * n2msub * n3m
    // int buffer_cd_size = std::max({h1p * n2msub * n3msub_Isub,
    //                                h1psub * n2msub * n3msub,
    //                                h1psub_Ksub * n2msub * n3m});
    // printf("Rank %8d:\n", mpi_topology::myrank);
    // printf("h1p: %8d, h1psub: %8d, h1psub_Ksub: %8d, n3msub_Isub: %8d, n3msub: %8d, n3m: %8d\n", h1p, h1psub, h1psub_Ksub, n3msub_Isub, n3msub, n3m);
   
    // Allocate arrays for matrix coefficients
    std::vector<double> Am_r(n3m * h1psub_Ksub * n2msub),
                        Ac_r(n3m * h1psub_Ksub * n2msub),
                        Ap_r(n3m * h1psub_Ksub * n2msub),
                        Be_r(n3m * h1psub_Ksub * n2msub);

    std::vector<double> Am_c(n3m * h1psub_Ksub * n2msub),
                        Ac_c(n3m * h1psub_Ksub * n2msub),
                        Ap_c(n3m * h1psub_Ksub * n2msub),
                        Be_c(n3m * h1psub_Ksub * n2msub);

    // auto idxM = [=](int k, int i, int j) {
    //     return (j * h1psub_Ksub + i) * n3m + k;
    // };

    auto idxM = [=](int k, int i, int j) {
        return ((i * n3m + k) * n2msub) + j;
    };

    // Build matrix coefficients for direct TDMA solver in y-direction
    // Calculate real and imaginary coefficients separately.
    for (int j = 0; j < n2msub; ++j) {
        int jp = j + 1;
        // dx2 = mpi_subdomain::dx2_sub
        // dx2_sub.assign(n2sub + 1, 0.0)
        // for (int j = 1; j <= n2msub; ++j) dx2_sub[j] = x2_sub[j + 1] - x2_sub[j];
        // dmx2 = mpi_subdomain::dmx2_sub
        // double fft_amj = 1.0 / dx2[j + 1] / dmx2[j + 1];
        // double fft_apj = 1.0 / dx2[j + 1] / dmx2[jp + 1];

        double fft_amj = 1.0 / dx2[j + 1] / dx2[j + 1];
        double fft_apj = 1.0 / dx2[j + 1] / dx2[j + 1];
        
        // if ((fft_amj != 64) || (fft_apj != 64)) {
        //     printf("WRONG CLACULATJE L DF SLKFJ FS \n");
        //     printf("fft_amj: %8.15f\n", fft_amj);
        //     printf("fft_apj: %8.15f\n", fft_apj);
        // }

        // // Lower boundary in y-direction
        // if (comm_1d_x2.myrank == 0 && j == 0)
        //     fft_amj = 0.0;
        // // Upper boundary in y-direction
        // if (comm_1d_x2.myrank == comm_1d_x2.nprocs - 1 && j == n2msub - 1)
        //     fft_apj = 0.0;

        

        // for (int j = 1; j <= n2msub; ++j) dx2_sub[j] = x2_sub[j + 1] - x2_sub[j];
        // // for (int j = 1; j <= n2msub; ++j) dmx2_sub[j] = 0.5 * (dx2_sub[j - 1] + dx2_sub[j]);
        // printf("Rank %8d:\n", mpi_topology::myrank);
        // printf("j: %8d\n", j);
        // printf("dx2[j+1]: %8.15f\n", dx2[j + 1]);
        // printf("dmx2[j+1]: %8.15f\n", dmx2[j + 1]);

        double fft_acj = -fft_amj - fft_apj;
        
        for (int i = 0; i < h1psub_Ksub; ++i) {
            for (int k = 0; k < n3m; ++k) {
                std::size_t id = idxM(k, i, j);
                // (j * h1psub_Ksub + i) * n3m + k
                auto val = RHSIKhat_Kline[(j * h1psub_Ksub + i) * n3m + k];

                // Define the RHS for both 'real' and 'complex'
                Be_r[id] = val.real();
                Be_c[id] = val.imag();

                Am_r[id] = fft_amj;                     // lower

                // center
                Ac_r[id] = fft_acj               // (-2 / dy**2)
                           - dxk2[i]             //
                           - dzk[k];  
                Ap_r[id] = fft_apj;                     // upper
                
                Am_c[id] = fft_amj;
                Ac_c[id] = fft_acj - dxk2[i] - dzk[k];
                Ap_c[id] = fft_apj;

                if (comm_1d_x2.myrank == 0 && j == 0) {
                    // a_0 = 0.0
                    Am_r[id] = 0.0;
                    Am_c[id] = 0.0;

                    Ac_r[id] = Ac_r[id] - 1.0 / dx2[j + 1] / dx2[j + 1];
                    Ac_c[id] = Ac_c[id] - 1.0 / dx2[j + 1] / dx2[j + 1];
                }
                if (comm_1d_x2.myrank == comm_1d_x2.nprocs - 1 && j == n2msub - 1) {
                    // c_N = 0.0
                    Ap_r[id] = 0.0;
                    Ap_c[id] = 0.0;

                    Ac_r[id] = Ac_r[id] - 1.0 / dx2[j + 1] / dx2[j + 1];
                    Ac_c[id] = Ac_c[id] - 1.0 / dx2[j + 1] / dx2[j + 1];
                }
            }
        }
    }

    // Special treatment for global 1st element at (1,1,1)
    // if (comm_1d_x1.myrank == 0 && comm_1d_x2.myrank == 0 &&
    //     comm_1d_x3.myrank == 0) {
    //     std::size_t id = idxM(0, 0, 0);
    //     Am_r[id] = 0.0;
    //     Ac_r[id] = 1.0;
    //     Ap_r[id] = 0.0;
    //     Be_r[id] = 0.0;
    // }

    // Print coefficients
    // if (mpi_topology::myrank == 0 || mpi_topology::myrank == 2){
    //     printf("Rank %8d: \n", mpi_topology::myrank);
    //     for (int j = 0; j < n2msub; j++) {
    //         // auto idxM = [=](int k, int i, int j) {
    //         //     return ((i * n3m + k) * n2mKsub) + j;
    //         // };
    //         int index_for_printing = idxM(1, 1, j);
    //         printf("n3m: %8d\n", n3m);
    //         printf("n2msub: %8d\n", n2msub);
    //         printf("Index for printing: %8d\n", index_for_printing);

            
    //         // print d_star
    //         // printf("d_star[%8d].real: %8.15f\n",j, Be_r[index_for_printing]);
    //         printf("a_star[%8d]: %8.15f\n",index_for_printing, Am_r[index_for_printing]);
    //         printf("c_star[%8d]: %8.15f\n",index_for_printing, Ap_r[index_for_printing]);
    //         printf("b_star[%8d]: %8.15f\n",index_for_printing, Ac_r[index_for_printing]);
    //     }
    // }

    timer::timer_stamp(9, stamp_fft);
    ptdma_plan_many plan_tdma;
    // Solve TDMA in y-direction : Obtain solutions for real and imaginary part separately.
    PaScaL_TDMA_plan_many_create(&plan_tdma,
                                 n3m * h1psub_Ksub,          // number of systems
                                 comm_1d_x2.myrank, 
                                 comm_1d_x2.nprocs,
                                 comm_1d_x2.mpi_comm);

    PaScaL_TDMA_many_solve(&plan_tdma, Am_r.data(), Ac_r.data(), Ap_r.data(),
                           Be_r.data(), n3m * h1psub_Ksub, n2msub);
    PaScaL_TDMA_many_solve(&plan_tdma, Am_c.data(), Ac_c.data(), Ap_c.data(),
                           Be_c.data(), n3m * h1psub_Ksub, n2msub);

    
    
    PaScaL_TDMA_plan_many_destroy(&plan_tdma, comm_1d_x2.nprocs);
    timer::timer_stamp(4, stamp_fft);

    // if ((mpi_topology::myrank == 0 || mpi_topology::myrank == 2)) {
    //     printf("Rank %8d: \n", mpi_topology::myrank);
    //     // PaScaL_TDMA_many_solve(&plan_tdma, Am_r.data(), Ac_r.data(), Ap_r.data(),
    //     //                    Be_r.data(), n3m * h1psub_Ksub, n2msub);
    //     for (int j = 0; j < n2msub; j++) {
    //         // ((i * n3m + k) * n2mKsub) + j       (k,i,j)
    //         int index_for_printing = idxM(1, 1, j);

    //         // print d_star
    //         // printf("d_star[%8d].real: %8.15f\n",j, Be_r[index_for_printing]);
    //         printf("D[%8d]: %8.15f\n",index_for_printing, Be_c[index_for_printing]);
    //         // printf("b_star[%8d]: %8.15f\n",j, Ac_r[index_for_printing]);
    //     }
    // }

    // Build complex array from the TDM solution
    
    for (int j = 0; j < n2msub; ++j) {
        // if (h1psub_Ksub_ista == 1) {
        //     // zero the real component also
        //     Be_r[idxM(0, 0, j)] = 0.0;
        //     ///////////////////////////////
        //     Be_c[idxM(0, 0, j)] = 0.0;
        //     Be_c[idxM(h3p - 1, 0, j)] = 0.0;
        // } else if (h1psub_Ksub_iend == h1p) {
        //     Be_c[idxM(0, h1psub_Ksub - 1, j)] = 0.0;
        //     Be_c[idxM(h3p - 1, h1psub_Ksub - 1, j)] = 0.0;
        // }
        for (int i = 0; i < h1psub_Ksub; ++i)
            for (int k = 0; k < n3m; ++k) {
                std::size_t id = idxM(k, i, j);
                RHSIKhat_Kline[(j * h1psub_Ksub + i) * n3m + k] = {Be_r[id], Be_c[id]};
            }
    }

    // Deallocate A-matrix
    std::vector<double>().swap(Am_r);
    std::vector<double>().swap(Ac_r);
    std::vector<double>().swap(Ap_r);
    std::vector<double>().swap(Be_r);
    std::vector<double>().swap(Am_c);
    std::vector<double>().swap(Ac_c);
    std::vector<double>().swap(Ap_c);
    std::vector<double>().swap(Be_c);

    // std::string fname_tdma_real =
    //     "TDMA_real_rank" + std::to_string(mpi_topology::myrank) + ".txt";
    // std::string fname_tdma_imag =
    //     "TDMA_imag_rank" + std::to_string(mpi_topology::myrank) + ".txt";

    // mpi_post::write_fftz_result_components(RHSIKhat_Kline.data(),
    //                                       fname_tdma_real, 
    //                                       fname_tdma_imag, 
    //                                       h1psub_Ksub, 
    //                                       n2msub, 
    //                                       n3m, 
    //                                       0, 
    //                                       h1psub_Ksub - 1, 
    //                                       0, 
    //                                       n2msub - 1, 
    //                                       0, 
    //                                       n3m - 1);
    // 여기까지 validation 완료

    // Backward FFT in z-direction
    std::vector<std::complex<double>> RHSIhat_Kline2(
        h1psub_Ksub * n2msub * n3m);
    timer::timer_stamp(8, stamp_fft);
    plan1 = fftw_plan_many_dft(
        1, nz, h1psub_Ksub * n2msub,
        reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()), nz, 1, n3m,
        reinterpret_cast<fftw_complex*>(RHSIhat_Kline2.data()), nz,
        h1psub_Ksub * n2msub, 1, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute_dft(plan1,
                     reinterpret_cast<fftw_complex*>(RHSIKhat_Kline.data()),
                     reinterpret_cast<fftw_complex*>(RHSIhat_Kline2.data()));
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);

    // Alltoall K to C
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(RHSIhat_Kline2.data(), countsendK.data(), countdistK.data(),
                  ddtype_cplx_K_in_C2K.data(),
                  buffer_cd1.data(), countsendK.data(), countdistK.data(),
                  ddtype_cplx_C_in_C2K.data(),
                  comm_1d_x3.mpi_comm);
    timer::timer_stamp(6, stamp_fft);

    // Alltoall C to I
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(buffer_cd1.data(), countsendI.data(), countdistI.data(),
                  ddtype_cplx_C_in_C2I.data(),
                  buffer_cd2.data(), countsendI.data(), countdistI.data(),
                  ddtype_cplx_I_in_C2I.data(),
                  comm_1d_x1.mpi_comm);
    timer::timer_stamp(6, stamp_fft);

    // Backward FFT in x-direction
    timer::timer_stamp(8, stamp_fft);
    plan1 = fftw_plan_many_dft_c2r(
        1, n, n2msub * n3msub_Isub,
        reinterpret_cast<fftw_complex*>(buffer_cd2.data()), n, 1, h1p,
        buffer_dp1.data(), n, 1, n1m, FFTW_ESTIMATE);
    fftw_execute_dft_c2r(plan1,
                         reinterpret_cast<fftw_complex*>(buffer_cd2.data()),
                         buffer_dp1.data());
    fftw_destroy_plan(plan1);
    timer::timer_stamp(5, stamp_fft);

    // std::string fname = "solution_rank" + std::to_string(mpi_topology::myrank) + ".txt";

    // // buffer_dp_size = max(n1m * n2mIsub * n3msub, n1msub * n2msub * n3msub);
    // mpi_post::write_scalar_components(buffer_dp1, 
    //                                   fname, 
    //                                   n1m, 
    //                                   n2msub, 
    //                                   n3msub, 
    //                                   0, 
    //                                   n1m, 
    //                                   0, 
    //                                   n2msub, 
    //                                   0, 
    //                                   n3msub);

    // Alltoall I to C
    timer::timer_stamp(8, stamp_fft);
    MPI_Alltoallw(buffer_dp1.data(), countsendI.data(), countdistI.data(),
                  ddtype_dble_I_in_C2I.data(),
                  buffer_dp2.data(), countsendI.data(), countdistI.data(),
                  ddtype_dble_C_in_C2I.data(),
                  comm_1d_x1.mpi_comm);
    timer::timer_stamp(6, stamp_fft);
    


    // 여기서부터 이거 왜하는지 모르겠음 ㅇㅇ
    // Update the temporary solution
    // timer::timer_stamp(8, stamp_fft);
    double factor = 1.0 / static_cast<double>(n1m) /
                    static_cast<double>(n3m);
    double* tmp = buffer_dp2.data();
    for (int k = 0; k < n3msub; ++k)
        for (int j = 0; j < n2msub; ++j)
            for (int i = 0; i < n1msub; ++i)
                tmp[(k * n2msub + j) * n1msub + i] *= factor;

    // Calculate thee average of obtained incremental (P)
    // double AVERsub = 0.0;
    // for (int k = 0; k < n3msub; ++k)
    //     for (int j = 0; j < n2msub; ++j)
    //         for (int i = 0; i < n1msub; ++i)
    //             AVERsub += tmp[(k * n2msub + j) * n1msub + i];

    // double AVERmpi = 0.0;
    // MPI_Allreduce(&AVERsub, &AVERmpi, 1, MPI_DOUBLE, MPI_SUM,
    //               MPI_COMM_WORLD);
    // AVERmpi /= static_cast<double>(n1m) * static_cast<double>(n2m) *
    //            static_cast<double>(n3m);

    // Remove the average
    for (int k = 1; k < n3msub + 1; ++k)
        for (int j = 1; j < n2msub + 1; ++j)
            for (int i = 1; i < n1msub + 1; ++i)
                P[(k * (n2msub + 2) + j) * (n1msub + 2) + i] =
                tmp[((k-1) * n2msub + (j-1)) * n1msub + i-1];
                // tmp[((k-1) * n2msub + (j-1)) * n1msub + i-1] - AVERmpi;
    std::string fname = "solution_rank" + std::to_string(mpi_topology::myrank) + ".txt";
    mpi_post::write_scalar_components(P, 
                                      fname, 
                                      n1msub + 2, 
                                      n2msub + 2, 
                                      n3msub + 2, 
                                      1, 
                                      n1msub, 
                                      1, 
                                      n2msub, 
                                      1, 
                                      n3msub);

    // std::string fname = "solution_rank" + std::to_string(mpi_topology::myrank) + ".txt";
    // printf("Rank %8d: n1msub = %8d, n2msub = %8d, n3msub = %8d\n", mpi_topology::myrank, n1msub, n2msub, n3msub);
    // mpi_post::write_scalar_components(P, 
    //                                   fname, 
    //                                   n1msub + 2, 
    //                                   n2msub + 2, 
    //                                   n3msub + 2, 
    //                                   1, 
    //                                   n1msub, 
    //                                   1, 
    //                                   n2msub, 
    //                                   1, 
    //                                   n3msub);
    // buffer_dp_size = max(n1m * n2mIsub * n3msub, n1msub * n2msub * n3msub);
    
    // mpi_post::write_scalar_components(P, 
    //                                   fname, 
    //                                   n1msub, 
    //                                   n2msub, 
    //                                   n3msub, 
    //                                   0, 
    //                                   n1msub, 
    //                                   0, 
    //                                   n2msub, 
    //                                   0, 
    //                                   n3msub);

    

//  inline std::size_t idx_p(int i, int j, int k) {
//     return (static_cast<std::size_t>(k) * (n2sub + 1) + j) * (n1sub + 1) + i;
// }

    // Boundary condition treatment in y-direction for the example problem
    // Lower wall
    // if (comm_1d_x2.myrank == 0) {
    //     double Pbc_a = std::pow(dmx2[1] + dmx2[2], 2.0) /
    //                    (std::pow(dmx2[1] + dmx2[2], 2.0) -
    //                     std::pow(dmx2[1], 2.0));
    //     double Pbc_b = std::pow(dmx2[1], 2.0) /
    //                    (std::pow(dmx2[1] + dmx2[2], 2.0) -
    //                     std::pow(dmx2[1], 2.0));
    //     for (int k = 0; k < n3sub; ++k)
    //         for (int i = 0; i < n1msub; ++i)
    //             P[idx_p(i, 0, k)] = Pbc_a * P[idx_p(i, 1, k)] -
    //                                  Pbc_b * P[idx_p(i, 2, k)];
    // }
    // // Upper wall
    // if (comm_1d_x2.myrank == comm_1d_x2.nprocs - 1) {
    //     double Pbc_a =
    //         std::pow(dmx2[n2sub] + dmx2[n2msub], 2.0) /
    //         (std::pow(dmx2[n2sub] + dmx2[n2msub], 2.0) -
    //          std::pow(dmx2[n2sub], 2.0));
    //     double Pbc_b = std::pow(dmx2[n2sub], 2.0) /
    //                    (std::pow(dmx2[n2sub] + dmx2[n2msub], 2.0) -
    //                     std::pow(dmx2[n2sub], 2.0));
    //     for (int k = 0; k < n3sub; ++k)
    //         for (int i = 0; i < n1msub; ++i)
    //             P[idx_p(i, n2sub, k)] =
    //                 Pbc_a * P[idx_p(i, n2msub, k)] -
    //                 Pbc_b * P[idx_p(i, n2sub - 2, k)];
    // }
    // 그냥 다 삭제요 ㅃㅃ


    timer::timer_stamp(7, stamp_fft);

    std::vector<double>().swap(buffer_dp1);
    std::vector<double>().swap(buffer_dp2);
    std::vector<std::complex<double>>().swap(buffer_cd1);
    std::vector<std::complex<double>>().swap(buffer_cd2);
    timer::timer_stamp(8, stamp_fft);
}

} // namespace mpi_poisson
