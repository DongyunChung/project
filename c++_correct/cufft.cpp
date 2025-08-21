#include <iostream>
#include <vector>
#include <cmath>
#include <vector>
#include <fftw3.h>
#include <complex>
#include <mpi.h>
#include <inttypes.h>
#include "PaScaL_POISSON_FFT.hpp"

void fft_x_direction(
    const double* input_data, // size: N_z_mpi * N_y_mpi * N_x
    fftw_complex* output_data // size: N_z_mpi * N_y_mpi * (N_x / 2 + 1)
) {
    double* temp_in = fftw_alloc_real(N_x);
    fftw_complex* temp_out = fftw_alloc_complex(N_x / 2 + 1);
    
    fftw_plan plan = fftw_plan_dft_r2c_1d(N_x, temp_in, temp_out, FFTW_MEASURE);
    
    for (int k = 0; k < N_z_mpi; ++k) {
        for (int j = 0; j < N_y_mpi; ++j) {
            // Fill temp_in with x-line at (j,k)
            for (int i = 0; i < N_x; ++i) {
                temp_in[i] = input_data[k * N_y_mpi * N_x + j * N_x + i];
            }

            // FFT along x
            fftw_execute(plan);

            // Store output
            for (int i = 0; i < (N_x / 2 + 1); ++i) {
                int idx_out = k * N_y_mpi * (N_x / 2 + 1) + j * (N_x / 2 + 1) + i;
                output_data[idx_out][0] = temp_out[i][0] / sqrt(N_x);  // real
                output_data[idx_out][1] = temp_out[i][1] / sqrt(N_x);  // imag
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(temp_in);
    fftw_free(temp_out);
}

void ifft_x_direction(
    const fftw_complex* input_data, // size: N_z_mpi * N_y_mpi * N_x
    double* output_data // size: N_z_mpi * N_y_mpi * (N_x / 2 + 1)
) {
    fftw_complex* temp_in = fftw_alloc_complex(N_kx);
    double* temp_out = fftw_alloc_real(N_x);
    
    fftw_plan plan = fftw_plan_dft_c2r_1d(N_x, temp_in, temp_out, FFTW_MEASURE);
    
    for (int k = 0; k < N_z_mpi; ++k) {
        for (int j = 0; j < N_y_mpi; ++j) {
            // Fill temp_in with x-line at (j,k)
            for (int i = 0; i < (N_x / 2 + 1); ++i) {
                int idx_in = k * N_y_mpi * (N_kx) + j * (N_kx) + i;
                temp_in[i][0] = input_data[idx_in][0]; // real
                temp_in[i][1] = input_data[idx_in][1]; // imag
            }

            // FFT along x
            fftw_execute(plan);

            // Store output
            for (int i = 0; i < N_x; ++i) {
                int idx_out = k * N_y_mpi * N_x + j * N_x + i;
                output_data[idx_out] = temp_out[i] / sqrt(N_x);
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(temp_in);
    fftw_free(temp_out);
}

void transpose_xy_pencil_corrected(
    fftw_complex* send_buf,
    fftw_complex* recv_buf,
    int N_x_mpi,
    int my_col,
    MPI_Comm comm,
    MPI_Request* request
) {
    int N_y_mpi_new = N_y_mpi / p_y;

    std::vector<MPI_Datatype> send_types(size, MPI_DATATYPE_NULL);
    std::vector<MPI_Datatype> recv_types(size, MPI_DATATYPE_NULL);

    std::vector<MPI_Count> send_counts(size, 0);
    std::vector<MPI_Count> recv_counts(size, 0);

    // MPI_Aint base_send,temp_send, base_recv,temp_recv, temp_addr;
    // MPI_Get_address(&send_buf[0], &base_send);
    
    // MPI_Get_address(&recv_buf[0], &base_recv);
    std::vector<MPI_Aint> send_displs_aint(size, 0), recv_displs_aint(size, 0);

    // total layout
    int full_sizes[3] = {N_z_mpi, N_y_mpi, N_kx};         // z, y, kx
    int transposed_sizes[3] = {N_z_mpi, N_y, N_x_mpi};    // z, kx, y
    
    for (int their_col = 0; their_col < p_y; ++their_col) {
        int p = my_floor * p_y + their_col;  // 같은 z층의 rank만 대상

        int their_N_x_mpi = (N_x / (2*p_y)) + (their_col / (p_y - 1));

        // Send subarray: my local (z, y, kx)
        int send_start[3] = {0, 0, their_col * original_N_x_mpi}; // 원래 배열 기준 어디서부터 보낼건지
        int send_subsizes[3] = {N_z_mpi, N_y_mpi, their_N_x_mpi}; // 원래 배열 상에서 z, y, x 방향으로 얼마만큼 잘라서 줄건지
        
        MPI_Type_create_subarray(
            3,                            // 차원 수
            full_sizes,                   // 전체 배열 크기 (전역 배열)
            send_subsizes,                // 내가 보내거나 받을 블럭의 크기
            send_start,                   // 내가 보내거나 받을 블럭의 시작 위치 (원래 배열 기준)
            MPI_ORDER_C,                  // 메모리 순서 (MPI_ORDER_C = C 배열)
            MPI_C_DOUBLE_COMPLEX,         // 원소 타입 
            &send_types[p]);              // 결과 타입 (이 안에 서브배열 구조가 저장됨)

        MPI_Type_commit(&send_types[p]);

        send_counts[p] = 1;

        // Receive subarray: new (z, kx, y)
        int recv_start[3] = {0, 0, their_col * N_y_mpi * N_x_mpi};      // 바뀐 배열 기준 어디서부터 받을건지
        int recv_subsizes[3] = {N_z_mpi, N_y_mpi, N_x_mpi};   // 바뀐 배열 상에서 z, x, y방향으로 얼마만큼 잘라서 받을건지

        MPI_Type_create_subarray(
            3,                            // 차원 수
            transposed_sizes,             // 전체 배열 크기 (전역 배열)
            recv_subsizes,                // 내가 보내거나 받을 블럭의 크기
            recv_start,                   // 내가 보내거나 받을 블럭의 시작 위치 (원래 배열 기준)
            MPI_ORDER_C,                  // 메모리 순서 (MPI_ORDER_C = C 배열)
            MPI_C_DOUBLE_COMPLEX,         // 원소 타입 
            &recv_types[p]);              // 결과 타입 (이 안에 서브배열 구조가 저장됨)
        MPI_Type_commit(&recv_types[p]);

        // MPI_Barrier(comm);
        // printf(" Rank %8d sent data to Rank %8d: \nsended: %8d, received: %8d\n", rank,
        //         p, N_z_mpi* N_y_mpi*their_N_x_mpi, N_z_mpi * N_x_mpi * N_y_mpi);
        // MPI_Barrier(comm);
        recv_counts[p] = 1;

        // MPI_Get_address(&send_buf[their_col * original_N_x_mpi], &temp_addr);
        // MPI_Get_address(reinterpret_cast<void*>(&send_buf[0]), &temp_addr);
        // printf("Rank %8d to Rank %8d: Actual displacement (send): %8d\n", rank, p,their_col * original_N_x_mpi);
        // send_displs_aint[p] = temp_addr - base_send;

        // MPI_Get_address(&recv_buf[their_col * N_y_mpi], &temp_addr);
        // MPI_Get_address(reinterpret_cast<void*>(&recv_buf[0]), &temp_addr);
        // printf("Rank %8d to Rank %8d: Actual displacement (recv): %8d\n",rank, p, their_col * N_y_mpi);
        // recv_displs_aint[p] = temp_addr - base_recv;
    }
    
    // for (int i = 0; i < size; ++i) {
    //     if (i == rank) {
    //         printf("Rank: %8d\n", rank);
    //         // printf("Send buff size: %8d\n", N_y_mpi * N_z_mpi * (N_x / 2 + 1));
    //         // printf("Recv buff size: %8d\n", N_z_mpi * N_x_mpi * N_y);
    //         for (int j = 0; j < size; ++j) {
    //             // printf("Sendcounts[%8d]: %8d\n", j, send_counts[j]);
    //             // printf("Recvcounts[%8d]: %8d\n", j, recv_counts[j]);
                
    //             printf("Rank %8d sent to Rank %8d from address %lld\n", rank, j, (long long)send_displs_aint[j]);
                
    //             printf("Rank %8d received from Rank %8d on address %lld\n", rank, j, (long long)recv_displs_aint[j]);
                
    //             fflush(stdout);
    //         }
    //     }
    //     MPI_Barrier(comm);
    // }
    // for (int i = 0; i < size; ++i) {
    //     if (i == rank) {
    //         printf("Rank: %8d\n", rank);
    //         for (int j = 0; j < size; ++j) {
    //             printf("Sendcounts[%8d]: %8d\n", j, send_counts[j]);
    //             printf("Recvcounts[%8d]: %8d\n", j, recv_counts[j]);

    //             fflush(stdout);
    //         }
    //     }
    //     MPI_Barrier(comm);
    // }

    // for (int p = 0; p < size; ++p) {
    //     if (send_counts[p] == 0) {
    //         send_displs_aint[p] = 0;
    //         send_types[p] = MPI_DATATYPE_NULL;
    //     }
    //     if (recv_counts[p] == 0) {
    //         recv_displs_aint[p] = 0;
    //         recv_types[p] = MPI_DATATYPE_NULL;
    //     }
    // }

    // MPI_Barrier(comm);
    
    // MPI_Alltoallw_c(
    //     send_buf, send_counts.data(), send_displs_aint.data(), send_types.data(),
    //     recv_buf, recv_counts.data(), recv_displs_aint.data(), recv_types.data(),
    //     comm
    // );
    MPI_Ialltoallw_c(
    send_buf, send_counts.data(), send_displs_aint.data(), send_types.data(),
    recv_buf, recv_counts.data(), recv_displs_aint.data(), recv_types.data(),
    comm, request
    );

    MPI_Wait(request, MPI_STATUS_IGNORE);

    // MPI_Barrier(comm);
    for (int p = 0; p < size; ++p) {
        if (send_types[p] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&send_types[p]);
        }
        if (recv_types[p] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&recv_types[p]);
        }
    }
}

void transpose_yx_pencil_corrected(
    fftw_complex* send_buf,
    fftw_complex* recv_buf,
    int N_x_mpi,
    int my_col,
    MPI_Comm comm,
    MPI_Request* request
) {

    std::vector<MPI_Datatype> send_types(size, MPI_DATATYPE_NULL);
    std::vector<MPI_Datatype> recv_types(size, MPI_DATATYPE_NULL);

    std::vector<MPI_Count> send_counts(size, 0);
    std::vector<MPI_Count> recv_counts(size, 0);

    // MPI_Aint base_send,temp_send, base_recv,temp_recv, temp_addr;
    // MPI_Get_address(&send_buf[0], &base_send);
    
    // MPI_Get_address(&recv_buf[0], &base_recv);
    std::vector<MPI_Aint> send_displs_aint(size, 0), recv_displs_aint(size, 0);

    // total layout
    int full_sizes[3] = {N_z_mpi, N_y, N_x_mpi};
    int transposed_sizes[3] = {N_z_mpi, N_y_mpi, N_kx};

    for (int their_col = 0; their_col < p_y; ++their_col) {
        int p = my_floor * p_y + their_col;  // 같은 z층의 rank만 대상
        int their_N_x_mpi = (N_x / (2*p_y)) + (their_col / (p_y - 1));

        // Send subarray: my local (z, y, kx)
        int send_start[3] = {0, their_col * N_y_mpi, 0}; // 원래 배열 기준 어디서부터 보낼건지
        int send_subsizes[3] = {N_z_mpi, N_y_mpi, N_x_mpi}; // 원래 배열 상에서 z, y, x 방향으로 얼마만큼 잘라서 줄건지
        
        MPI_Type_create_subarray(
            3,                            // 차원 수
            full_sizes,                   // 전체 배열 크기 (전역 배열)
            send_subsizes,                // 내가 보내거나 받을 블럭의 크기
            send_start,                   // 내가 보내거나 받을 블럭의 시작 위치 (원래 배열 기준)
            MPI_ORDER_C,                  // 메모리 순서 (MPI_ORDER_C = C 배열)
            MPI_C_DOUBLE_COMPLEX,         // 원소 타입 
            &send_types[p]);              // 결과 타입 (이 안에 서브배열 구조가 저장됨)

        MPI_Type_commit(&send_types[p]);

        send_counts[p] = 1;

        // Receive subarray: new (z, kx, y)
        int recv_start[3] = {0, 0, their_col * original_N_x_mpi};      // 바뀐 배열 기준 어디서부터 받을건지
        int recv_subsizes[3] = {N_z_mpi, N_y_mpi, their_N_x_mpi};   // 바뀐 배열 상에서 z, x, y방향으로 얼마만큼 잘라서 받을건지

        MPI_Type_create_subarray(
            3,                            // 차원 수
            transposed_sizes,             // 전체 배열 크기 (전역 배열)
            recv_subsizes,                // 내가 보내거나 받을 블럭의 크기
            recv_start,                   // 내가 보내거나 받을 블럭의 시작 위치 (원래 배열 기준)
            MPI_ORDER_C,                  // 메모리 순서 (MPI_ORDER_C = C 배열)
            MPI_C_DOUBLE_COMPLEX,         // 원소 타입 
            &recv_types[p]);              // 결과 타입 (이 안에 서브배열 구조가 저장됨)
        MPI_Type_commit(&recv_types[p]);

        // MPI_Barrier(comm);
        // printf(" Rank %8d sent data to Rank %8d: \nsended: %8d, received: %8d\n", rank,
        //         p, N_z_mpi* N_y_mpi*their_N_x_mpi, N_z_mpi * N_x_mpi * N_y_mpi);

        recv_counts[p] = 1;

        // MPI_Get_address(&send_buf[their_col * original_N_x_mpi], &temp_addr);
        // MPI_Get_address(reinterpret_cast<void*>(&send_buf[0]), &temp_addr);
        // printf("Rank %8d to Rank %8d: Actual displacement (send): %8d\n", rank, p,their_col * original_N_x_mpi);
        // send_displs_aint[p] = temp_addr - base_send;

        // MPI_Get_address(&recv_buf[their_col * N_y_mpi], &temp_addr);
        // MPI_Get_address(reinterpret_cast<void*>(&recv_buf[0]), &temp_addr);
        // printf("Rank %8d to Rank %8d: Actual displacement (recv): %8d\n",rank, p, their_col * N_y_mpi);
        // recv_displs_aint[p] = temp_addr - base_recv;
    }
    
    // for (int i = 0; i < size; ++i) {
    //     if (i == rank) {
    //         printf("Rank: %8d\n", rank);
    //         // printf("Send buff size: %8d\n", N_y_mpi * N_z_mpi * (N_x / 2 + 1));
    //         // printf("Recv buff size: %8d\n", N_z_mpi * N_x_mpi * N_y);
    //         for (int j = 0; j < size; ++j) {
    //             // printf("Sendcounts[%8d]: %8d\n", j, send_counts[j]);
    //             // printf("Recvcounts[%8d]: %8d\n", j, recv_counts[j]);
                
    //             printf("Rank %8d sent to Rank %8d from address %lld\n", rank, j, (long long)send_displs_aint[j]);
                
    //             printf("Rank %8d received from Rank %8d on address %lld\n", rank, j, (long long)recv_displs_aint[j]);
                
    //             fflush(stdout);
    //         }
    //     }
    //     MPI_Barrier(comm);
    // }
    // for (int i = 0; i < size; ++i) {
    //     if (i == rank) {
    //         printf("Rank: %8d\n", rank);
    //         for (int j = 0; j < size; ++j) {
    //             printf("Sendcounts[%8d]: %8d\n", j, send_counts[j]);
    //             printf("Recvcounts[%8d]: %8d\n", j, recv_counts[j]);

    //             fflush(stdout);
    //         }
    //     }
    //     MPI_Barrier(comm);
    // }

    // for (int p = 0; p < size; ++p) {
    //     if (send_counts[p] == 0) {
    //         send_displs_aint[p] = 0;
    //         send_types[p] = MPI_DATATYPE_NULL;
    //     }
    //     if (recv_counts[p] == 0) {
    //         recv_displs_aint[p] = 0;
    //         recv_types[p] = MPI_DATATYPE_NULL;
    //     }
    // }

    // MPI_Barrier(comm);
    
    // MPI_Alltoallw_c(
    //     send_buf, send_counts.data(), send_displs_aint.data(), send_types.data(),
    //     recv_buf, recv_counts.data(), recv_displs_aint.data(), recv_types.data(),
    //     comm
    // );
    MPI_Ialltoallw_c(
    send_buf, send_counts.data(), send_displs_aint.data(), send_types.data(),
    recv_buf, recv_counts.data(), recv_displs_aint.data(), recv_types.data(),
    comm, request
    );

    MPI_Wait(request, MPI_STATUS_IGNORE);
    
    for (int p = 0; p < size; ++p) {
        if (send_types[p] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&send_types[p]);
        }
        // MPI_Type_free(&send_types[p]);
        // MPI_Type_free(&recv_types[p]);
        if (recv_types[p] != MPI_DATATYPE_NULL) {
            MPI_Type_free(&recv_types[p]);
        }
    }
    
}

void fft_y_direction(
    std::vector<fftw_complex>& input,  // input: z-x-y (전치 후 결과)
    std::vector<fftw_complex>& output,  // output: z-x-ky (FFT된 결과)
    int N_x_mpi
) {

    fftw_complex* in = fftw_alloc_complex(N_y);
    fftw_complex* out = fftw_alloc_complex(N_y);

    fftw_plan plan = fftw_plan_dft_1d(N_y, in, out, FFTW_FORWARD, FFTW_MEASURE);

    for (int k = 0; k < N_z_mpi; ++k) {
        for (int i = 0; i < N_x_mpi; ++i) {
            for (int j = 0; j < N_y; ++j) {
                in[j][0] = input[IDX3D_yfft(i, j, k)][0];  // real
                in[j][1] = input[IDX3D_yfft(i, j, k)][1];  // imag
            }

            fftw_execute(plan);

            for (int j = 0; j < N_y; ++j) {
                output[IDX3D_yfft(i, j, k)][0] = out[j][0] / sqrt(N_y);
                output[IDX3D_yfft(i, j, k)][1] = out[j][1] / sqrt(N_y);
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}


void ifft_y_direction(
    std::vector<fftw_complex>& input,   // input: z-x-ky (FFT된 입력)
    std::vector<fftw_complex>& output,  // output: z-x-y (전치 후 실공간)
    int N_x_mpi
) {
    fftw_complex* in = fftw_alloc_complex(N_y);
    fftw_complex* out = fftw_alloc_complex(N_y);

    fftw_plan plan = fftw_plan_dft_1d(N_y, in, out, FFTW_BACKWARD, FFTW_MEASURE);

    for (int k = 0; k < N_z_mpi; ++k) {
        for (int i = 0; i < N_x_mpi; ++i) {
            for (int j = 0; j < N_y; ++j) {
                in[j][0] = input[IDX3D_yfft(i, j, k)][0];  // real
                in[j][1] = input[IDX3D_yfft(i, j, k)][1];  // imag
            }

            fftw_execute(plan);

            for (int j = 0; j < N_y; ++j) {
                output[IDX3D_yfft(i, j, k)][0] = out[j][0] / sqrt(N_y);
                output[IDX3D_yfft(i, j, k)][1] = out[j][1] / sqrt(N_y);
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}