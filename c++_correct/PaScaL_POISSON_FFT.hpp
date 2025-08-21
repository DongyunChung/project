#ifndef TDMA_HPP
#define TDMA_HPP

#include <vector>
#include <mpi.h>

#define l_x 1.0
#define l_y 1.0
#define l_z 1.0
#define dx (1.0 * l_x / N_x)
#define dy (1.0 * l_y / N_y)
#define dz (1.0 * l_y / N_z)

#define N_x 64
#define N_y 64
#define N_z 64
#define p_z 4
#define p_y 4
#define N_z_mpi (N_z / p_z) // MPI SIZE = 8
#define N_y_mpi (N_y / p_y) // MPI SIZE = 8
#define N_kx (((N_x) / 2) + 1)
#define original_N_x_mpi (N_x /(2*p_y))
#define TDMA_many_size (N_y / p_z)



#define pi 3.14159265358979323846

#define IDX3D(i, j, k) ((k) * (N_y_mpi * N_x) + (j) * N_x + (i))

#define IDX3D_xfft(i, j, k) ((k) * (N_y_mpi * N_kx) + (j) * N_kx + (i))
#define IDX3D_yfft(i, j, k) ((k) * (N_x_mpi * N_y) + (j) * N_x_mpi + (i))
#define IDX3D_C2(ix, jy, k) ((k) * N_y * N_x + (jy) * N_x + (ix))

#define IDX_TDMA(i, ix, jy) ((i) * (N_x_mpi * (N_y / p_z)) + (jy) * (N_x_mpi) + (ix))

#define IDX_send(kz, jy, ix) ( (kz) * ((N_y / p_z) * (p_z - 1) * (N_x_mpi)) + (jy) * (N_x_mpi) + (ix))
#define IDX_recv(kz, jy, ix) ( (kz) * ((N_y / p_z) * (N_x_mpi)) + (jy) * (N_x_mpi) + (ix))
// #define IDX_recv(kz, ix, jy, real_or_imag) ( ((kz) + ((6 * p_z) * real_or_imag)) * (N_x_mpi * (N_y / p_z)) + (jy) * (N_x_mpi) + (ix))

extern int rank;
extern int size;
extern int my_floor;
extern int my_col;
extern int N_x_mpi;

extern double b_0;
extern double b_N;
extern double a_0;
extern double c_N;
extern double a;
extern double b;
extern double c;
extern MPI_Datatype recv_type;

void init_mpi(int *argc, char ***argv);
void fill_source_array(std::vector<double>& source_array);
void fill_analytic_solution(std::vector<double>& analytic_solution);

void fft_x_direction(
    const double* input_data, // size: N_z_mpi * N_y_mpi * N_x
    fftw_complex* output_data // size: N_z_mpi * N_y_mpi * (N_x / 2 + 1)
);
void ifft_x_direction(
    const fftw_complex* input_data, // size: N_z_mpi * N_y_mpi * N_x
    double* output_data // size: N_z_mpi * N_y_mpi * (N_x / 2 + 1)
);
void transpose_xy_pencil_corrected(
    fftw_complex* send_buf,
    fftw_complex* recv_buf,
    int N_x_mpi,
    int my_col,
    MPI_Comm comm,
    MPI_Request* request
);

void transpose_yx_pencil_corrected(
    fftw_complex* send_buf,
    fftw_complex* recv_buf,
    int N_x_mpi,
    int my_col,
    MPI_Comm comm,
    MPI_Request* request
);

void fft_y_direction(
    std::vector<fftw_complex>& input,  // input: z-x-y (전치 후 결과)
    std::vector<fftw_complex>& output,  // output: z-x-ky (FFT된 결과)
    int N_x_mpi
);

void ifft_y_direction(
    std::vector<fftw_complex>& input,   // input: z-x-ky (FFT된 입력)
    std::vector<fftw_complex>& output,  // output: z-x-y (전치 후 실공간)
    int N_x_mpi
);

void write_fft_result_components(const std::vector<fftw_complex>& complex_array,
                                 const std::string& real_filename,
                                 const std::string& imag_filename,
                                 int z_size,
                                 int y_size,
                                 int x_size);

void write_fft_result_x(const std::vector<fftw_complex>& fft_result,
                                 const std::string& real_filename,
                                 const std::string& imag_filename);

void write_scalar_components(const std::vector<double>& solution,
                                 const std::string& filename,
                                 int z_size,
                                 int y_size,
                                 int x_size);

void complex_add(fftw_complex a, fftw_complex b, fftw_complex result);
void complex_subtract(double areal,double acomp, fftw_complex b, fftw_complex result);
void complex_mult(double areal,double acomp, fftw_complex b, fftw_complex result);
void complex_divide(fftw_complex a, fftw_complex b, fftw_complex result);

void make_modified_TDM(int ix, int global_ix, int jy, int recv_floor, int recv_rank, int recv_data_j,std::vector<fftw_complex>& fft_final, std::vector<fftw_complex>& d_star, std::vector<double>& c_star, std::vector<double>& a_star);
void make_send_data(int ix, int jy, int send_data_j, std::vector<double>& send_data, std::vector<fftw_complex>& d_star, std::vector<double>& c_star, std::vector<double>& a_star);
void make_recv_data(int ix, int jy, int recv_data_j, std::vector<double>& recv_data_per_node, std::vector<fftw_complex>& d_star, std::vector<double>& c_star, std::vector<double>& a_star);
void thomas_algorithm(std::vector<double>& recv_data_per_node, std::vector<fftw_complex>& solution_to_Thomas_algorithm, int ix, int recv_data_j, std::vector<fftw_complex>& c_star, std::vector<fftw_complex>& d_star);
void PaScaL_TDMA(std::vector<fftw_complex>& fft_final, std::vector<fftw_complex>& d_star, std::vector<double>& recv_data_per_node, std::vector<fftw_complex>& solution_to_Thomas_algorithm, std::vector<fftw_complex>& c_star_single, std::vector<fftw_complex>& d_star_single);

#endif