#include "modules.hpp"
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

static void initial(int argc, char** argv);
static void clean();

int main(int argc, char** argv) {
    initial(argc, argv);

    int timestep;
    int Tstepmax = 1;
    std::array<std::string,64> timer_str;
    int stamp_main = 1;

    timer_str[0]  = "[Main] poisson RHS             ";
    timer_str[1]  = "[Main] poisson FFT             ";
    timer_str[2]  = "[Main] ghostcell update        ";
    timer_str[3]  = "[fft] TDMA                     ";
    timer_str[4]  = "[fft] FFT                      ";
    timer_str[5]  = "[fft] ALLTOALL                 ";
    timer_str[6]  = "[fft] UPDATE SOLUTION          ";
    timer_str[7]  = "[fft] others                   ";
    timer_str[8]  = "[fft] build TDMA               ";
    timer_str[9]  = "[tdma] calc TDMA               ";
    timer_str[10] = "[tdma] comm TDMA               ";
    timer_str[11] = "[RHS] allocate";
    timer_str[12] = "[RHS] calc";

    if (mpi_topology::myrank == 0)
        std::cout << "[Main] Iteration starts!" << std::endl;

    for (timestep = 1; timestep <= Tstepmax; ++timestep) {
        if (mpi_topology::myrank == 0) {
            std::cout << std::endl;
            std::cout << "[Main] tstep=" << timestep << std::endl;
        }

        std::vector<std::string> tvec(timer_str.begin(), timer_str.end());
        timer::timer_init(13, tvec);
        if (mpi_topology::myrank == 0)
            std::cout << "[Main] Timer initialized!" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);

        timer::timer_stamp0(stamp_main);
        mpi_poisson::mpi_poisson_RHS();
        timer::timer_stamp(1, stamp_main);
        mpi_poisson::mpi_Poisson_FFT2(mpi_subdomain::dx2_sub, mpi_subdomain::dmx2_sub);
        timer::timer_stamp(2, stamp_main);
        mpi_subdomain::mpi_subdomain_ghostcell_update(mpi_poisson::P.data());
        timer::timer_stamp(3, stamp_main);
        /////////////////////////////////////////////////////////////////////
        if (timestep == Tstepmax) {
            mpi_poisson::mpi_poisson_exact_sol();
            mpi_post::mpi_Post_error(mpi_topology::myrank,
                                     mpi_poisson::P,
                                     mpi_poisson::exact_sol,
                                     global::rms);

            using namespace mpi_subdomain;
            std::vector<double> local_P(static_cast<std::size_t>(n1msub) * n2msub * n3msub);
            auto idx_full = [=](int i, int j, int k) {
                return static_cast<std::size_t>((k * (n2sub + 1) + j) * (n1sub + 1) + i);
            };
            for (int k = 1; k <= n3msub; ++k)
                for (int j = 1; j <= n2msub; ++j)
                    for (int i = 1; i <= n1msub; ++i) {
                        std::size_t iloc = static_cast<std::size_t>((k-1) * n2msub * n1msub + (j-1) * n1msub + (i-1));
                        local_P[iloc] = mpi_poisson::P[idx_full(i,j,k)];
                    }
            std::string fname = "solution_rank" + std::to_string(mpi_topology::myrank) + ".txt";
            mpi_post::write_scalar_components(local_P, fname, n3msub, n2msub, n1msub);
        }
        /////////////////////////////////////////////////////////////////////
        timer::timer_reduction();
        timer::timer_output(mpi_topology::myrank, mpi_topology::nprocs);
        if (mpi_topology::myrank == 0)
            std::cout << std::endl;
    }

    clean();
    return 0;
}

static void initial(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_topology::nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_topology::myrank);

    if (mpi_topology::myrank == 0)
        std::cout << "[Main] The simulation starts!" << std::endl;

    if (mpi_topology::myrank == 0) {
        system("mkdir -p ./data");
        system("mkdir -p ./data/1_continue");
        system("mkdir -p ./data/2_instanfield");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    global::global_inputpara();
    if (mpi_topology::myrank == 0)
        std::cout << "[Main] Read input parameters!" << std::endl;

    mpi_topology::mpi_topology_make();
    mpi_subdomain::mpi_subdomain_make();
    
    mpi_subdomain::mpi_subdomain_mesh();
    
    mpi_subdomain::mpi_subdomain_indices();
    mpi_subdomain::mpi_subdomain_DDT_ghostcell();
    mpi_poisson::mpi_poisson_allocation();
    mpi_subdomain::mpi_subdomain_DDT_transpose2();
    mpi_poisson::mpi_poisson_wave_number();
    mpi_subdomain::mpi_subdomain_ghostcell_update(mpi_poisson::P.data());
    MPI_Barrier(MPI_COMM_WORLD);

    // mpi_Post_allocation(1); // something problem when use multi node.

    if (mpi_topology::myrank == 0)
        std::cout << "[Main] Simulation setup completed!" << std::endl;
}

static void clean() {
    mpi_poisson::mpi_poisson_clean();
    mpi_subdomain::mpi_subdomain_indices_clean();
    mpi_subdomain::mpi_subdomain_clean();
    mpi_topology::mpi_topology_clean();
    MPI_Finalize();
    if (mpi_topology::myrank == 0)
        std::cout << "[Main] The main simulation complete! " << std::endl;
}