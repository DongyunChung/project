!>
!> @brief       Main execution program for the example problem of Poisson equation.
!>

module wrapper_module
    use mpi
    use mpi_topology
    use global
    use mpi_subdomain    
    use mpi_poisson

    use mpi_Post
    use timer
end module


program main
  
    use wrapper_module

    implicit none

    integer :: timestep, Tstepmax = 1
    integer :: ierr

    character(len=64)   :: timer_str(64)
    double precision    :: timer_a,timer_b
    integer, parameter  :: stamp_main = 1
    call initial

    ! Timer string setup. 
    timer_str(1)  = '[Main] poisson RHS             '
    timer_str(2)  = '[Main] poisson FFT             '
    timer_str(3)  = '[Main] ghostcell update        '
    timer_str(4)  = '[fft] TDMA                     '
    timer_str(5)  = '[fft] FFT                      '
    timer_str(6)  = '[fft] ALLTOALL                 '
    timer_str(7)  = '[fft] UPDATE SOLUTION          '
    timer_str(8)  = '[fft] others                   '
    timer_str(9)  = '[fft] build TDMA               '
    timer_str(10) = '[tdma] calc TDMA               '
    timer_str(11) = '[tdma] comm TDMA               '
    timer_str(12) = '[RHS] allocate'
    timer_str(13) = '[RHS] calc'

    if(myrank==0) write(*,*) '[Main] Iteration starts!'
    do timestep = 1, Tstepmax
        if(myrank==0) write(*,*) ''
        if(myrank==0) write(*,*) '[Main] tstep=', timestep

        call timer_init(13,timer_str)
        if(myrank==0) write(*,*) '[Main] Timer initialized!'
        call MPI_Barrier(MPI_COMM_WORLD,ierr)
    
        call timer_stamp0(stamp_main)
        call mpi_poisson_RHS
        call timer_stamp(1,stamp_main)
        call mpi_Poisson_FFT2(dx2_sub,dmx2_sub,P)
        call timer_stamp(2,stamp_main)
        call mpi_subdomain_ghostcell_update(P)
        call timer_stamp(3,stamp_main)

        ! call mpi_poisson_exact_sol()
        ! call mpi_Post_error(myrank, P, exact_sol, rms)
          
        ! call mpi_Post_FileOut_InstanField(myrank,P)

        call timer_reduction()
        call timer_output(myrank, nprocs)
        if(myrank==0) write(*,*) ''
    enddo
    
    call clean

end


subroutine initial

    use wrapper_module
    implicit none
    integer :: ierr

    call MPI_Init(ierr)
    call MPI_Comm_size( MPI_COMM_WORLD, nprocs, ierr)
    call MPI_Comm_rank( MPI_COMM_WORLD, myrank, ierr)

    if(myrank==0) write(*,*) '[Main] The simulation starts!'

    if(myrank==0) call system('mkdir -p ./data')
    if(myrank==0) call system('mkdir -p ./data/1_continue')
    if(myrank==0) call system('mkdir -p ./data/2_instanfield')
    call MPI_Barrier(MPI_COMM_WORLD,ierr)

    call global_inputpara()
    if(myrank==0) write(*,*) '[Main] Read input parameters!'

    call mpi_topology_make()
    call mpi_subdomain_make()
    call mpi_subdomain_mesh()
    call mpi_subdomain_indices()

    call mpi_subdomain_DDT_ghostcell()

    call mpi_poisson_allocation()

    call mpi_subdomain_DDT_transpose2()
    call mpi_poisson_wave_number()

    call mpi_subdomain_ghostcell_update(P)

    call MPI_Barrier(MPI_COMM_WORLD,ierr)

    ! call mpi_Post_allocation(1) ! something problem when use multi node.

    if(myrank==0) write(*,*) '[Main] Simulation setup completed!'

end subroutine initial


subroutine clean

    use wrapper_module
    implicit none
    integer :: ierr

    call mpi_poisson_clean()

    call mpi_subdomain_indices_clean()
    call mpi_subdomain_clean()

    call mpi_topology_clean()

    call MPI_FINALIZE(ierr)

    if(myrank==0) write(*,*) '[Main] The main simulation complete! '
    
end subroutine clean