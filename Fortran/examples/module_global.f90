!======================================================================================================================
!> @file        module_global.f90
!> @brief       This file contains a module of global input parameters for the example problem of PaScaL_TCS.
!> @details     The input parameters include global domain information, boundary conditions, fluid properties, 
!>              flow conditions, and simulation control parameters.
!> @author      
!>              - Kiha Kim (k-kiha@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
!>              - Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
!>              - Jung-Il Choi (jic@yonsei.ac.kr), Department of Computational Science & Engineering, Yonsei University
!>
!> @date        October 2022
!> @version     1.0
!> @par         Copyright
!>              Copyright (c) 2022 Kiha Kim and Jung-Il choi, Yonsei University and 
!>              Ji-Hoon Kang, Korea Institute of Science and Technology Information, All rights reserved.
!> @par         License     
!>              This project is release under the terms of the MIT License (see LICENSE in )
!======================================================================================================================

!>
!> @brief       Module for global parameters.
!> @details     This global module has simulation parameters and a subroutine to initialize the parameters. 
!>
module global
    implicit none
    double precision, parameter :: PI = acos(-1.d0)

    ! Computational size for the physical domain and time discretization
    integer :: n1,n2,n3
    integer :: n1m,n2m,n3m
    integer :: n1p,n2p,n3p
    integer :: np1,np2,np3

    logical :: pbc1, pbc2, pbc3
    integer :: UNIFORM1,UNIFORM2,UNIFORM3
    double precision :: GAMMA1, GAMMA2, GAMMA3

    double precision :: rms,rms_local

    ! Physical size of the computational domain
    double precision :: L1,L2,L3
    double precision :: H,Aspect1,Aspect2,Aspect3
    double precision :: x1_start,x2_start,x3_start
    double precision :: x1_end,x2_end,x3_end

    character(len=128) dir_cont_fileout
    character(len=128) dir_instantfield
    character(len=128) dir_cont_filein
    integer :: print_start_step,print_interval_step,print_j_index_wall,print_j_index_bulk,print_k_index
    integer :: ContinueFilein
    integer :: ContinueFileout
    
    contains
    !>
    !> @brief       Assign global parameters.
    !> @param       np_dim      Number of MPI processes in 3D topology
    !>

    subroutine global_inputpara()

        implicit none
        double precision :: tttmp(1:3)
        character(len=512) :: temp_char
        integer :: i

        ! Namelist variables for file input
        namelist /meshes/               n1m, n2m, n3m
        namelist /MPI_procs/            np1, np2, np3
        namelist /periodic_boundary/    pbc1, pbc2, pbc3
        namelist /uniform_mesh/         uniform1, uniform2, uniform3
        namelist /mesh_stretch/         gamma1, gamma2, gamma3
        namelist /aspect_ratio/         H, Aspect1, Aspect2, Aspect3
        namelist /sim_continue/         ContinueFilein, ContinueFileout, dir_cont_filein, dir_cont_fileout, dir_instantfield


        ! Using file input
        open(unit = 1, file = "PARA_INPUT.dat")
        read(1, sim_continue)
        read(1, meshes)
        read(1, MPI_procs)
        read(1, periodic_boundary)
        read(1, uniform_mesh)
        read(1, mesh_stretch)
        read(1, aspect_ratio)
        close(1)


        ! Computational size for the physical domain and time discretization
        n1=n1m+1;n1p=n1+1;
        n2=n2m+1;n2p=n2+1;
        n3=n3m+1;n3p=n3+1;

        ! Physical size of the computational domain
        L1=H*Aspect1
        L2=H*Aspect2
        L3=H*Aspect3

        x1_start = 0.0d0
        x2_start = 0.0d0
        x3_start = 0.0d0

        x1_end=x1_start+L1
        x2_end=x2_start+L2
        x3_end=x3_start+L3
        
        tttmp(1)=L1/dble(n1-1)
        tttmp(2)=L2/dble(n2-1)
        tttmp(3)=L3/dble(n3-1)

        
    end subroutine global_inputpara

end module global