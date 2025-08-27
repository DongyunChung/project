V34 :0x24 wrapper_module
8 main.f90 S624 0
08/27/2025  21:30:00
use mpi_topology public 0 direct
use mpi_subdomain public 0 direct
use global public 0 direct
use mpi_poisson public 0 direct
use mpi_post public 0 direct
use mpi public 0 direct
use timer public 0 direct
use iso_fortran_env private
use iso_c_binding private
enduse
D 58 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 61 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 64 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 67 23 6 1 11 54 0 0 0 0 0
 0 54 11 11 54 54
D 70 23 6 1 11 55 0 0 0 0 0
 0 55 11 11 55 55
D 73 23 6 1 11 55 0 0 0 0 0
 0 55 11 11 55 55
D 76 26 687 8 686 7
D 85 26 690 8 689 7
S 624 24 0 0 0 9 1 0 5013 10005 0 A 0 0 0 0 B 0 5 0 0 0 0 0 0 0 0 0 0 14 0 0 0 0 0 0 0 0 5 0 0 0 0 0 0 wrapper_module
S 633 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 636 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 641 3 0 0 0 6 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6
S 642 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
S 643 3 0 0 0 7 1 1 0 0 0 A 0 0 0 0 B 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7
R 665 7 22 iso_fortran_env integer_kinds$ac
R 667 7 24 iso_fortran_env logical_kinds$ac
R 669 7 26 iso_fortran_env real_kinds$ac
R 686 25 7 iso_c_binding c_ptr
R 687 5 8 iso_c_binding val c_ptr
R 689 25 10 iso_c_binding c_funptr
R 690 5 11 iso_c_binding val c_funptr
R 724 6 45 iso_c_binding c_null_ptr$ac
R 726 6 47 iso_c_binding c_null_funptr$ac
R 727 26 48 iso_c_binding ==
R 729 26 50 iso_c_binding !=
A 13 2 0 0 0 6 633 0 0 0 13 0 0 0 0 0 0 0 0 0 0 0
A 30 2 0 0 0 6 636 0 0 0 30 0 0 0 0 0 0 0 0 0 0 0
A 32 2 0 0 0 6 641 0 0 0 32 0 0 0 0 0 0 0 0 0 0 0
A 54 2 0 0 0 7 642 0 0 0 54 0 0 0 0 0 0 0 0 0 0 0
A 55 2 0 0 0 7 643 0 0 0 55 0 0 0 0 0 0 0 0 0 0 0
A 61 1 0 1 0 58 665 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 67 1 0 1 0 64 667 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 71 1 0 3 0 70 669 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 125 1 0 0 0 76 724 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
A 128 1 0 0 0 85 726 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Z
J 69 1 1
V 61 58 7 0
R 0 61 0 0
A 0 6 0 0 1 3 1
A 0 6 0 0 1 30 1
A 0 6 0 0 1 32 1
A 0 6 0 0 1 13 0
J 71 1 1
V 67 64 7 0
R 0 67 0 0
A 0 6 0 0 1 3 1
A 0 6 0 0 1 30 1
A 0 6 0 0 1 32 1
A 0 6 0 0 1 13 0
J 73 1 1
V 71 70 7 0
R 0 73 0 0
A 0 6 0 0 1 32 1
A 0 6 0 0 1 13 0
J 133 1 1
V 125 76 7 0
S 0 76 0 0 0
A 0 6 0 0 1 2 0
J 134 1 1
V 128 85 7 0
S 0 85 0 0 0
A 0 6 0 0 1 2 0
Z
