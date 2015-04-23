g++ -Wall -std=c++11  -fopenmp -D_USE_OPENBLAS -O3 -L/PATH_TO_OPENBLAS/lib  -lopenblas   -I /PATH_TO_OPENBLAS/include -march=native -mavx2 -mfma compare_blas.cpp  -lopenblas -o m
