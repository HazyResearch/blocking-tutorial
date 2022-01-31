# export PATH_TO_OPENBLAS=/usr/local/opt/openblas
# g++ -Wall -std=c++11 -Xpreprocessor -fopenmp -D_USE_OPENBLAS -O3 -L/PATH_TO_OPENBLAS/lib  -lopenblas   -I /PATH_TO_OPENBLAS/include -march=native -mavx2 -mfma compare_blas.cpp  -lopenblas -o m
OMP_NUM_THREADS=8 g++ -Wall -std=c++11 -fopenmp -D_USE_OPENBLAS -O3 -L/usr/lib  -lopenblas   -I /usr/include/openblas -march=native -mavx2 -mfma -funroll-loops compare_blas.cpp  -lopenblas -o matmul
