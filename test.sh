# export PATH_TO_OPENBLAS=/usr/local/opt/openblas
# g++ -Wall -std=c++11 -Xpreprocessor -fopenmp -D_USE_OPENBLAS -O3 -L/PATH_TO_OPENBLAS/lib  -lopenblas   -I /PATH_TO_OPENBLAS/include -march=native -mavx2 -mfma compare_blas.cpp  -lopenblas -o m
threads=$1  # The custom variable (number of threads) passed as the first argument
if [ -z "$threads" ]; then
  threads=8
fi
export OMP_NUM_THREADS=$threads

g++ -Wall -std=c++11 -fopenmp -D_USE_OPENBLAS -O3 \
-L/usr/lib  -lopenblas   -I /usr/include/openblas \
-march=native -mavx2 -mfma -funroll-loops \
-DOMP_NUM_THREADS_MACRO=$threads \
compare_blas.cpp  -lopenblas -o matmul

./matmul

