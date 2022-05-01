nvcc -arch=sm_86 -lcublas -lcurand -DCUDA_LEGACY tc.cu
echo "LEGACY"
./a.out

nvcc -arch=sm_86 -lcublas -lcurand -DCUDA_MANAGED tc.cu
echo "MANAGED"
./a.out

rm ./a.out
