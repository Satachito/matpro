nvcc -arch=sm_86 -lcublas -lcurand -DCUDA_LEGACY ccShared.cu
echo "LEGACY"
./a.out

nvcc -arch=sm_86 -lcublas -lcurand -DCUDA_MANAGED ccShared.cu
echo "MANAGED"
./a.out

rm ./a.out
