nvcc -arch=sm_86 -lcublas -lcurand -DCUDA_LEGACY cc.cu
echo "LEGACY"
./a.out

nvcc -arch=sm_86 -lcublas -lcurand -DCUDA_MANAGED cc.cu
echo "MANAGED"
./a.out

rm ./a.out
