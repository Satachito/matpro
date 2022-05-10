nvcc -arch=sm_86 -DCUDA_MANAGED -O4 tc.cu
echo "MANAGED"
./a.out

nvcc -arch=sm_86 -DCUDA_LEGACY -O4 tc.cu
echo "LEGACY"
./a.out

rm ./a.out
