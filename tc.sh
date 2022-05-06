#nvcc -arch=sm_86 -DCUDA_LEGACY tc.cu
#echo "LEGACY"
#./a.out

nvcc -arch=sm_86 -DCUDA_MANAGED -O4 tc.cu
echo "MANAGED"
./a.out

rm ./a.out
