#nvcc -arch=sm_86 -DCUDA_LEGACY -O4 ccShared.cu
#echo "LEGACY"
#./a.out

nvcc -arch=sm_86 -DCUDA_MANAGED -O4 ccShared.cu
echo "MANAGED"
./a.out

rm ./a.out
