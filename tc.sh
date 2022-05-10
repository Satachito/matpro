nvcc	\
	-DCUDA_MANAGED	\
	-arch=sm_86	\
	--std=c++17	\
	-gencode arch=compute_70,code=sm_70	\
	-gencode arch=compute_75,code=sm_75	\
	-gencode arch=compute_80,code=sm_80	\
	-gencode arch=compute_86,code=compute_86	\
	tc.cu

echo "MANAGED"
./a.out

nvcc	\
	-DCUDA_LEGACY	\
	-arch=sm_86	\
	--std=c++17	\
	-gencode arch=compute_70,code=sm_70	\
	-gencode arch=compute_75,code=sm_75	\
	-gencode arch=compute_80,code=sm_80	\
	-gencode arch=compute_86,code=compute_86	\
	tc.cu

echo "LEGACY"
./a.out

rm ./a.out
