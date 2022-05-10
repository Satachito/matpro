nvcc	\
	-DCUDA_MANAGED	\
	--std=c++17	\
	-O4	\
	-o _	\
	cc.cu

echo 'Managed'
./_

nvcc	\
	-DCUDA_LEGACY	\
	--std=c++17	\
	-O4	\
	-o _	\
	cc.cu

echo 'Legacy'
./_

#	-gencode arch=compute_80,code=sm_80	\
#	-gencode arch=compute_86,code=compute_86	\
