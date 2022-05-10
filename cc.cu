#include <iostream>
#include <iomanip>
#include <curand.h>

using namespace std;

#include "JP/CUDA/JPCuda.h"

#include <chrono>
using namespace chrono;

template < size_t K_SIZE, size_t N_SIZE > __global__ void
MatPro(
	const	float* a
,	const	float* b
,			float* c
) {
	auto $ = 0.;
	auto n = blockIdx.x * blockDim.x + threadIdx.x;
	auto m = blockIdx.y * blockDim.y + threadIdx.y;
	for ( auto k = 0; k < K_SIZE; k++ ) $ += a[ m * K_SIZE + k ] * b[ k * N_SIZE + n ];
	c[ m * N_SIZE + n ] = $;
//printf( "%02d %02d %02ld\n", n, m, m * N_SIZE + n );
}

#include "CONSTANTS.h"

void
Main() {

//	cerr << fixed << setprecision( 3 );

	CUDAMemory<float>	a( M * K );
	DummyData( a );

	CUDAMemory<float>	b( K * N );
	DummyData( b );

	CUDAMemory<float>	c( M * N );

	auto timer = system_clock::now();
	MatPro< K, N ><<< dim3( N / 32, M / 32 ), dim3( 32, 32 ) >>>( a.$, b.$, c.$ );
	cudaDeviceSynchronize();
	c.DtoH();
	printf( "%ld ns\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() );

	a.DtoH();
	b.DtoH();

c.Host()[ M * N - 1 ] = -1;
	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) $ += a.Host()[ m * K + k ] * b.Host()[ k * N + n ];
			auto _ = c.Host()[ m * N + n ];
			if ( abs( $ - _ ) > 0.01 ) {
				cerr << m << ',' << n << ' ' << $ << ':' << _ << ':' << abs( $ - _ ) << endl;
				throw "eh?";
			}
		}
	}
}

int
main( int argc, char* argv[] ) {
	cudaDeviceProp _;
	cudaGetDeviceProperties( &_, 0 );
	cerr << "maxThreadsPerBlock: " << _.maxThreadsPerBlock << endl;
	Main();
}

