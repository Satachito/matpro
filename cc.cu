#include <iostream>
#include <iomanip>
#include <curand.h>

using namespace std;

#include <chrono>
using namespace chrono;

#include "JP/CUDA/JPCuda.h"
using namespace nvcuda;

template < typename Fa, typename Fb, typename Fc > __global__ void
matmal_16x16(
	const	Fa* a
,	const	Fb* b
,			Fc* c
,	size_t	M
,	size_t	N
,	size_t	K
) {
	auto $ = 0.;
	auto n = blockIdx.x * blockDim.x + threadIdx.x;
	auto m = blockIdx.y * blockDim.y + threadIdx.y;
	for ( auto k = 0; k < K; k++ ) $ += float( a[ m * K + k ] ) * float( b[ k * N + n ] );
	c[ m * N + n ] = $;
}

#define	D	4096
#define	M	D
#define	K	D
#define	N	D

struct
CudaTimer {
	cudaEvent_t	start;
	cudaEvent_t	stop;

	CudaTimer() {
		_C( cudaEventCreate( &start ) );
		_C( cudaEventCreate( &stop ) );
		_C( cudaEventRecord( start ) );
	}
	~
	CudaTimer() {
		_C( cudaEventRecord( stop ) );
		_C( cudaEventSynchronize( stop ) );
		float _ = 0;
		_C( cudaEventElapsedTime( &_, start, stop ) );
		printf( "CudaTimer: %f ms\n", _ );
		_C( cudaEventDestroy( start ) );
		_C( cudaEventDestroy( stop ) );
	}
};

void
Main() {

//	cerr << fixed << setprecision( 3 );

	CUDAMemory< half > a( M * K );
	DummyData( a );
	a.DtoH();
//	a.Dump< K >();

	CUDAMemory< half > b( K * N );
	DummyData( b );
	b.DtoH();
//	b.Dump< N >();

	CUDAMemory< float > c( M * N );
//	c.Zeroset();
//	cudaDeviceSynchronize();

	auto timer = system_clock::now();
	matmal_16x16<<< dim3( M / 32, N / 32 ), dim3( 32, 32 ) >>>( a.$, b.$, c.$, M, N, K );	//	32: FIXED NUMBER warp size
	//	Managed でやるときはこれが必須！
	cudaDeviceSynchronize();
	printf( "%f ms\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() / 1000000. );
	c.DtoH();
	printf( "%f ms\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() / 1000000. );
//	c.Dump< N >();

//	for ( auto m = 0; m < M; m++ ) {
//		for ( auto n = 0; n < N; n++ ) {
//			auto $ = 0.;
//			for ( auto k = 0; k < K; k++ ) $ += float( a( m * K + k ) ) * float( b( k * N + n ) );
//			auto _ = float( c( m * N + n ) );
//			if ( abs( $ - _ ) > 0.01 ) cerr << m << ',' << n << ' ' << $ << ':' << _ << ':' << abs( $ - _ ) << endl;
//		}
//	}
}

int
main( int argc, char* argv[] ) {
	cudaDeviceProp _;
	cudaGetDeviceProperties( &_, 0 );
	cerr << "maxThreadsPerBlock: " << _.maxThreadsPerBlock << endl;
	Main();
}

