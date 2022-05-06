#include <iostream>
#include <iomanip>
#include <curand.h>

using namespace std;

#include <chrono>
using namespace chrono;

#include "JP/CUDA/JPCuda.h"
using namespace nvcuda;

#include "CONSTANTS.h"

template < typename Fa, typename Fb, typename Fc > __global__ void
MatM(
	const	Fa* a
,	const	Fb* b
,			Fc* c
) {
	__shared__	float	A[ 32 ][ 32 ];
	__shared__	float	B[ 32 ][ 32 ];

	auto n = blockIdx.x * 32;
	auto m = blockIdx.y * 32;

	auto $ = 0.;
	for ( auto k = 0; k < M; k += 32 ) {
		A[ threadIdx.y ][ threadIdx.x ] = a[ ( m + threadIdx.y ) * K + k + threadIdx.x ];
		B[ threadIdx.y ][ threadIdx.x ] = b[ ( k + threadIdx.y ) * N + n + threadIdx.x ];
		__syncthreads();

		for ( auto _ = 0; _ < 32; _++ ) $ += A[ threadIdx.y ][ _ ] * B[ _ ][ threadIdx.x ];
	}
	c[ ( m + threadIdx.y ) * N + n + threadIdx.x ] = $;
}

void
Main() {
	CUDAMemory< half >	a( M * K );
	DummyData( a );

	CUDAMemory< half >	b( K * N );
	DummyData( b );

	CUDAMemory< float >	c( M * N );

auto timer = system_clock::now();

	MatM<<< dim3( M / 32, N / 32 ), dim3( 32, 32 ) >>>( a.$, b.$, c.$ );
	cudaDeviceSynchronize();

printf( "%ld ns\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() );
	c.DtoH();
printf( "%ld ns\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() );

	a.DtoH();
	b.DtoH();

	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) $ += float( a( m * K + k ) ) * float( b( k * N + n ) );
			auto _ = float( c( m * N + n ) );
			if ( abs( $ - _ ) > 0.05 ) cerr << m << ',' << n << ' ' << $ << ':' << _ << ':' << abs( $ - _ ) << endl;
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

