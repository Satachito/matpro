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
MatPro(
	const	Fa* _a
,	const	Fb* _b
,			Fc* _c
) {
//printf( "TX: %d %d %d %d %d %d %d\n", gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x );
// A,B,Cのfragment

	wmma::fragment< wmma::matrix_a, 16, 16, 16, Fa, wmma::row_major > a;
	wmma::fragment< wmma::matrix_b, 16, 16, 16, Fb, wmma::row_major > b;
	wmma::fragment< wmma::accumulator, 16, 16, 16, Fc > c;

	wmma::fill_fragment( c, 0 );

	for ( auto k = 0; k < K; k += 16 ) {
//if ( threadIdx.x == 0 ) printf( "%d %d %d %lu %lu\n", blockIdx.x, blockIdx.y, k, ( blockIdx.y * K * 16 + k ), ( k * N + blockIdx.x * 16 ) );
		wmma::load_matrix_sync( a, _a + ( blockIdx.y * K * 16 + k ), K );
		wmma::load_matrix_sync( b, _b + ( k * N + blockIdx.x * 16 ), N );
		wmma::mma_sync( c, a, b, c );
	}

	wmma::store_matrix_sync( _c + ( blockIdx.y * N * 16 + blockIdx.x * 16 ), c, N, wmma::mem_row_major );
}

void
Main() {

	CUDAMemory< half > a( M * K );
	DummyData( a );

	CUDAMemory< half > b( K * N );
	DummyData( b );

	CUDAMemory< float > c( M * N );

	auto timer = system_clock::now();
	MatPro<<< dim3( N / 16, M / 16 ), 32 >>>( a.$, b.$, c.$ );	//	32: FIXED NUMBER warp size
	cudaDeviceSynchronize();
	printf( "%ld ns\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() );
	c.DtoH();
	printf( "%ld ns\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() );

	a.DtoH();
	b.DtoH();

c.$[ M * N - 1 ] = 828;
	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) {
				cerr << float( a( m * K + k ) ) << ':' << float( b( k * N + n ) ) << ':' << endl;
				$ += float( a( m * K + k ) ) * float( b( k * N + n ) );
			}
			auto _ = float( c( m * N + n ) );
		//	if ( abs( $ - _ ) > 0.01 ) cerr << m << ',' << n << ' ' << $ << ':' << _ << ':' << abs( $ - _ ) << endl;
			cerr << m << ',' << n << ' ' << $ << ':' << _ << ':' << abs( $ - _ ) << endl;
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

