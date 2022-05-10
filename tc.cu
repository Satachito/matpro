#include <iostream>
#include <iomanip>
#include <curand.h>

using namespace std;

#include <chrono>
using namespace chrono;

#include "JP/CUDA/JPCuda.h"
using namespace nvcuda;

#include "CONSTANTS.h"

template < typename F > __global__ void
MatPro(
	const	half*	_a
,	const	half*	_b
,			F*		_c
) {
	wmma::fragment< wmma::matrix_a, 16, 16, 16, half, wmma::row_major > a;
	wmma::fragment< wmma::matrix_b, 16, 16, 16, half, wmma::row_major > b;
	wmma::fragment< wmma::accumulator, 16, 16, 16, F > c;

	wmma::fill_fragment( c, 0 );

	for ( auto k = 0; k < K; k += 16 ) {
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
	MatPro<<< dim3( N / 16, M / 16 ), 32 >>>( a.$, b.$, c.$ );	//	32: WARP SIZE ( FIXED NUMBER ) 
	cudaDeviceSynchronize();
	c.DtoH();
	printf( "%ld ns\n", duration_cast<std::chrono::nanoseconds>( system_clock::now() - timer ).count() );

	a.DtoH();
	b.DtoH();

	c.Host()[ M * N - 1 ] = -1;
	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) {
				$ += float( a.Host()[ m * K + k ] ) * float( b.Host()[ k * N + n ] );
			}
			auto _ = float( c.Host()[ m * N + n ] );
			if ( abs( $ - _ ) > 1 ) {
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

