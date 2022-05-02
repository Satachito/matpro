#include	<string.h>

#include	<iostream>
#include	<vector>
using namespace std;

#include	<chrono>
using namespace chrono;


#include	<immintrin.h>

#define	D	1024
#define	M	D
#define	K	D
#define	N	D

void
MatPro( float* a, float* b, float* c ) {
#pragma omp parallel for
	for ( auto m = 0; m < M; m++ ) {
		auto cPtr = c + m * N;
		for ( auto n = 0; n < N; n++ ) {
			auto $ = _mm512_setzero_ps();
			for ( auto k = 0; k < K; k += 16 ) {
				$ = _mm512_fmadd_ps(
					_mm512_load_ps( a + m * K + k )
				,	_mm512_load_ps( b + n * K + k )
				,	$
				);
			}
			cPtr[ n ] = _mm512_reduce_add_ps( $ );
		}
	}
}

void
MatProCached( float* a, float* b, float* c ) {
	__m512 m512s[ K / 16 ];
	for ( auto m = 0; m < M; m++ ) {
		for ( auto k = 0; k < K; k += 16 ) {
			m512s[ k / 16 ] = _mm512_load_ps( a + m * K + k );
		}
		auto cPtr = c + m * N;
#pragma omp parallel for
		for ( auto n = 0; n < N; n++ ) {
			auto $ = _mm512_setzero_ps();
			for ( auto k = 0; k < K; k += 16 ) {
				$ = _mm512_fmadd_ps(
					m512s[ k / 16 ]
				,	_mm512_load_ps( b + n * K + k )
				,	$
				);
			}
			cPtr[ n ] = _mm512_reduce_add_ps( $ );
		}
	}
}

void
Check( float* a, float* b, float* c ) {
	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) {
				$ += a[ m * K + k ] * b[ n * K + k ];
			}
			if ( abs( c[ m * N + n ] - $ ) > 0.001 ) {
				cerr << m << ':' << n << ':' << $ << ':' << c[ m * N + n ] << endl;
				throw "eh?";
			}
		}
	}
}

void
Main() {
	auto a = new ( align_val_t{ 64 } ) float[ M * K ]();	//	Zero cleared
	for ( auto _ = 0; _ < M * K; _++ ) a[ _ ] = _ / (double)( M * K );
	auto b = new ( align_val_t{ 64 } ) float[ K * N ]();	//	Zero cleared
	for ( auto _ = 0; _ < K * N; _++ ) b[ _ ] = _ / (double)( K * N );
	auto c = new ( align_val_t{ 64 } ) float[ M * N ];
	
	{	auto start = system_clock::now();
		MatPro( a, b, c );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
	}
	Check( a, b, c );

	{	auto start = system_clock::now();
		MatProCached( a, b, c );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
	}
	Check( a, b, c );

	delete[] a;
	delete[] b;
	delete[] c;
}

int
main() {
	Main();
}

