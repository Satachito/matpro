#include	<string.h>

#include	<iostream>
#include	<vector>
using namespace std;

#include	<chrono>
using namespace chrono;


#include	<immintrin.h>

#define	M	512
#define	K	1024
#define	N	2048

void
MatPro( float* a, float* b, float* c ) {
	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) $ += a[ m * K + k ] * b[ k * N + n ];
			c[ m * N + n ] = $;
		}
	}
}

/*	NEED MAJOR CONVERSION
void
MatPro512( float* a, float* b, float* c ) {
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
MatPro512Cached( float* a, float* b, float* c ) {
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
*/

//	NEED BLOCK CONVERSION
void
MatPro512GEMM( float* a, float* b, float* c ) {
	for ( auto m = 0; m < M; m += 4 ) {
		for ( auto n = 0; n < N; n += 4 ) {
			auto $ = _mm512_setzero_ps();
			for ( auto k = 0; k < K; k += 4 ) {
				$ = _mm512_fmadd_ps(
					_mm512_load_ps( a + m * K + k * 4 )
				,	_mm512_load_ps( b + k * N + n * 4 )
				,	$
				);
			}
			_mm512_store_ps( c + m * N + n * 4, $ );
		}
	}
}

void
Check( float* p, float* q ) {
	for ( auto m = 0; m < M; m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto _ = m * N + n;
			if ( abs( p[ _ ] - q[ _ ] ) > 0.001 ) {
				cerr << m << ':' << n << ':' << p[ _ ] << ':' << q[ _ ] << endl;
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
	
	auto ANS = new ( align_val_t{ 64 } ) float[ M * N ];
	{	auto start = system_clock::now();
		MatPro( a, b, ANS );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
	}

	{	//	BLOCK CONVERSION
		auto tmp = new ( align_val_t{ 64 } ) float[ M * K ]();	//	Zero cleared
		auto _ = 0;
		for ( auto m = 0; m < M; m += 4 ) {
			for ( auto k = 0; k < K; k += 4 ) {
				auto $ = a + m * K + k;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += K;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += K;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += K;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
			}
		}
		delete[] a;
		a = tmp;
	}

//for ( auto _ = 0; _ < 1024; ) {
//	cerr << ' ' << a[ _ ];
//	if ( ++_ % 16 == 0 ) cerr << endl;
//}

	{	//	BLOCK CONVERSION
		auto tmp = new ( align_val_t{ 64 } ) float[ K * N ]();	//	Zero cleared
		auto _ = 0;
		for ( auto k = 0; k < K; k += 4 ) {
			for ( auto n = 0; n < N; n += 4 ) {
				auto $ = b + k * N + n;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += N;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += N;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += N;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
			}
		}
		delete[] b;
		b = tmp;
	}

	{	auto start = system_clock::now();
		MatPro512GEMM( a, b, c );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
	}
	{	//	BLOCK REVERSE CONVERSION
		auto tmp = new ( align_val_t{ 64 } ) float[ K * N ]();	//	Zero cleared
		auto _ = 0;
		for ( auto k = 0; k < K; k += 4 ) {
			for ( auto n = 0; n < N; n += 4 ) {
				auto $ = b + k * N + n;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += N;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += N;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
				$ += N;
				tmp[ _++ ] = *$; tmp[ _++ ] = *( $ + 1 ); tmp[ _++ ] = *( $ + 2 ); tmp[ _++ ] = *( $ + 3 );
			}
		}
		delete[] b;
		b = tmp;
	}

	Check( c, ANS );

	delete[] a;
	delete[] b;
	delete[] c;
}

int
main() {
	Main();
}

