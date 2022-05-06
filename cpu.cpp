#include	<string.h>

#include	<iostream>
#include	<vector>
using namespace std;

#include	<chrono>
using namespace chrono;


#include	<immintrin.h>

void
DumpM512( __m512 _ ) {
	cerr << _[  0 ] << ',' << _[  1 ] << ',' << _[  2 ] << ',' << _[  3 ] << endl;
	cerr << _[  4 ] << ',' << _[  5 ] << ',' << _[  6 ] << ',' << _[  7 ] << endl;
	cerr << _[  8 ] << ',' << _[  9 ] << ',' << _[ 10 ] << ',' << _[ 11 ] << endl;
	cerr << _[ 12 ] << ',' << _[ 13 ] << ',' << _[ 14 ] << ',' << _[ 15 ] << endl;
}

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

struct
Args {
	float*	a;
	float*	b;
	float*	c;
};

#define	NUM_THREADS	64
void*
SubMatPro( void* _ ) {
	float* a = ((Args*)_)->a;
	float* b = ((Args*)_)->b;
	float* c = ((Args*)_)->c;
	for ( auto m = 0; m < ( M / NUM_THREADS ); m++ ) {
		for ( auto n = 0; n < N; n++ ) {
			auto $ = 0.;
			for ( auto k = 0; k < K; k++ ) $ += a[ m * K + k ] * b[ k * N + n ];
			c[ m * N + n ] = $;
		}
	}
	pthread_exit( NULL );
}

void
MatProMulti( float* a, float* b, float* c ) {
	pthread_t	p[ NUM_THREADS ];
	Args		args[ NUM_THREADS ];
	for ( auto _ = 0; _ < NUM_THREADS; _++ ) {
		args[ _ ].a = a + K * ( M / NUM_THREADS ) * _;
		args[ _ ].b = b;
		args[ _ ].c = c + N * ( M / NUM_THREADS ) * _;
		pthread_create( &p[ _ ], NULL, SubMatPro, args + _ );
	}
	for ( auto _ = 0; _ < NUM_THREADS; _++ ) {
		pthread_join( p[ _ ], NULL );
	}
}

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

#undef	NUM_THREADS
#define	NUM_THREADS	32
void*
SubMatPro512( void* _ ) {
	float* a = ((Args*)_)->a;
	float* b = ((Args*)_)->b;
	float* c = ((Args*)_)->c;
	for ( auto m = 0; m < ( M / NUM_THREADS ); m++ ) {
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
	pthread_exit( NULL );
}

void
MatPro512Multi( float* a, float* b, float* c ) {
	pthread_t	p[ NUM_THREADS ];
	Args		args[ NUM_THREADS ];
	for ( auto _ = 0; _ < NUM_THREADS; _++ ) {
		args[ _ ].a = a + K * ( M / NUM_THREADS ) * _;
		args[ _ ].b = b;
		args[ _ ].c = c + N * ( M / NUM_THREADS ) * _;
		pthread_create( &p[ _ ], NULL, SubMatPro512, args + _ );
	}
	for ( auto _ = 0; _ < NUM_THREADS; _++ ) {
		pthread_join( p[ _ ], NULL );
	}
}

/*
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

	{	auto start = system_clock::now();
		MatProMulti( a, b, c );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
		Check( c, ANS );
	}

	{	//	B BLOCK CONVERSION
		auto tmp = new ( align_val_t{ 64 } ) float[ K * N ];
		auto _ = 0;
		for ( auto n = 0; n < N; n++ ) {
			for ( auto k = 0; k < K; k++ ) {
				tmp[ _++ ] = b[ n + k * N ]; 
			}
		}
		delete[] b;
		b = tmp;
cerr << hex << long( b ) << " - " << long( b + K * N ) << dec << endl;
	}

	{	auto start = system_clock::now();
		MatPro512( a, b, c );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
		Check( c, ANS );
	}
	{	auto start = system_clock::now();
		MatPro512Multi( a, b, c );
		cerr << duration_cast<nanoseconds>( system_clock::now() - start ).count() << " ns" << endl;
		Check( c, ANS );
	}

	delete[] a;
	delete[] b;
	delete[] c;
}

int
main() {
	try {
		Main();
	} catch ( const char* _ ) {
		cerr << _ << endl;
	}
}

