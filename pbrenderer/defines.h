#pragma once
// Struct alignment is handled differently between the CUDA compiler and other

// compilers (e.g. GCC, MS Visual C++ .NET)

#ifdef __CUDACC__

#define ALIGN(x)  __align__(x)

#else

#if defined(_MSC_VER) && (_MSC_VER >= 1300)

// Visual C++ .NET and later

#define ALIGN(x) __declspec(align(x))

#else

#if defined(__GNUC__)

// GCC

#define ALIGN(x)  __attribute__ ((aligned (x)))

#else

// all other compilers

#define ALIGN(x)

#endif

#endif

#endif

#define EPS 0.0001f

typedef unsigned int uint;
typedef unsigned char uchar;
