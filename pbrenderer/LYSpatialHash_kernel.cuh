#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#define INF 0x7f800000
#define NINF 0xff800000

#include "defines.h"
#include "LYVertex.h"
#include "vector_types.h"
#include "helper_math.h"
#include <math_constants.h>
#include <vector_functions.h>
#include <device_functions.h>
typedef unsigned int uint;

// simulation parameters

typedef struct ALIGN(16) _SimParams
{
	float3 force;
	float3 colliderPos;
	float3 Xc;
    float3 worldOrigin;
    float3 cellSize;
	float3 Ax;
	float3 Nx;

	uint3 gridSize;

	float dmin;
	float w_tot;
	float R;
	float RMAX;
	float RMIN;
	float colliderRadius;
    
	uint numCells;
    uint numBodies;
    uint maxParticlesPerCell;

	
}SimParams;

#endif
