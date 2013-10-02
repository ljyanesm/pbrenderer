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

#include "LYVertex.h"
#include "vector_types.h"
#include <vector_functions.h>
#include <device_functions.h>
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    float3 colliderPos;
    float  colliderRadius;

    float3 gravity;
    float globalDamping;
    float particleRadius;

    uint3 gridSize;
    uint numCells;
    float3 worldOrigin;
    float3 cellSize;

    uint numBodies;
    uint maxParticlesPerCell;

    float spring;
    float damping;
    float shear;
    float attraction;
    float boundaryDamping;

	float dmin;
	float alpha;
	float epsilon;
	float beta;
	float gamma;
	float phi;
	float forceSpring;

	float RMAX;
	float RMIN;

	float3 force;
	float3 Xc;
};

#endif
