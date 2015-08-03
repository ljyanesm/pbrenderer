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

#include <glm\glm.hpp>
#include "defines.h"
#include "LYVertex.h"

#include "vector_types.h"
#include "helper_math.h"
#include <math_constants.h>
#include <vector_functions.h>
#include <device_functions.h>
typedef unsigned int uint;

// simulation parameters

typedef struct _CollisionInfo{
	glm::vec4 Ax;
	glm::vec4 Nx;
	glm::vec4 force;

	float w_tot;
	float wn_tot;

} CollisionInfo;

typedef struct _SimParams
{
	float3 force;
    float3 cellSize;
	float3 Ax;
	float3 Nx;

	uint3 gridSize;

	float dmin;
	float w_tot;
	float wn_tot;
	float R;
	float RMAX;
	float RMIN;
	float colliderRadius;
    
	uint numCells;
    size_t numBodies;
    uint maxParticlesPerCell;
	
}SimParams;

enum CollisionCheckType{
	BASIC,
	NAIVE,
	DYNAMIC,
	TWO_STEP,
	NUM_TYPES
};

class ccConfiguration { // Collision check configuration call object
public:
	float3					pos;
	float					pad;
	float4					forceVector; 
	size_t					numVertices;
	size_t					numToolVertices;
	float					R;
	float					voxSize;
	uint					maxNumCollectionElements;
	CollisionCheckType		collisionCheckType;

	glm::vec4*	toolPos;
	LYVertex*	sortedPos; 
	float4*		force; 
	uint*		gridParticleIndex; 
	uint*		cellStart; 
	uint*		cellEnd; 
	SimParams*	dev_params;

	uint*		collectionCellStart;
	uint*		collectionVertices;

	uint*		totalVertices_2Step;

	uint		numNeighborCells;
	uint		maxSearchRange;	
	uint		maxSearchRangeSq;

	uint*		totalVertices;
};

struct InteractionCellsArgs {
	float3		pos;
	float4		forceVector;
	size_t		numNeighborCells;
	uint		maxSearchRange;
	uint		maxSearchRangeSq;
	uint		totalNeighborhoodSize;

	uint*		gridParticleIndex;
	uint*		cellStart;
	uint*		cellEnd;
	LYVertex*	sortedPos;
	float4*		force;
	SimParams*	dev_params;
	uint*		totalVertices;

	uint*		collectionCellStart;
	uint*		collectionVertices;
};

struct SingleCollisionCheckArgs {
	float4		forceVector;
	size_t		numVertices;
	float3		pos;
	LYVertex*	sortedPos;
	float4*		force;
	uint*		gridParticleIndex;
	uint*		cellStart;
	uint*		cellEnd;
	SimParams*	dev_params;
};

struct ToolCollisionCheckArgs {
	float4			forceVector;
	size_t			numVertices;
	size_t			numToolVertices;
	glm::vec4*		toolPos;
	LYVertex*		sortedPos;
	float4*			force;
	uint*			gridParticleIndex;
	uint*			cellStart;
	uint*			cellEnd;
	SimParams*		dev_params;
};

struct CollisionCheckArgs{
	float4			forceVector;
	float3			pos;
	float			padding;
	uint*			collectionCellStart;
	uint*			collectionVertices;
	LYVertex*		sortedPos;
	float4*			force;
	uint*			gridParticleIndex;
	SimParams*		dev_params;
};

struct OvershootArgs{
	LYVertex		*sortedPos;
	float4			*sinking;				// Device pointer to the float4 sinking property
	float3			pos;
	float			influenceRadius;
	uint			numVertices;
};
#endif
