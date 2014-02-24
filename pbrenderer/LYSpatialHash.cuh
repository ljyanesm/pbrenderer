#include "defines.h"
#include "LYSpatialHash_kernel.cuh"
extern "C"
{
	void setParameters(SimParams *hostParams);

	void calcHash(uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *pos,
		size_t    numVertices);

	void reorderDataAndFindCellStart(uint  *cellStart,
		uint  *cellEnd,
		LYVertex *sortedPos,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *oldPos,
		size_t   numVertices,
		uint   numCells);

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, size_t numVertices);

	void collisionCheck(float3 pos, LYVertex *sortedPos, float4 *force, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);
	void _collisionCheckD(float3 pos, LYVertex *oldPos, float4 *force, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);
	void _collisionCheck_cellsD(float3 pos, LYVertex *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numCells);

	void updatePositions(LYVertex *sortedPos, float4 *forces, LYVertex *oldPos, size_t numVertices);
	void _updatePositions(LYVertex *sortedPos, float4 *forces, LYVertex *oldPos, size_t numVertices);
}