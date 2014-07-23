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

	void collisionCheck  (ccConfiguration &arguments);
	void _collisionCheckD(float3 pos, LYVertex *oldPos, float4 *force, float4 forceVector, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);

	void updatePositions(LYVertex *sortedPos, float4 *forces, LYVertex *oldPos, size_t numVertices);
	void _updatePositions(LYVertex *sortedPos, float4 *forces, LYVertex *oldPos, size_t numVertices);

	void updateDensities(LYVertex *sortedPos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);
	void _updateDensities(LYVertex *sortedPos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);

	void updateProperties(LYVertex *sortedPos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);
	void _updateProperties(LYVertex *sortedPos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);

	void _naiveDynamicCollisionCheckD(float3 pos, LYVertex *oldPos, float4 *force, float4 forceVector, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);
	void _dynamicCollisionCheckD(float3 pos, LYVertex *oldPos, float4 *force, float4 forceVector, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices);
}