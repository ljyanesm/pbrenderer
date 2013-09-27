#include "defines.h"
#include "LYSpatialHash_kernel.cuh"
extern "C"
{
	void setParameters(SimParams *hostParams);

	void calcHash(uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *pos,
		int    numVertices);

	void reorderDataAndFindCellStart(uint  *cellStart,
		uint  *cellEnd,
		LYVertex *sortedPos,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *oldPos,
		uint   numVertices,
		uint   numCells);

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numVertices);

	void collisionCheck(float3 pos, LYVertex *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, float3 *forceFeedback, uint numVertices);
}