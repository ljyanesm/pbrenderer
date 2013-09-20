#include "defines.h"
#include "LYSpatialHash_kernel.cuh"

#if USE_TEX
// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> oldPosTex;
texture<float4, 1, cudaReadModeElementType> oldVelTex;

texture<uint, 1, cudaReadModeElementType> gridParticleHashTex;
texture<uint, 1, cudaReadModeElementType> cellStartTex;
texture<uint, 1, cudaReadModeElementType> cellEndTex;
#endif
__constant__ SimParams params;

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x) / params.cellSize.x);
	gridPos.y = floor((p.y) / params.cellSize.y);
	gridPos.z = floor((p.z) / params.cellSize.z);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
	gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (params.gridSize.y-1);
	gridPos.z = gridPos.z & (params.gridSize.z-1);
	return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate grid hash value for each particle
__global__
	void calcHashD(uint*   gridParticleHash,  // output
	uint*   gridParticleIndex, // output
	LYVertex* pos,               // input: positions
	uint    numVertices)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numVertices) return;

	volatile glm::vec3 p = pos[index].m_pos;

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
	void reorderDataAndFindCellStartD(uint*   cellStart,        // output: cell start index
	uint*   cellEnd,          // output: cell end index
	LYVertex* sortedPos,        // output: sorted positions
	uint *  gridParticleHash, // input: sorted grid hashes
	uint *  gridParticleIndex,// input: sorted particle indices
	LYVertex* oldPos,           // input: sorted position array
	uint    numVertices)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numVertices) {
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x+1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index-1];
		}
	}

	__syncthreads();

	if (index < numVertices) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numVertices - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		LYVertex pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch

		sortedPos[index] = pos;
	}

	__syncthreads ();
}
