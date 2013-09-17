#include "LYSpatialHash.h"


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
	float4* pos,               // input: positions
	uint    numParticles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numParticles) return;

	volatile float4 p = pos[index];

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
	float4* sortedPos,        // output: sorted positions
	float4* sortedNor,        // output: sorted positions
	float4* sortedVel,        // output: sorted velocities
	float* sortedMass,       // output: sorted masses
	float* sortedPressure,
	float4* sortedForce,
	uint *  gridParticleHash, // input: sorted grid hashes
	uint *  gridParticleIndex,// input: sorted particle indices
	float4* oldPos,           // input: sorted position array
	float4* oldNor,           // input: sorted position array
	float4* oldVel,           // input: sorted velocity array
	float* oldMass,          // input: sorted mass array
	float* oldPressure,
	float4* oldForce,
	uint    numParticles)
{
	extern __shared__ uint sharedHash[];    // blockSize + 1 elements
	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
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

	if (index < numParticles) {
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

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];
		float4 nor = FETCH(oldNor, sortedIndex);       // macro does either global read or texture fetch
		float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
		//		printf("pos[%3d] = (%4.3f, %4.3f, %4.3f)\n", index, pos.x, pos.y, pos.z);
		float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
		float4 force = FETCH(oldForce, sortedIndex);
		float mass = FETCH(oldMass, sortedIndex);
		float pressure = FETCH(oldPressure, sortedIndex);

		sortedPos[index] = pos;
		sortedNor[index] = nor;
		sortedVel[index] = vel;
		sortedMass[index] = mass;
		sortedPressure[index] = pressure;
		sortedForce[index] = force;
		//printf("Particle[%4d].mass = %.4f\n", index, mass);
	}

	__syncthreads ();
}
