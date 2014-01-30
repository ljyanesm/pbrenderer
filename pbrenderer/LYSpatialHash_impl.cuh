#include "LYSpatialHash_kernel.cuh"
#include "helper_math.h"
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
	size_t    numVertices)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numVertices) return;

	volatile float3 p = pos[index].m_pos;

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
	size_t    numVertices)
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

__device__ float wendlandWeight(float dist)
{
	float a = 1-dist;
	return ( (a*a*a*a) * ((4*a) + 1) );
}

__device__ inline float KernelQuintic( const float &R, const float &h )
{
	float a_d;

	a_d = 7.0f/(4.0f*CUDART_PI_F*h*h);     //2d Normalization constant
	float tmp (1.0f - R/2.0f);
	if( R < 2.0f )
		return a_d * (tmp)*(tmp)*(tmp)*(tmp) * (2.0f*R+1.0f);
	else
		return 0.0f;
}

__device__
float3 _collideCell(int3    gridPos,
                   uint    index,
                   float3  pos,
                   LYVertex *oldPos,
                   uint   *cellStart,
                   uint   *cellEnd,
				   float3 *OAx,
				   float3 *ONx)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

	float R = 0.4f;
	float w_tot = 0.0f;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
	float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
	float3 Ax = make_float3(0.0f, 0.0f, 0.0f);
	float3 Nx = make_float3(0.0f, 0.0f, 0.0f);
    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                LYVertex pos2 = FETCH(oldPos, j);
				float3 npos;
				npos = pos2.m_pos - pos;
				float dist = length(npos);
				if (dist < R)
				{
					float w = KernelQuintic(R,dist);
					w_tot += w;
					Ax += w * pos2.m_pos;
					float3 Ntmp = w * pos2.m_normal;
					float norm = length(Ntmp);
					Nx += Ntmp / norm;
				}
            }
        }
		if (w_tot != 0.0)
			Ax /= w_tot;
    }
	*OAx += Ax;
	*ONx += Nx;
    return Ax;
}

__global__
void _collisionCheck_cellsD(float3 pos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numCells)

{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index >= numCells) return;
	int start = cellStart[index];
	int end   = cellEnd  [index];
	for (int i = start; i < end; i++)
	{
		LYVertex pos2 = FETCH(oldPos, index);
		float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

		float3 force = make_float3(0.0f, 0.0f, 0.0f);

		float3 Ax = make_float3(0.0f);
		float3 Nx = make_float3(0.0f);

		float3 npos;
		float w = 0.0f;
		npos = pos2.m_pos - pos;
		float dist = length(npos);
		float R = params.R;

		if (dist < R)
		{
			w= KernelQuintic(dist,R);
			Ax += w * pos2.m_pos;
			float3 Ntmp = w * pos2.m_normal;
			Nx += Ntmp;
		}
		else {
			return;
		}

		atomicAdd(&dev_params->Ax.x, Ax.x);
		atomicAdd(&dev_params->Ax.y, Ax.y);
		atomicAdd(&dev_params->Ax.z, Ax.z);
		atomicAdd(&dev_params->Nx.x, Nx.x);
		atomicAdd(&dev_params->Nx.y, Nx.y);
		atomicAdd(&dev_params->Nx.z, Nx.z);
		atomicAdd(&dev_params->w_tot, w);
	}
}
__global__
void _collisionCheckD(float3 pos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numVertices) return;

    // read particle data from sorted arrays
 	//int3 gridPos = calcGridPos(make_float3(pos.x, pos.y, pos.z));
	LYVertex pos2 = FETCH(oldPos, index);
	float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

	float3 force = make_float3(0.0f, 0.0f, 0.0f);

	float3 Ax = make_float3(0.0f);
	float3 Nx = make_float3(0.0f);

	float3 npos;
	float w = 0.0f;
	npos = pos2.m_pos - pos;
	float dist = length(npos);
	float R = params.R;

	if (dist < R)
	{
		w= KernelQuintic(dist,R);
		Ax += w * pos2.m_pos;
		float3 Ntmp = w * pos2.m_normal;
		Nx += Ntmp;
	}
	else {
		return;
	}

	atomicAdd(&dev_params->Ax.x, Ax.x);
	atomicAdd(&dev_params->Ax.y, Ax.y);
	atomicAdd(&dev_params->Ax.z, Ax.z);
	atomicAdd(&dev_params->Nx.x, Nx.x);
	atomicAdd(&dev_params->Nx.y, Nx.y);
	atomicAdd(&dev_params->Nx.z, Nx.z);
	atomicAdd(&dev_params->w_tot, w);
}
