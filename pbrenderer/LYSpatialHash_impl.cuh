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
	uint    numVertices)
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

__device__ float wendlandWeight(float dist)
{
	float a = 1-dist;
	return ( (a*a*a*a) * ((4*a) + 1) );
}

__device__
float3 collideCell(int3		gridPos,
                   uint		index,
				   float	R,
                   float3	Xp,
                   LYVertex *oldPos,
                   uint		*cellStart,
                   uint		*cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);

	float3 Oi = make_float3(0.0f, 0.0f, 0.0f);
	uint N = 0;
	if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 Pi = FETCH(oldPos, j).m_pos;
				float dist = length(Xp - Pi);
				Oi += (R - dist) * ((Xp - Pi) / dist);
				N++;
            }
        }
    }
    return (Oi/N);
}

__device__
float3 calcMean(int3 gridPos, uint index, float3 Xp, LYVertex *oldPos, uint *cellStart, uint *cellEnd, float3 *newMin)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);
	
	uint numVert = 0;

	float3 mean = make_float3(0.0f);
	float3 Xmin = make_float3(INF);
	float minimum = INF;

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);
        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
				float3 Xi = FETCH(oldPos, j).m_pos;
				numVert++;
				mean += Xi;
				float testVal = length(Xp - Xi);
				if (  testVal < minimum )
				{
					minimum = testVal;
					*newMin = Xi;
				}
            }
        }
    }
    return (mean/numVert);
}

__device__
float calcSD(int3 gridPos, uint index, float3 mean, LYVertex *oldPos, uint *cellStart, uint *cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);
	
	uint numVert = 0;

	float result = 0.0f;
    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
				float3 Xi = FETCH(oldPos, j).m_pos;
				numVert++;
				result += length(Xi - mean);
            }
        }
    }
    return (result/numVert);
}

__global__
void collisionCheckD(float3 pos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, uint numVertices)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numVertices) return;

    // read particle data from sorted arrays
 	int3 gridPos = calcGridPos(pos);
	float3 total_force = make_float3(0.0f);
	float3 force = make_float3(0.0f);
	float h = 0.0f;
	float Rp = dev_params->RMAX;
	float3 Xh = pos;
	float sd = 0.0f;

	int maskSize = 1;
	
	float3 meanHolder = make_float3(0.0f);
	float3 Xc = make_float3(INF);
	float3 Pi = FETCH(oldPos, index).m_pos;
	for (int z=-maskSize; z<=maskSize; z++)
    {
        for (int y=-maskSize; y<=maskSize; y++)
        {
            for (int x=-maskSize; x<=maskSize; x++)
            {
                int3 neighbourPos;
				float3 newMin = Xc;
				neighbourPos.x = gridPos.x + x;
				neighbourPos.y = gridPos.y + y;
				neighbourPos.z = gridPos.z + z;
				meanHolder += calcMean(neighbourPos, index, Pi, oldPos, cellStart, cellEnd, &dev_params->Xc);
				if ( length(Xc) > length(newMin) ) Xc = newMin;
            }
        }
    }

	dev_params->Xc = Xc;

	for (int z=-maskSize; z<=maskSize; z++)
    {
        for (int y=-maskSize; y<=maskSize; y++)
        {
            for (int x=-maskSize; x<=maskSize; x++)
            {
                int3 neighbourPos;
				neighbourPos.x = gridPos.x + x;
				neighbourPos.y = gridPos.y + y;
				neighbourPos.z = gridPos.z + z;
				sd += calcSD(neighbourPos, index, meanHolder, oldPos, cellStart, cellEnd);
            }
        }
    }
	uint3 nSize = make_uint3(0);
	Rp = dev_params->alpha * 1.06 * pow(sd, -1.0f/5.0f);
	nSize.x = ceil(Rp / dev_params->cellSize.x);
	nSize.y = ceil(Rp / dev_params->cellSize.y);
	nSize.z = ceil(Rp / dev_params->cellSize.z);

	if ( nSize.x < dev_params->RMIN) nSize.x = dev_params->RMIN;
	if ( nSize.y < dev_params->RMIN) nSize.y = dev_params->RMIN;
	if ( nSize.z < dev_params->RMIN) nSize.z = dev_params->RMIN;

	float3 Vn = make_float3(0.0f);
	for (int z=-maskSize; z<=maskSize; z++)
    {
        for (int y=-maskSize; y<=maskSize; y++)
        {
            for (int x=-maskSize; x<=maskSize; x++)
            {
                int3 neighbourPos;
				neighbourPos.x = gridPos.x + x;
				neighbourPos.y = gridPos.y + y;
				neighbourPos.z = gridPos.z + z;
				Vn += collideCell(neighbourPos, index, h, pos, oldPos, cellStart, cellEnd);
            }
        }
    }
	float nlength = length(Vn);
	float3 N = Vn / nlength;

	float theta = dot(Xc - dev_params->colliderPos, N);

	float error = INF;
	while ( error > dev_params->epsilon ) {
		float3 tmp = dev_params->colliderPos - Xc;
		dev_params->colliderPos.x += -dev_params->gamma * (dev_params->colliderPos.x - Xh.x) - (dev_params->beta*(length(tmp) * cos(theta)) - dev_params->dmin) * ( sin(theta) - sqrt((tmp.x*tmp.x)));
		dev_params->colliderPos.y += -dev_params->gamma * (dev_params->colliderPos.y - Xh.y) - (dev_params->beta*(length(tmp) * cos(theta)) - dev_params->dmin) * ( sin(theta) - sqrt((tmp.y*tmp.y)));
		dev_params->colliderPos.z += -dev_params->gamma * (dev_params->colliderPos.z - Xh.z) - (dev_params->beta*(length(tmp) * cos(theta)) - params.dmin) * ( sin(theta) - sqrt((tmp.z*tmp.z)));
	}

	if ( dot(Vn, N) > 0)
	{
		float3 Vh = Xh - params.colliderPos;
		total_force = -(nlength - params.dmin) * N * dev_params->forceSpring;
	}

	dev_params->force = total_force;
	dev_params->colliderRadius = Rp;
	dev_params->force = make_float3(0.0f);
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

	float R = 0.50f;
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
                float3 pos2 = FETCH(oldPos, j).m_pos;
				float3 nor2 = FETCH(oldPos, j).m_normal;
				float3 npos;
				npos = pos2 - pos;
				float dist = length(npos);
				if (dist < R)
				{
					float w = wendlandWeight(dist/R);
					w_tot += w;
					Ax += w * pos2;

					float3 Ntmp = w * nor2;
					float norm = length(Ntmp);
					Nx += Ntmp / norm;
				}
            }
        }
		Ax /= w_tot;
    }
	*OAx += Ax;
	*ONx += Nx;
    return Ax;
}

__global__
void _collisionCheckD(float3 pos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, uint numVertices)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numVertices) return;

    // read particle data from sorted arrays
 	int3 gridPos = calcGridPos(make_float3(pos.x, pos.y, pos.z));

	float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

	float3 force = make_float3(0.0f, 0.0f, 0.0f);

	float3 Ax = make_float3(0.0f);
	float3 Nx = make_float3(0.0f);
	//TODO:	Calculate the real size of the neighborhood using the different 
	//		functions for neighbor calculation [square, sphere, point]
	int maskSize = 2;
    for (int z=-maskSize; z<=maskSize; z++)
    {
        for (int y=-maskSize; y<=maskSize; y++)
        {
            for (int x=-maskSize; x<=maskSize; x++)
            {
                int3 neighbourPos;
				neighbourPos = gridPos + make_int3(x,y,z);
				force = _collideCell(neighbourPos, index, pos, oldPos, cellStart, cellEnd, &Ax, &Nx);
                total_force += force;
            }
        }
    }

	dev_params->Ax = Ax;
	dev_params->Nx = Nx;
	dev_params->force = total_force;
}
