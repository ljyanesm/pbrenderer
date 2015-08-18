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

//Round a / b to nearest higher integer value
__host__ __device__ uint iDivUp(size_t a, uint b)
{
	return ((uint) a % b != 0) ? ((uint) a / b + 1) : ((uint) a / b);
}

// compute grid and thread block size for a given number of elements
__host__ __device__ void computeGridSize(size_t n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, static_cast<uint>(n));
	numBlocks = iDivUp(n, numThreads);
}

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
		LYVertex pos = FETCH(oldPos,sortedIndex);       // macro does either global read or texture fetch
		sortedPos[index] = pos;
	}

	__syncthreads ();
}

__device__ float wendlandWeight(float dist)
{
	float a = 1-dist;
	return ( (a*a*a*a) * ((4*a) + 1) );
}

__device__ int translateRange(int value, int lMin, int lMax, int rMin, int rMax)
{
	int lSpan = lMax-lMin;
	int rSpan = rMax-rMin;

	float valueScaled = float(value - lMin) / float(lSpan);

	return (int) rMin + (valueScaled * rSpan); 
}

__global__
	void _collectInteractionCells(InteractionCellsArgs arguments)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index > arguments.numNeighborCells) return;

	// Get the position of the voxel we are using to calculate force.
	int3 gridPos = calcGridPos(arguments.pos);
	float3 nSize = ( arguments.dev_params->R / arguments.dev_params->cellSize );
	nSize.x = ceil(nSize.x);
	nSize.y = ceil(nSize.y);
	nSize.z = ceil(nSize.z);

	uint maxSearchRange = arguments.maxSearchRange;
	uint maxSearchRangeSq = maxSearchRange*maxSearchRange;
	uint iX, iY, iZ;
	iX = (int)(index % maxSearchRange);
	iY = (int)((index/maxSearchRange) % maxSearchRange);
	iZ = (int)(index / (maxSearchRangeSq));

	int x = translateRange(iX, 0, maxSearchRange, -maxSearchRange/2, maxSearchRange/2);
	int y = translateRange(iY, 0, maxSearchRange, -maxSearchRange/2, maxSearchRange/2);
	int z = translateRange(iZ, 0, maxSearchRange, -maxSearchRange/2, maxSearchRange/2);
	int3 neighbourPos = gridPos + make_int3(x , y, z);
	uint  neighbourGridPos = calcGridHash(neighbourPos);
	uint cellStartI = FETCH(arguments.cellStart, neighbourGridPos);
	if (cellStartI == 0xffffffff) return;
	uint N = FETCH(arguments.cellEnd, neighbourGridPos) - cellStartI;
	arguments.collectionCellStart[index] = cellStartI;
	arguments.collectionVertices[index] = N;

	__syncthreads();
}

__global__
	void _computeCollisionCheck(CollisionCheckArgs arguments)
{
	uint blkIdx = blockIdx.x;
	uint trdIdx = threadIdx.x;

	__shared__ uint blkCellStart;
	__shared__ uint blkNumVerts;

	if (trdIdx == 0) {
		blkCellStart = arguments.collectionCellStart[blkIdx];
		blkNumVerts  = arguments.collectionVertices[blkIdx];
	}
	__syncthreads();

	if (trdIdx < blkNumVerts){
		uint index = blkCellStart+trdIdx;
		LYVertex pos2 = FETCH(arguments.sortedPos, index);
		float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

		float3 Ax = make_float3(0.0f);
		float3 Nx = make_float3(0.0f);

		float3 npos;
		float w = 0.0f;
		npos = pos2.m_pos - arguments.pos;
		float dist = length(npos);
		float R = params.R;

		if (dist > R) return;
		else{
			w = wendlandWeight(dist/R);
			Ax += w * pos2.m_pos;
			Nx += w * pos2.m_normal;
			uint sortedIndex = arguments.gridParticleIndex[index];
			if (length(arguments.forceVector) > 0.03f) arguments.force[sortedIndex] += arguments.forceVector*0.001f;
			atomicAdd(&arguments.dev_params->Ax.x, Ax.x);
			atomicAdd(&arguments.dev_params->Ax.y, Ax.y);
			atomicAdd(&arguments.dev_params->Ax.z, Ax.z);
			atomicAdd(&arguments.dev_params->Nx.x, Nx.x);
			atomicAdd(&arguments.dev_params->Nx.y, Nx.y);
			atomicAdd(&arguments.dev_params->Nx.z, Nx.z);
			atomicAdd(&arguments.dev_params->w_tot, w);
		}
	}

}

__global__
void naiveChildCollisionKernel(float3 pos, LYVertex *oldPos, float4 *force, float4 forceVector, uint *gridParticleIndex, SimParams *dev_params, int firstVertex, size_t numVertices){

	uint index =  __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index > numVertices) return;

	index+=firstVertex;

	LYVertex pos2 = FETCH(oldPos, index);
	float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

	float3 Ax = make_float3(0.0f);
	float3 Nx = make_float3(0.0f);

	float3 npos;
	float w = 0.0f;
	npos = pos2.m_pos - pos;
	float dist = length(npos);
	float R = params.R;

	if (dist > R) return;
	else{
		w = wendlandWeight(dist/R);
		Ax += w * pos2.m_pos;
		Nx += w * pos2.m_normal;
		uint sortedIndex = gridParticleIndex[index];
		if (length(forceVector) > 0.03f) force[sortedIndex] += forceVector*0.001f;
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
	void _naiveDynamicCollisionCheckD(SingleCollisionCheckArgs args)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	// Get the position of the voxel we are using to calculate force.
	int3 gridPos = calcGridPos(args.pos);
	float3 nSize = ( args.dev_params->R / args.dev_params->cellSize );
	nSize.x = ceil(nSize.x);
	nSize.y = ceil(nSize.y);
	nSize.z = ceil(nSize.z);
	//printf("gridPos = %d %d %d\n", gridPos.x, gridPos.y, gridPos.z);
	//printf("nSize = %f %f %f\n", nSize.x, nSize.y, nSize.z);	
	// For all the voxels around currentVoxel go to the neighbors and launch threads on each vertex on them.
	//return;
	uint totalVertices = 0;
	for(int z=-nSize.z; z<=nSize.z; z++) {
		for(int y=-nSize.y; y<=nSize.y; y++) {
			for(int x=-nSize.x; x<=nSize.x; x++) {
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				uint  neighbourGridPos = calcGridHash(neighbourPos);
				uint cellStartI = FETCH(args.cellStart, neighbourGridPos);
				if (cellStartI == 0xffffffff) continue;
				uint N = FETCH(args.cellEnd, neighbourGridPos) - cellStartI;
//				printf("GridPos = %d,  Number particles = %d\n", neighbourGridPos, N);
				uint numThreads, numBlocks;
				computeGridSize(N, 16, numBlocks, numThreads);
				// Launch N child threads to add information from neighbor cells
#if (__CUDA_ARCH__ >= 350)
				naiveChildCollisionKernel<<<1, N>>>(args.pos, args.sortedPos, args.force, args.forceVector, args.gridParticleIndex, args.dev_params, cellStartI, N);
#endif
				totalVertices += N;
			}
		}
	}


	//printf("Total amount of vertices = %d\n", totalVertices);

	__syncthreads();
}

__global__
	void _naiveDynamicToolCollisionCheckD(ToolCollisionCheckArgs args)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index > args.numToolVertices) return;
	float3 pos = make_float3(args.toolPos[index].x, args.toolPos[index].y, args.toolPos[index].z);

	// Get the position of the voxel we are using to calculate force.
	int3 gridPos = calcGridPos(pos);
	float3 nSize = ( args.dev_params->R / args.dev_params->cellSize );
	nSize.x = ceil(nSize.x);
	nSize.y = ceil(nSize.y);
	nSize.z = ceil(nSize.z);
	//printf("gridPos = %d %d %d\n", gridPos.x, gridPos.y, gridPos.z);
	//printf("nSize = %f %f %f\n", nSize.x, nSize.y, nSize.z);	
	// For all the voxels around currentVoxel go to the neighbors and launch threads on each vertex on them.
	//return;
	uint totalVertices = 0;
	for(int z=-nSize.z; z<=nSize.z; z++) {
		for(int y=-nSize.y; y<=nSize.y; y++) {
			for(int x=-nSize.x; x<=nSize.x; x++) {
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				uint  neighbourGridPos = calcGridHash(neighbourPos);
				uint cellStartI = FETCH(args.cellStart, neighbourGridPos);
				if (cellStartI == 0xffffffff) continue;
				uint N = FETCH(args.cellEnd, neighbourGridPos) - cellStartI;
//				printf("GridPos = %d,  Number particles = %d\n", neighbourGridPos, N);
				uint numThreads, numBlocks;
				computeGridSize(N, 16, numBlocks, numThreads);
				// Launch N child threads to add information from neighbor cells
#if (__CUDA_ARCH__ > 350)
				naiveChildCollisionKernel<<<1, N>>>(pos, args.sortedPos, args.force, args.forceVector, args.gridParticleIndex, args.dev_params, cellStartI, N);
#endif
				totalVertices += N;
			}
		}
	}


	//printf("Total amount of vertices = %d\n", totalVertices);

	__syncthreads();
}

// Number of 'CellStarts' and 'N vertices' is based on the formula for Moore's neighborhood for up to r = 15
// Neighbors = (2r + 1)^3
// 

__device__ uint vertexIndex[50000];
//__device__ uint vertexToolIndex[4913][50000];
__device__ uint vertexToolIndex[1][1];

__global__
	void childCollisionKernel(float3 pos, LYVertex *oldPos, float4 *force, float4 forceVector, uint *gridParticleIndex, SimParams *dev_params, uint totalCells, uint totalVertices){

		uint index =  __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index > totalVertices) return;
		//uint cellToCheckIndex = 0;
		//// Improve with a binary search (the numVertsCheck array is incrementally ordered!)
		//uint numVerts = numVertsCell[cellToCheckIndex];
		//while (numVerts < index){ 
		//	numVerts = numVertsCell[++cellToCheckIndex];
		//}

		//uint firstVertex = cellsToCheck[cellToCheckIndex];
		//uint vIndex = firstVertex + (index - numVerts);

		//printf("tId = %d ; cellStart = %d ; vIndex = %d\n", index, firstVertex, vIndex);

		uint vIndex = vertexIndex[index];

		LYVertex pos2 = FETCH(oldPos, vIndex);
		float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

		float3 Ax = make_float3(0.0f);
		float3 Nx = make_float3(0.0f);

		float3 npos;
		float w = 0.0f;
		npos = pos2.m_pos - pos;
		float dist = length(npos);
		float R = params.R;

		if (dist > R) return;
		else{
			w = wendlandWeight(dist/R);
			Ax += w * pos2.m_pos;
			Nx += w * pos2.m_normal;
			uint sortedIndex = gridParticleIndex[vIndex];
			if (length(forceVector) > 0.03f) force[sortedIndex] += forceVector*0.001f;
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
	void childToolCollisionKernel(float3 pos, LYVertex *oldPos, uint pIndex, float4 *force, float4 forceVector, uint *gridParticleIndex, SimParams *dev_params, uint totalCells, uint totalVertices){

		uint index =  __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
		if (index > totalVertices) return;
		//uint cellToCheckIndex = 0;
		//// Improve with a binary search (the numVertsCheck array is incrementally ordered!)
		//uint numVerts = numVertsCell[cellToCheckIndex];
		//while (numVerts < index){ 
		//	numVerts = numVertsCell[++cellToCheckIndex];
		//}

		//uint firstVertex = cellsToCheck[cellToCheckIndex];
		//uint vIndex = firstVertex + (index - numVerts);

		//printf("tId = %d ; cellStart = %d ; vIndex = %d\n", index, firstVertex, vIndex);

		uint vIndex = vertexToolIndex[pIndex][index];

		LYVertex pos2 = FETCH(oldPos, vIndex);
		float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

		float3 Ax = make_float3(0.0f);
		float3 Nx = make_float3(0.0f);

		float3 npos;
		float w = 0.0f;
		npos = pos2.m_pos - pos;
		float dist = length(npos);
		float R = params.R;

		if (dist > R) return;
		else{
			w = wendlandWeight(dist/R);
			Ax += w * pos2.m_pos;
			Nx += w * pos2.m_normal;
			uint sortedIndex = gridParticleIndex[vIndex];
			if (length(forceVector) > 0.03f) force[sortedIndex] += forceVector*0.001f;
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
	void _dynamicCollisionCheckD(SingleCollisionCheckArgs args)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	// Get the position of the voxel we are using to calculate force.
	int3 gridPos = calcGridPos(args.pos);
	float3 nSize = ( args.dev_params->R / args.dev_params->cellSize );
	nSize.x = ceil(nSize.x);
	nSize.y = ceil(nSize.y);
	nSize.z = ceil(nSize.z);
	//printf("gridPos = %d %d %d\n", gridPos.x, gridPos.y, gridPos.z);
	//printf("nSize = %f\n", nSize.x);	
	// For all the voxels around currentVoxel go to the neighbors and launch threads on each vertex on them.

	uint totalVertices = 0;
	uint totalCells = 0;
	//return;
	for(int z=-nSize.z; z<=nSize.z; z++) {
		for(int y=-nSize.y; y<=nSize.y; y++) {
			for(int x=-nSize.x; x<=nSize.x; x++) {
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				uint  neighbourGridPos = calcGridHash(neighbourPos);
				uint cellStartI = FETCH(args.cellStart, neighbourGridPos);
				if (cellStartI == 0xffffffff) continue;
				uint N = FETCH(args.cellEnd, neighbourGridPos) - cellStartI;
				//printf("GridPos = %d,  Number particles = %d\n", neighbourGridPos, N);
				// Accumulate information from neighbor cells on 'starts' and 'num_elems' arrays

				for ( int j = 0; j < N; j++){
					vertexIndex[totalVertices + j] = cellStartI+j;
				}

				//cellsToCheck[totalCells] = cellStartI;
				//numVertsCheck[totalCells] = N;
				totalVertices += N;
				totalCells++;

			}
		}
	}

	// Prefix-sum the numVertsCheck to get the actual 
	//numVertsCell[0] = numVertsCheck[0];
	//for (int i = 1; i < totalCells; i++)
	//{
	//	numVertsCell[i] = numVertsCell[i-1] + numVertsCheck[i];
	//	//printf("numVertsCell[%d] = %d\n", i, numVertsCell[i]);
	//}

	//for (int i = 0; i < totalCells; i++)
	//	for ( int j = 0; j < numVertsCheck[i]; j++){
	//		vertexIndex[i*totalCells + j] = cellsToCheck[i]+j;
	//	}

	//printf("Total amount of cells = %d\n", totalCells);
	//printf("Total amount of vertices = %d\n", totalVertices);
	if (totalVertices <= 0) return;
	uint numThreads, numBlocks;
	computeGridSize(totalVertices, 256, numBlocks, numThreads);
	// TODO: Improve by making a block per cellToCheck!!
#if (__CUDA_ARCH__ >= 350)
	childCollisionKernel<<<numBlocks, numThreads>>>(args.pos, args.sortedPos, args.force, args.forceVector, args.gridParticleIndex, args.dev_params, totalCells, totalVertices);
#endif
	__syncthreads();
}


__global__
	void _dynamicToolCollisionCheckD(ToolCollisionCheckArgs args)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index > args.numToolVertices) return;
	float3 pos = make_float3(args.toolPos[index].x, args.toolPos[index].y, args.toolPos[index].z);

	// Get the position of the voxel we are using to calculate force.
	int3 gridPos = calcGridPos(pos);
	float3 nSize = ( args.dev_params->R / args.dev_params->cellSize );
	nSize.x = ceil(nSize.x);
	nSize.y = ceil(nSize.y);
	nSize.z = ceil(nSize.z);
	//printf("nSize = %f\n", nSize.x);	
	// For all the voxels around currentVoxel go to the neighbors and launch threads on each vertex on them.

	uint totalVertices = 0;
	uint totalCells = 0;
	//return;
	for(int z=-nSize.z; z<=nSize.z; z++) {
		for(int y=-nSize.y; y<=nSize.y; y++) {
			for(int x=-nSize.x; x<=nSize.x; x++) {
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				uint  neighbourGridPos = calcGridHash(neighbourPos);
				uint cellStartI = FETCH(args.cellStart, neighbourGridPos);
				if (cellStartI == 0xffffffff) continue;
				uint N = FETCH(args.cellEnd, neighbourGridPos) - cellStartI;
				//printf("GridPos = %d,  Number particles = %d\n", neighbourGridPos, N);
				// Accumulate information from neighbor cells on 'starts' and 'num_elems' arrays

				for ( int j = 0; j < N; j++){
					vertexToolIndex[index][totalVertices + j] = cellStartI+j;
				}

				//cellsToCheck[totalCells] = cellStartI;
				//numVertsCheck[totalCells] = N;
				totalVertices += N;
				totalCells++;

			}
		}
	}

	// Prefix-sum the numVertsCheck to get the actual 
	//numVertsCell[0] = numVertsCheck[0];
	//for (int i = 1; i < totalCells; i++)
	//{
	//	numVertsCell[i] = numVertsCell[i-1] + numVertsCheck[i];
	//	//printf("numVertsCell[%d] = %d\n", i, numVertsCell[i]);
	//}

	//for (int i = 0; i < totalCells; i++)
	//	for ( int j = 0; j < numVertsCheck[i]; j++){
	//		vertexIndex[i*totalCells + j] = cellsToCheck[i]+j;
	//	}

	//printf("Total amount of cells = %d\n", totalCells);
	if (totalVertices <= 0) return;
	uint numThreads, numBlocks;
	computeGridSize(totalVertices, 256, numBlocks, numThreads);
	// TODO: Improve by making a block per cellToCheck!!
#if (__CUDA_ARCH__ > 350)
	childToolCollisionKernel<<<numBlocks, numThreads>>>(pos, args.sortedPos, index, args.force, args.forceVector, args.gridParticleIndex, args.dev_params, totalCells, totalVertices);
#endif
	__syncthreads();
}

__global__
void _collisionCheckD(float3 pos, LYVertex *oldPos, float4 *force, float4 forceVector, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numVertices) return;

    // read particle data from sorted arrays
	LYVertex pos2 = FETCH(oldPos, index);
	float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

	float3 Ax = make_float3(0.0f);
	float3 Nx = make_float3(0.0f);

	float3 npos;
	float w = 0.0f;
	npos = pos2.m_pos - pos;
	float dist = length(npos);
	float R = params.R;

	if (dist < R)
	{
		w = wendlandWeight(dist/R);
		Ax += w * pos2.m_pos;
		Nx += w * pos2.m_normal;
		uint sortedIndex = gridParticleIndex[index];
		if (length(forceVector) > 0.03f) force[sortedIndex] += forceVector*0.001f;
		atomicAdd(&dev_params->Ax.x, Ax.x);
		atomicAdd(&dev_params->Ax.y, Ax.y);
		atomicAdd(&dev_params->Ax.z, Ax.z);
		atomicAdd(&dev_params->Nx.x, Nx.x);
		atomicAdd(&dev_params->Nx.y, Nx.y);
		atomicAdd(&dev_params->Nx.z, Nx.z);
		atomicAdd(&dev_params->w_tot, w);
	}
	else {
		return;
	}
	__syncthreads();
}

__global__
void _updatePositions(LYVertex *sortedPos, float4 *forces, LYVertex *oldPos, size_t numVertices)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numVertices) return;

    // read particle data from sorted arrays
	float3	F = make_float3(FETCH(forces, index));
	if (length(F) < 0.0001) return;

	float3 new_pos = F;
	oldPos[index].m_pos -= new_pos;
	
	forces[index] = make_float4(0.0f);
}

__global__
	void _computeOvershoot(OvershootArgs args)
{
	uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= args.numVertices) return;

	// read particle data from sorted arrays
	LYVertex pos2 = FETCH(args.sortedPos, index);

	float3 npos;
	float3 di = make_float3(0.0f);
	float R = 2*args.influenceRadius;
	npos = pos2.m_pos - args.pos;
	float dist = length(npos);

	if (dist > R) return;
	
	di = (R - dist) * ( (npos)/dist );

	atomicAdd(&args.sinking->x, di.x);
	atomicAdd(&args.sinking->y, di.y);
	atomicAdd(&args.sinking->z, di.z);
}