#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <vector_functions.h>
#include <device_functions.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>


#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "LYSpatialHash_impl.cuh"

extern "C" {

	void setParameters(SimParams *hostParams)
	{
		// copy parameters to constant memory
		checkCudaErrors( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
	}


	//Round a / b to nearest higher integer value
	uint iDivUp(uint a, uint b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}

	// compute grid and thread block size for a given number of elements
	void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
	{
		numThreads = min(blockSize, n);
		numBlocks = iDivUp(n, numThreads);
	}

	void calcHash(uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *pos,
		int    numVertices)
	{
		uint numThreads, numBlocks;
		computeGridSize(numVertices, 256, numBlocks, numThreads);

		// execute the kernel
		calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
			gridParticleIndex,
			(LYVertex *) pos,
			numVertices);

		// check if kernel invocation generated an error
		getLastCudaError("Kernel execution failed");
	}

	void reorderDataAndFindCellStart(uint  *cellStart,
		uint  *cellEnd,
		LYVertex *sortedPos,
		uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *oldPos,
		uint   numVertices,
		uint   numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numVertices, 256, numBlocks, numThreads);
		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numVertices*sizeof(float4)));
		checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numVertices*sizeof(float4)));
#endif

		uint smemSize = sizeof(uint)*(numThreads+1);
		reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
			cellStart,
			cellEnd,
			(LYVertex *) sortedPos,
			gridParticleHash,
			gridParticleIndex,
			(LYVertex *) oldPos,
			numVertices);
		getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

#if USE_TEX
		checkCudaErrors(cudaUnbindTexture(oldPosTex));
		checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
	}

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numVertices)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
			thrust::device_ptr<uint>(dGridParticleHash + numVertices),
			thrust::device_ptr<uint>(dGridParticleIndex));
	}

	void collisionCheck(float3 pos, LYVertex *sortedPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, float3 *forceFeedback, uint numVertices)
	{
#if USE_TEX
        checkCudaErrors(cudaBindTexture(0, oldPosTex, sortedPos, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, oldVelTex, sortedVel, numParticles*sizeof(float4)));
        checkCudaErrors(cudaBindTexture(0, cellStartTex, cellStart, numCells*sizeof(uint)));
        checkCudaErrors(cudaBindTexture(0, cellEndTex, cellEnd, numCells*sizeof(uint)));
#endif

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numVertices, 64, numBlocks, numThreads);
		
		// execute the kernel
        collisionCheckD<<< numBlocks, numThreads >>>(pos,
											(LYVertex *)sortedPos,
                                              gridParticleIndex,
                                              cellStart,
                                              cellEnd,
											  forceFeedback,
                                              numVertices);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

#if USE_TEX
        checkCudaErrors(cudaUnbindTexture(oldPosTex));
        checkCudaErrors(cudaUnbindTexture(oldVelTex));
        checkCudaErrors(cudaUnbindTexture(cellStartTex));
        checkCudaErrors(cudaUnbindTexture(cellEndTex));
#endif
	}
	
}