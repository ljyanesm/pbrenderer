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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" {

	void setParameters(SimParams *hostParams)
	{
		// copy parameters to constant memory
		checkCudaErrors( cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)) );
	}

	void calcHash(uint  *gridParticleHash,
		uint  *gridParticleIndex,
		LYVertex *pos,
		size_t    numVertices)
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
		size_t   numVertices,
		uint   numCells)
	{
		uint numThreads, numBlocks;
		computeGridSize(numVertices, 256, numBlocks, numThreads);
		// set all cells to empty
		checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

#if USE_TEX
		checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numVertices*sizeof(float4)));
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

	void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, size_t numVertices)
	{
		thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
			thrust::device_ptr<uint>(dGridParticleHash + numVertices),
			thrust::device_ptr<uint>(dGridParticleIndex));
	}

	void collisionCheck(ccConfiguration &arguments)
	{
		// Get the size of the collision radius
		float3 QP = arguments.pos;
		float r = arguments.R;

		// Generate the position of the 'tool' points to calculate the collisions
		std::vector<glm::vec4> toolVertices(arguments.numToolVertices);
		for(int i = 0; i < arguments.numToolVertices; i++)
		{
			float t = i / arguments.numToolVertices;
			toolVertices.at(i) = glm::vec4((float)( (-arguments.voxSize*5) * (1-t) + (arguments.voxSize*5)*(t) ) + QP.x, QP.y, QP.z, 0);
		}

		cudaMemcpy(arguments.toolPos, toolVertices.data(), arguments.numToolVertices*sizeof(glm::vec4), cudaMemcpyHostToDevice);

		// Calculate the size of the neighborhood based on the radius
		int nSize = round(arguments.voxSize*r);
		// Calculate the voxel position of the query point
		glm::vec4 voxelPos = glm::vec4(QP.x, QP.y, QP.z, 0) / nSize;

		getLastCudaError("Before collisionCheck Kernel execution failed");

		uint numThreads, numBlocks;
		computeGridSize(arguments.numToolVertices, 256, numBlocks, numThreads);

		// Using dynamic parallelism only execute threads on the neighborhood of the selected QP

		if (arguments.naiveDynamicCollisionCheck)
		{
			_naiveDynamicToolCollisionCheckD<<< numBlocks, numThreads>>>(	arguments.toolPos,
				(LYVertex *) arguments.sortedPos,
				(float4 *) arguments.force,
				arguments.forceVector,
				arguments.gridParticleIndex,
				arguments.cellStart,
				arguments.cellEnd,
				arguments.dev_params,
				arguments.numVertices,
				arguments.numToolVertices);
		}
		else {
			_dynamicToolCollisionCheckD<<< numBlocks, numThreads >>>(	arguments.toolPos,
				(LYVertex *) arguments.sortedPos,
				(float4 *) arguments.force,
				arguments.forceVector,
				arguments.gridParticleIndex,
				arguments.cellStart,
				arguments.cellEnd,
				arguments.dev_params,
				arguments.numVertices,
				arguments.numToolVertices);
		}
#if 0
		if (arguments.naiveDynamicCollisionCheck)
		{
			_naiveDynamicCollisionCheckD<<< 1, 1 >>>(	arguments.pos,
				(LYVertex *) arguments.sortedPos,
				(float4 *) arguments.force,
				arguments.forceVector,
				arguments.gridParticleIndex,
				arguments.cellStart,
				arguments.cellEnd,
				arguments.dev_params,
				arguments.numVertices);
		}
		else 
		{
			_dynamicCollisionCheckD<<< 1, 1 >>>(	arguments.pos,
				(LYVertex *) arguments.sortedPos,
				(float4 *) arguments.force,
				arguments.forceVector,
				arguments.gridParticleIndex,
				arguments.cellStart,
				arguments.cellEnd,
				arguments.dev_params,
				arguments.numVertices);
		}
#endif

		{
			// thread per particle
			uint numThreads, numBlocks;
			computeGridSize(arguments.numVertices, 256, numBlocks, numThreads);

			// execute the kernel
			//_collisionCheckD<<< numBlocks, numThreads >>>(	arguments.pos,
			//	(LYVertex *) arguments.sortedPos,
			//	(float4 *) arguments.force,
			//	arguments.forceVector,
			//	arguments.gridParticleIndex,
			//	arguments.cellStart,
			//	arguments.cellEnd,
			//	arguments.dev_params,
			//	arguments.numVertices);
			 //check if kernel invocation generated an error
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
		}	
}

	void updatePositions(LYVertex *sortedPos, float4 *force, LYVertex *oldPos, size_t numVertices)
	{
        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numVertices, 256, numBlocks, numThreads);

		// execute the kernel
        _updatePositions<<< numBlocks, numThreads >>>(
												(LYVertex *) sortedPos,
												(float4 *) force,
												(LYVertex *) oldPos,
												numVertices);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

	}

	void updateProperties(LYVertex *sortedPos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices)
	{

		uint numThreads, numBlocks;
		computeGridSize(numVertices, 256, numBlocks, numThreads);

		// execute the kernel
		_updateProperties<<< numBlocks, numThreads>>>(
														(LYVertex*) sortedPos,
														(LYVertex*) oldPos,
														gridParticleIndex,
														cellStart,
														cellEnd,
														dev_params,
														numVertices);
		// check if kernel invocation generated an error
		getLastCudaError("Kernel _updateProperties failed!");
	}

	void updateDensities(LYVertex *sortedPos, LYVertex *oldPos, uint *gridParticleIndex, uint *cellStart, uint *cellEnd, SimParams *dev_params, size_t numVertices)
	{
		uint numThreads, numBlocks;
		computeGridSize(numVertices, 256, numBlocks, numThreads);

		// execute the kernel
		_updateDensities<<< numBlocks, numThreads>>>(
			(LYVertex*) sortedPos,
			(LYVertex*) oldPos,
			gridParticleIndex,
			cellStart,
			cellEnd,
			dev_params,
			numVertices);
		// check if kernel invocation generated an error
		getLastCudaError("Kernel _updateDensities failed!");
	}
}
