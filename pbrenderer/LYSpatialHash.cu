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
#include "thrust\extrema.h"

#include "LYCudaHelper.cuh"
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

	void collisionCheck(const ccConfiguration &arguments)
	{
		// Get the size of the collision radius
		float3 QP = arguments.pos;
		float r = arguments.R;

		// Calculate the size of the neighborhood based on the radius
		int nSize = (int) roundf((float) (r / arguments.voxSize));
		// Calculate the voxel position of the query point
		glm::vec4 voxelPos = glm::vec4(QP.x, QP.y, QP.z, 0) / (float) nSize;

		getLastCudaError("Before collisionCheck Kernel execution failed");

		uint numThreads, numBlocks;
		computeGridSize(arguments.numToolVertices, 32, numBlocks, numThreads);

		// Using dynamic parallelism only execute threads on the neighborhood of the selected QP

		switch (arguments.collisionCheckType)
		{
		case CollisionCheckType::NAIVE:
			{
				SingleCollisionCheckArgs args;
				args.pos = arguments.pos;
				args.sortedPos = arguments.sortedPos;
				args.force = arguments.force;
				args.forceVector = arguments.forceVector;
				args.gridParticleIndex = arguments.gridParticleIndex;
				args.cellStart = arguments.cellStart;
				args.cellEnd = arguments.cellEnd;
				args.dev_params = arguments.dev_params;
				args.numVertices = arguments.numVertices;

				_naiveDynamicCollisionCheckD<<< 1, 1 >>>(args);	
			} break;
		case CollisionCheckType::DYNAMIC:
			{
				SingleCollisionCheckArgs args;
				args.pos = arguments.pos;
				args.sortedPos = arguments.sortedPos;
				args.force = arguments.force;
				args.forceVector = arguments.forceVector;
				args.gridParticleIndex = arguments.gridParticleIndex;
				args.cellStart = arguments.cellStart;
				args.cellEnd = arguments.cellEnd;
				args.dev_params = arguments.dev_params;
				args.numVertices = arguments.numVertices;


				_dynamicCollisionCheckD<<< numBlocks, numThreads >>>(args);
			} break;
		case CollisionCheckType::TWO_STEP:
			{
				uint totalNeighborhoodSize = (2*nSize+1);
				totalNeighborhoodSize = totalNeighborhoodSize*totalNeighborhoodSize*totalNeighborhoodSize;
				
				if (totalNeighborhoodSize > arguments.maxNumCollectionElements) totalNeighborhoodSize = arguments.maxNumCollectionElements;

				computeGridSize(totalNeighborhoodSize, 256, numBlocks, numThreads);

				InteractionCellsArgs args;
				
				args.pos					= arguments.pos;
				args.forceVector			= arguments.forceVector;
				args.numNeighborCells		= totalNeighborhoodSize;
				args.maxSearchRange			= arguments.maxSearchRange;
				args.maxSearchRangeSq		= arguments.maxSearchRangeSq;
				args.gridParticleIndex		= arguments.gridParticleIndex;
				args.cellStart				= arguments.cellStart;
				args.cellEnd				= arguments.cellEnd;
				args.sortedPos				= arguments.sortedPos;
				args.force					= arguments.force;
				args.dev_params				= arguments.dev_params;
				args.totalVertices			= arguments.totalVertices_2Step;
				args.collectionCellStart	= arguments.collectionCellStart;
				args.collectionVertices		= arguments.collectionVertices;
				args.totalNeighborhoodSize	= totalNeighborhoodSize;
				
				// Collect all the cells and number of vertices that are in the area of the query point
				_collectInteractionCells<<< numBlocks, numThreads >>> (args);

				//Get the max number of vertices in a cell
				thrust::device_ptr<uint> dev_ptr = thrust::device_pointer_cast(arguments.collectionVertices);
				thrust::device_ptr<uint> maxElem = thrust::max_element(dev_ptr, dev_ptr + totalNeighborhoodSize);
				uint maxPointsCell = *maxElem;

				CollisionCheckArgs args2;
				args2.forceVector			= arguments.forceVector;
				args2.pos					= arguments.pos;
				args2.collectionCellStart	= arguments.collectionCellStart;
				args2.collectionVertices	= arguments.collectionVertices;
				args2.sortedPos				= arguments.sortedPos;
				args2.force					= arguments.force;
				args2.gridParticleIndex		= arguments.gridParticleIndex;
				args2.dev_params			= arguments.dev_params;

				// Launch maxPointsCell threads on each block, and launch one block per interaction cell
				// read the collectionCellStart and collectionVertices to shared mem, then decide if the
				// current thread is inside the collectionVertices boundary and compute the interaction
				if (maxPointsCell > 0) _computeCollisionCheck <<< totalNeighborhoodSize, maxPointsCell >>> (args2);
				gpuErrchk(cudaPeekAtLastError());
				checkCudaErrors(cudaMemset(arguments.collectionVertices, 0, totalNeighborhoodSize*sizeof(uint)));

			} break;
		case CollisionCheckType::BASIC:
			{
				// thread per particle
				uint numThreads, numBlocks;
				computeGridSize(arguments.numVertices, 512, numBlocks, numThreads);

				// execute the kernel
				_collisionCheckD<<< numBlocks, numThreads >>>(	arguments.pos,
					(LYVertex *) arguments.sortedPos,
					(float4 *) arguments.force,
					arguments.forceVector,
					arguments.gridParticleIndex,
					arguments.cellStart,
					arguments.cellEnd,
					arguments.dev_params,
					arguments.numVertices);
				 //check if kernel invocation generated an error
				gpuErrchk(cudaPeekAtLastError());
			} break;
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

	void computeOvershoot(OvershootArgs args)
	{
		uint numThreads, numBlocks;
		computeGridSize(args.numVertices, 512, numBlocks, numThreads);
		// execute the kernel
		_computeOvershoot<<<numBlocks, numThreads>>>(args);
		cudaDeviceSynchronize();
		// check if kernel invocation generated an error
		getLastCudaError("Kernel _computeOvershoot failed!");
	}
}

