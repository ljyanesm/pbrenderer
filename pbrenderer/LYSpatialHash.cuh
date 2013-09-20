#include "defines.h"
#include "LYSpatialHash_kernel.cuh"
extern "C"
{
	void allocateArray(void **devPtr, int size);
	void freeArray(void *devPtr);
	void *mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource);
	void unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);
	void copyArrayFromDevice(void *host, const void *device, struct cudaGraphicsResource **cuda_vbo_resource, int size);
	void copyArrayToDevice(void *device, const void *host, int offset, int size);
	void registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource);
	void unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource);

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

}