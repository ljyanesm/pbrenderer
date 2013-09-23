#include "LYCudaHelper.cuh"

void LYCudaHelper::allocateArray(void **devPtr, int size)
{
	checkCudaErrors(cudaMalloc(devPtr, size));
}

void LYCudaHelper::freeArray(void *devPtr)
{
	checkCudaErrors(cudaFree(devPtr));
}

void LYCudaHelper::registerGLBufferObject(uint vbo, struct cudaGraphicsResource **cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(cuda_vbo_resource, vbo,
		cudaGraphicsMapFlagsNone));
}

void LYCudaHelper::unregisterGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
}

void *LYCudaHelper::mapGLBufferObject(struct cudaGraphicsResource **cuda_vbo_resource)
{
	void *ptr;
	checkCudaErrors(cudaGraphicsMapResources(1, cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes,
		*cuda_vbo_resource));
	return ptr;
}

void LYCudaHelper::unmapGLBufferObject(struct cudaGraphicsResource *cuda_vbo_resource)
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void LYCudaHelper::copyArrayToDevice(void *device, const void *host, int offset, int size)
{
	checkCudaErrors(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void LYCudaHelper::copyArrayFromDevice(void *host, const void *device,
struct cudaGraphicsResource **cuda_vbo_resource, int size)
{
	if (cuda_vbo_resource)
	{
		device = mapGLBufferObject(cuda_vbo_resource);
	}

	checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));

	if (cuda_vbo_resource)
	{
		unmapGLBufferObject(*cuda_vbo_resource);
	}
}

