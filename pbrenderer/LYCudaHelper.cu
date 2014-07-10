#include "LYCudaHelper.cuh"

void LYCudaHelper::allocateHostArray(void **devPtr, size_t size, unsigned int flag)
{
	checkCudaErrors(cudaHostAlloc(devPtr, size, cudaHostAllocMapped));
}

void LYCudaHelper::memsetDeviceArray(void **devPtr, size_t size, int value)
{
	checkCudaErrors(cudaMemset(devPtr, value, size));
}

void LYCudaHelper::freeHostArray(void *devPtr)
{
	checkCudaErrors(cudaFreeHost(devPtr));
}

void LYCudaHelper::allocateArray(void **devPtr, size_t size)
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

void LYCudaHelper::getMappedPointer(void **device, void *host, uint flag)
{
	checkCudaErrors(cudaHostGetDevicePointer(device, host, flag));
}

void LYCudaHelper::printMemInfo()
{
	size_t gpuFreeMem, gpuTotalMem;
	cudaError_t error(cudaMemGetInfo(&gpuFreeMem, &gpuTotalMem));
	std::cout << "cudaMemGetInfo error code (" << error << "): " << cudaGetErrorString(error) << std::endl;
	gpuFreeMem /= 1024*1024;
	gpuTotalMem /= 1024*1024;
	printf("Total amount of MB available: %Iu MB \nTotal amount of device memory: %Iu MB\n", gpuFreeMem, gpuTotalMem);
}