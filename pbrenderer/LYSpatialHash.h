#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <device_functions.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include "glm/glm.hpp"
#include "defines.h"

#include "Collider.h"
#include "LYCudaHelper.cuh"
#include "LYSpaceHandler.h"
#include "LYSpatialHash.cuh"
#include "LYSpatialHash_kernel.cuh"

class LYSpatialHash : public LYSpaceHandler
{

public:
	LYSpatialHash(void);
	LYSpatialHash(uint vbo, size_t numVertices, uint3 gridSize);
	~LYSpatialHash(void);

	void	update();
	void	clear();

	void	setVBO(uint vbo);
	void	setDeviceVertices(LYVertex *hostVertices);

	void	setInfluenceRadius(float r);

	void	calculateCollisions(float3 pos);
	float3	calculateFeedbackUpdateProxy(Collider *pos);

	void	dump();

private:
	cudaGraphicsResource *m_vboRes;

	LYVertex	*m_src_points;			// Source points saved in the GPU
	LYVertex	*m_sorted_points;		// Sorted points saved in the GPU

	uint		*m_hCellStart;
	uint		*m_hCellEnd;

	uint		*m_cellStart;
	uint		*m_cellEnd;
	uint		*m_pointHash;
	uint		*m_pointGridIndex;
	uint		m_gridSortBits;

	uint		m_srcVBO;
	size_t		m_numVertices;

	uint3		m_gridSize;
	uint 		m_numGridCells;

	float3		*m_dForceFeedback;
	float3		*m_uForceFeedback;
	float3		*m_forceFeedback;

	bool		m_dirtyPos;

	SimParams	m_params;
	SimParams	*m_hParams;
	SimParams	*m_dParams;
};
