#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <device_functions.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include "glm/glm.hpp"
#include "defines.h"

#include "LYTimer.h"

#include "LYCudaHelper.cuh"
#include "LYSpaceHandler.h"
#include "LYSpatialHash.cuh"
#include "LYSpatialHash_kernel.cuh"

// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

class LYSpatialHash : public LYSpaceHandler
{

public:
	LYSpatialHash(void);
	LYSpatialHash(uint vbo, uint numVertices, uint3 gridSize);
	~LYSpatialHash(void);

	void	update();
	void	clear();

	void	setVBO(uint vbo);
	void	setDeviceVertices(LYVertex *hostVertices);

	LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize); // All the cells in the neighborhood of the solicited point
	LYCell* getNeighboors(glm::vec3 pos, float radius);			// All the cells inside the sphere defined by [p, r]
	LYCell* getNeighboors(glm::vec3 pmin, glm::vec3 pmax);		// All cells inside the defined AABB by [min, max]

	void	setInfluenceRadius(float r);

	void	selectVisiblePoints();

	void	calculateCollisions(float3 pos);

	float3	getForceFeedback(float3 pos);
	float3	calculateFeedbackUpdateProxy(LYVertex *pos);

	void	dump();

private:
	LYVertex *m_src_points;			// Source points saved in the GPU
	LYVertex *m_sorted_points;		// Sorted points saved in the GPU


	uint	*m_hCellStart;
	uint	*m_hCellEnd;

	uint	*m_cellStart;
	uint	*m_cellEnd;
	uint	*m_pointHash;
	uint	*m_pointGridIndex;
	uint	m_gridSortBits;

	uint	m_srcVBO;
	uint	m_numVertices;

	uint3	m_gridSize;
	uint 	m_numGridCells;
	cudaGraphicsResource *m_vboRes;

	float3 *m_dForceFeedback;
	float3 *m_uForceFeedback;
	float3 *m_forceFeedback;

	bool	m_dirtyPos;

	SimParams m_params;

	SimParams *m_hParams;
	SimParams *m_dParams;

};
