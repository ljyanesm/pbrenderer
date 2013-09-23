#pragma once

#include <cuda_runtime.h>

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
class LYSpatialHash : public LYSpaceHandler
{

public:
	LYSpatialHash(void);
	LYSpatialHash(uint vbo, uint numVertices, uint3 gridSize);
	~LYSpatialHash(void);

	void update();
	void clear();

	void setVBO(uint vbo);
	void setDeviceVertices(LYVertex *hostVertices);

	LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize); // All the cells in the neighborhood of the solicited point
	LYCell* getNeighboors(glm::vec3 pos, float radius);			// All the cells inside the sphere defined by [p, r]
	LYCell* getNeighboors(glm::vec3 pmin, glm::vec3 pmax);		// All cells inside the defined AABB by [min, max]

	void dump();

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

	SimParams m_params;
};
