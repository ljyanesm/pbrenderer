#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "defines.h"
#include "LYVertex.h"
#include "LYCell.h"
class LYSpatialHash
{

public:
	LYSpatialHash(void);
	~LYSpatialHash(void);

	void update();
	void clear();

	void setDeviceVertices(LYVertex *hostVertices);

	LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize); // All the cells in the neighborhood of the solicited point
	LYCell* getNeighboors(glm::vec3 pos, float radius); // All the cells inside the sphere defined by [p, r]
	LYCell* getNeighboors(glm::vec3 min, glm::vec3 max); // All cells inside the defined AABB by [min, max]

private:
	LYVertex *m_src_points;			// Source points saved in the GPU
	LYVertex *m_sorted_points;		// Sorted points saved in the GPU

	void calcHash();
	void reorderDataAndFindCellStart();

	uint	*m_cellStart;
	uint	*m_cellEnd;
	uint	*m_pointHash;
	uint	*m_pointGridIndex;
	uint	m_gridSortBits;
};

