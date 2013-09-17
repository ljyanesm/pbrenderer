#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "defines.h"

#include "LYSpaceHandler.h"
class LYSpatialHash : LYSpaceHandler
{

public:
	LYSpatialHash(void);
	~LYSpatialHash(void);

	void update(){}
	void clear(){}

	void setDeviceVertices(LYVertex *hostVertices){}

	LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize){ return new LYCell();} // All the cells in the neighborhood of the solicited point
	LYCell* getNeighboors(glm::vec3 pos, float radius){return new LYCell();} // All the cells inside the sphere defined by [p, r]
	LYCell* getNeighboors(glm::vec3 pmin, glm::vec3 pmax){return new LYCell();} // All cells inside the defined AABB by [min, max]

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

