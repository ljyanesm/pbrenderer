#pragma once

/*

Based on Tero Karras 2012
Maximizing parallelism in the construction of BVHs, Octrees and k-d Trees.

*/

#include "defines.h"
#include "LYCell.h"
#include "LYVertex.h"

class LYOctree
{
public:
	LYOctree(void);
	~LYOctree(void);

	void update();
	void clear();

	void setDeviceVertices(LYVertex *hostVertices);

	LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize); // All the cells in the neighborhood of the solicited point
	LYCell* getNeighboors(glm::vec3 pos, float radius); // All the cells inside the sphere defined by [p, r]
	LYCell* getNeighboors(glm::vec3 min, glm::vec3 max); // All cells inside the defined AABB by [min, max]

private:
	uint *m_mortonCodes;
};
