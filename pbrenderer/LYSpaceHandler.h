#pragma once

#include "LYVertex.h"
#include "LYCell.h"
class LYSpaceHandler
{
public:
	virtual void update() = 0;
	virtual void clear() = 0;

	virtual void setDeviceVertices(LYVertex *hostVertices) = 0;

	virtual LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize) = 0; // All the cells in the neighborhood of the solicited point
	virtual LYCell* getNeighboors(glm::vec3 pos, float radius) = 0; // All the cells inside the sphere defined by [p, r]
	virtual LYCell* getNeighboors(glm::vec3 pmin, glm::vec3 pmax) = 0; // All cells inside the defined AABB by [min, max]

};