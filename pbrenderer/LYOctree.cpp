#include "LYOctree.h"


LYOctree::LYOctree(void)
{
}


LYOctree::~LYOctree(void)
{
}


void	LYOctree::update()
{
}

void	LYOctree::clear()
{
}

void	LYOctree::setDeviceVertices(LYVertex *hostVertices)
{
}

LYCell* LYOctree::getNeighboors(glm::vec3 pos, int neighborhoodSize)
{ 
	return new LYCell();
}

LYCell* LYOctree::getNeighboors(glm::vec3 pos, float radius)
{
	return new LYCell();
}

LYCell* LYOctree::getNeighboors(glm::vec3 pmin, glm::vec3 pmax)
{
	return new LYCell();
}

