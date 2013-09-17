#pragma once
class LYSpaceHandler
{
public:
	LYSpaceHandler(void);
	~LYSpaceHandler(void);

	virtual void update();
	virtual void clear();

	virtual void setDeviceVertices(LYVertex *hostVertices);

	virtual LYCell* getNeighboors(glm::vec3 pos, int neighborhoodSize); // All the cells in the neighborhood of the solicited point
	virtual LYCell* getNeighboors(glm::vec3 pos, float radius); // All the cells inside the sphere defined by [p, r]
	virtual LYCell* getNeighboors(glm::vec3 min, glm::vec3 max); // All cells inside the defined AABB by [min, max]

private:


};