#pragma once

/*

Based on Tero Karras 2012
Maximizing parallelism in the construction of BVHs, Octrees and k-d Trees.

*/

#include "defines.h"

#include "LYSpaceHandler.h"

class LYOctree : public LYSpaceHandler
{
public:
	LYOctree(void);
	~LYOctree(void);

	void update();
	void clear();

	void setDeviceVertices(LYVertex *hostVertices);

private:
	uint *m_mortonCodes;
};

