#pragma once
#include <vector>
#include "LYSpaceHandler.h"
#include "LYCamera.h"
#include "LYMesh.h"

class LYWorld
{
public:
	LYWorld(void);
	~LYWorld(void);

private:
	std::vector<LYMesh> m_objects;
	std::vector<LYCamera> m_cameras;
};

