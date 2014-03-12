#pragma once
#include "helper_math.h"
#include "LYMesh.h"
#include "defines.h"
#include <vector>
#include <algorithm>
class ModelVoxelization
{
	enum{
		NON_FILLED = 2,
		FILLED = 0
	} VOXEL_STATUS;
	uint floodStartIndex;
	uint numVoxels;
	uint gridSize;
	uint gridSize2;
	std::vector<int> data;
	std::vector<bool> ignorePixels;

public:
	ModelVoxelization(LYMesh *mesh, uint gSz);
	~ModelVoxelization(void);
	LYMesh		*getModel();
private:
	void		_initialize(LYMesh *mesh, uint gSz); 
	bool		_floodFill();
	glm::vec3	getIndex(uint index);
	uint		getIndex(glm::vec3 index);
};

