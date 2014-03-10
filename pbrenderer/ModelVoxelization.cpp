#include "ModelVoxelization.h"


ModelVoxelization::ModelVoxelization(LYMesh *mesh, uint gSz)
{
	gridSize = gSz;
	_initialize(mesh, gSz+1);

	floodStartIndex = 1;

	while (!_floodFill())
	{
		gridSize /= 2;
		std::cout << "Flood fill failed, there was a hole in the surface, new gridSize = " << gridSize << std::endl;
		_initialize(mesh, gridSize+1);
	}
	std::cout << "Flood fill, succeeded with a gridSize = " << gridSize << std::endl;

}


void ModelVoxelization::_initialize( LYMesh *mesh, uint gSz )
{
	gridSize2 = gSz*gSz;
	numVoxels = gSz*gSz*gSz;
	data.resize(numVoxels);
	ignorePixels.resize(numVoxels);
	std::fill(data.begin(), data.end(), NON_FILLED);
	std::fill(ignorePixels.begin(), ignorePixels.end(), false);
	for (uint i = 0; i < mesh->getNumVertices(); i++)
	{
		glm::vec3 modelPos(mesh->getVertices()->at(i).m_pos.x, mesh->getVertices()->at(i).m_pos.y, mesh->getVertices()->at(i).m_pos.z);
		modelPos =  ( modelPos - mesh->getMinPoint() ) / (mesh->getScale());

		glm::vec3 voxelPosition(floor(modelPos.x*gridSize), floor(modelPos.y*gridSize), floor(modelPos.z*gridSize));
		uint index = getIndex(voxelPosition);
		data[index] = FILLED;
	}
}



ModelVoxelization::~ModelVoxelization(void)
{
	data.clear();
}

glm::vec3 ModelVoxelization::getIndex( uint index )
{
	glm::vec3 p;
	p.x = (float)(int)(index % gridSize);
	p.y = (float)(int)((index/gridSize) % gridSize);
	p.z = (float)(int)(index / (gridSize2));
	return p;
}

uint ModelVoxelization::getIndex( glm::vec3 index )
{
	if (index.x < 0 || index.x > gridSize ||
		index.y < 0 || index.y > gridSize ||
		index.z < 0 || index.z > gridSize ) 
		return -1;
	return (uint) (index.x + index.y * gridSize + index.z * gridSize2);
}

bool ModelVoxelization::_floodFill()
{
	if(floodStartIndex > 0 && floodStartIndex < numVoxels)
	{
		// store an active queue of pixels to be processed
		std::vector<unsigned int> queue;
		queue.push_back(floodStartIndex);
		// while there are pixels to process
		while(!queue.empty())
		{
			// store the index
			uint sampleIndex = queue.back();
			// remove the index from the queue
			queue.pop_back();
			// should we ignore this pixel
			if(!ignorePixels[sampleIndex])
			{
				// get the sample value
				uint sampleValue = this->data[sampleIndex];
				// if the sample value and the target value are equal
				if(sampleValue == NON_FILLED)
				{
					// fill the current pixel
					this->data[sampleIndex] = FILLED;
					// ignore this pixel in future
					ignorePixels[sampleIndex] = true;
					// get the x, y, z index components
					int iX = 0, iY = 0, iZ = 0;
					glm::vec3 iIndex(getIndex(sampleIndex));
					// get 4-connectivity neighbourhood
					uint neighbourhood[6];
					neighbourhood[0] = getIndex(iIndex - glm::vec3(-1 , +0 , +0));
					neighbourhood[1] = getIndex(iIndex - glm::vec3(+1 , +0 , +0));
					neighbourhood[2] = getIndex(iIndex - glm::vec3(+0 , +1 , +0));
					neighbourhood[3] = getIndex(iIndex - glm::vec3(+0 , -1 , +0));
					neighbourhood[4] = getIndex(iIndex - glm::vec3(+0 , +0 , +1));
					neighbourhood[5] = getIndex(iIndex - glm::vec3(+0 , +0 , -1));
					// add pixels with valid indices to the queue
					for(unsigned int i=0; i<6; i++)
					{
						if(neighbourhood[i] != -1 && neighbourhood[i] < numVoxels) // returns -1 if outside of array dimensions
						{
							queue.push_back(neighbourhood[i]);
						}
					}
				}
			}
		}
	}
	std::vector<int>::const_iterator location = std::find(data.begin(), data.end(), NON_FILLED);

	if (location != data.end()) {
		std::cout << "The non-filled element is at index: " << location - data.begin() << std::endl;
		return true;
	}
	else {
		std::cout << "All the elements in data have been filled" << std::endl;
		return false;
	}
}

LYMesh		* ModelVoxelization::getModel()
{
	std::vector<LYVertex>	modelVertices;
	std::vector<uint>		modelIndices;
	uint numIndices = 0;
	for (uint i = 0; i < numVoxels; i++)
	{
		if (data[i] == NON_FILLED)
		{
			LYVertex vert;
			glm::vec3 index = getIndex(i);
			vert.m_pos = make_float3(index.x / (float) gridSize, index.y / (float) gridSize, index.z / (float) gridSize);
			modelVertices.push_back(vert);
			modelIndices.push_back(numIndices);
			numIndices++;
		}

	}

	return new LYMesh(modelVertices, modelIndices);
}
