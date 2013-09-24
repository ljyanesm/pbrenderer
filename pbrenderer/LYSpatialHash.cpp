#include "LYSpatialHash.h"


LYSpatialHash::LYSpatialHash(void)
{
}

LYSpatialHash::LYSpatialHash(uint vbo, uint numVertices, uint3 gridSize) :
m_gridSize(gridSize)
{
	m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	m_srcVBO		= vbo;
	m_numVertices	= numVertices;
	LYCudaHelper::registerGLBufferObject(m_srcVBO, &m_vboRes);

	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.cellSize = make_float3(0.01f, 0.01f, 0.01f);
	m_params.numBodies = m_numVertices;

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));
	
	LYCudaHelper::allocateArray((void **)&m_sorted_points, m_numVertices*sizeof(LYVertex));

	LYCudaHelper::allocateArray((void **)&m_pointHash, m_numVertices*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_pointGridIndex, m_numVertices*sizeof(uint));

	LYCudaHelper::allocateArray((void **)&m_cellStart, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_cellEnd, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateHostArray((void **) &m_forceFeedback, sizeof(float3));
}


LYSpatialHash::~LYSpatialHash(void)
{
	delete m_hCellEnd;
	delete m_hCellStart;

	LYCudaHelper::freeArray(m_sorted_points);
	LYCudaHelper::freeArray(m_pointHash);
	LYCudaHelper::freeArray(m_pointGridIndex);
	LYCudaHelper::freeArray(m_cellStart);
	LYCudaHelper::freeArray(m_cellEnd);
	LYCudaHelper::freeHostArray(&m_forceFeedback);

	LYCudaHelper::unregisterGLBufferObject(m_vboRes);
}

void	LYSpatialHash::update()
{
	LYTimer t(true);
	LYVertex *dPos;
	dPos = (LYVertex *) LYCudaHelper::mapGLBufferObject(&m_vboRes);

	// update constants
	setParameters(&m_params);

	// calculate grid hash
	calcHash(
		m_pointHash,
		m_pointGridIndex,
		dPos,
		m_numVertices);

	// sort particles based on hash
	sortParticles(m_pointHash, m_pointGridIndex, m_numVertices);

	// reorder particle arrays into sorted order and
	// find start and end of each cell
	reorderDataAndFindCellStart(
		m_cellStart,
		m_cellEnd,
		m_sorted_points,
		m_pointHash,
		m_pointGridIndex,
		dPos,
		m_numVertices,
		m_numGridCells);

	LYCudaHelper::unmapGLBufferObject(m_vboRes);
}

void	LYSpatialHash::clear()
{
}

void	LYSpatialHash::setVBO(uint vbo)
{
	m_srcVBO = vbo;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_vboRes, vbo, cudaGraphicsMapFlagsWriteDiscard));

}
void	LYSpatialHash::setDeviceVertices(LYVertex *hostVertices)
{

}

LYCell* LYSpatialHash::getNeighboors(glm::vec3 pos, int neighborhoodSize)
{ 
	return new LYCell();
}

LYCell* LYSpatialHash::getNeighboors(glm::vec3 pos, float radius)
{
	return new LYCell();
}

LYCell* LYSpatialHash::getNeighboors(glm::vec3 pmin, glm::vec3 pmax)
{
	return new LYCell();
}


void
	LYSpatialHash::dump()
{
	// dump grid information
	LYCudaHelper::copyArrayFromDevice(m_hCellStart, m_cellStart, 0, sizeof(uint)*m_numGridCells);
	LYCudaHelper::copyArrayFromDevice(m_hCellEnd, m_cellEnd, 0, sizeof(uint)*m_numGridCells);
	uint maxCellSize = 0;

	for (uint i=0; i<m_numGridCells; i++)
	{
		if (m_hCellStart[i] != 0xffffffff)
		{
			uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

			if (cellSize > maxCellSize)
			{
				maxCellSize = cellSize;
			}
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}

void LYSpatialHash::calculateCollisions( float3 pos )
{
	m_forceFeedback = float3();
	collisionCheck(pos, m_sorted_points, m_pointGridIndex, m_cellStart, m_cellEnd, m_forceFeedback, m_numVertices);
}
