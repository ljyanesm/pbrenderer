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
	registerGLBufferObject(m_srcVBO, &m_vboRes);

	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.cellSize = make_float3(0.01f, 0.01f, 0.01f);
	m_params.numBodies = m_numVertices;

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));
	
	allocateArray((void **)&m_sorted_points, m_numVertices*sizeof(LYVertex));

	allocateArray((void **)&m_pointHash, m_numVertices*sizeof(uint));
	allocateArray((void **)&m_pointGridIndex, m_numVertices*sizeof(uint));

	allocateArray((void **)&m_cellStart, m_numGridCells*sizeof(uint));
	allocateArray((void **)&m_cellEnd, m_numGridCells*sizeof(uint));
}


LYSpatialHash::~LYSpatialHash(void)
{
	delete m_hCellEnd;
	delete m_hCellStart;

	freeArray(m_sorted_points);
	freeArray(m_pointHash);
	freeArray(m_pointGridIndex);
	freeArray(m_cellStart);
	freeArray(m_cellEnd);

	unregisterGLBufferObject(m_vboRes);
}

void	LYSpatialHash::update()
{
	LYTimer t(true);
	LYVertex *dPos;
	dPos = (LYVertex *) mapGLBufferObject(&m_vboRes);

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

	unmapGLBufferObject(m_vboRes);
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
	copyArrayFromDevice(m_hCellStart, m_cellStart, 0, sizeof(uint)*m_numGridCells);
	copyArrayFromDevice(m_hCellEnd, m_cellEnd, 0, sizeof(uint)*m_numGridCells);
	uint maxCellSize = 0;

	for (uint i=0; i<m_numGridCells; i++)
	{
		if (m_hCellStart[i] != 0xffffffff)
		{
			uint cellSize = m_hCellEnd[i] - m_hCellStart[i];

			//            printf("cell: %d, %d particles\n", i, cellSize);
			if (cellSize > maxCellSize)
			{
				maxCellSize = cellSize;
			}
		}
	}

	printf("maximum particles per cell = %d\n", maxCellSize);
}