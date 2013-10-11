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

	m_forceFeedback = new float3[1];
	m_hParams	=	new SimParams();

	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numGridCells;
	m_params.cellSize = make_float3(0.03f, 0.03f, 0.03f);
	m_params.numBodies = m_numVertices;

	m_hParams->gridSize		= m_gridSize;
	m_hParams->numCells		= m_numGridCells;
	m_hParams->cellSize		= make_float3(0.03f, 0.03f, 0.03f);
	m_hParams->numBodies	= m_numVertices;

	m_hParams->dmin			= 0.75f;
	m_hParams->w_tot		= 0.0f;
	m_hParams->epsilon		= 0.001f;
	m_hParams->beta			= 0.01f;
	m_hParams->gamma		= 0.01f;
	m_hParams->phi			= 0.01f;
	m_hParams->forceSpring	= 0.2f;
	m_hParams->Xc			= make_float3(INF);
	m_hParams->RMAX			= 2.0f;
	m_hParams->RMIN			= 0.2f;
	m_hParams->Ax			= make_float3(0.0f);
	m_hParams->Nx			= make_float3(0.0f);


	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));
	
	LYCudaHelper::allocateArray((void **)&m_sorted_points, m_numVertices*sizeof(LYVertex));

	LYCudaHelper::allocateArray((void **)&m_pointHash, m_numVertices*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_pointGridIndex, m_numVertices*sizeof(uint));

	LYCudaHelper::allocateArray((void **)&m_cellStart, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_cellEnd, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_dForceFeedback, sizeof(float3));
	LYCudaHelper::allocateArray((void **)&m_dParams, sizeof(SimParams));

	cudaDeviceSynchronize();
	m_dirtyPos = true;
}


LYSpatialHash::~LYSpatialHash(void)
{
	delete m_hCellEnd;
	delete m_hCellStart;
	delete m_forceFeedback;

	LYCudaHelper::freeArray(m_sorted_points);
	LYCudaHelper::freeArray(m_pointHash);
	LYCudaHelper::freeArray(m_pointGridIndex);
	LYCudaHelper::freeArray(m_cellStart);
	LYCudaHelper::freeArray(m_cellEnd);
	LYCudaHelper::freeArray(m_forceFeedback);

	LYCudaHelper::unregisterGLBufferObject(m_vboRes);
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


void	LYSpatialHash::update()
{
	LYTimer t(true);
	LYVertex *dPos;

	// update constants
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));
	if (m_dirtyPos) {
		setParameters(&m_params);
		dPos = (LYVertex *) LYCudaHelper::mapGLBufferObject(&m_vboRes);
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
		m_dirtyPos = false;
	}
}

void
	LYSpatialHash::dump()
{
	// dump grid information
	LYCudaHelper::copyArrayFromDevice(m_hCellStart, m_cellStart, 0, sizeof(uint)*m_numGridCells);
	LYCudaHelper::copyArrayFromDevice(m_hCellEnd, m_cellEnd, 0, sizeof(uint)*m_numGridCells);
	LYVertex *m_hPositions = new LYVertex[m_numVertices];
	LYCudaHelper::copyArrayFromDevice(m_hPositions, m_sorted_points, 0, sizeof(LYVertex)*m_numVertices);

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

	for (uint i = 0; i < m_numVertices; i++)
	{
		printf("P(%u) = (%6.3f, %6.3f, %6.3f)\n", i, m_hPositions[i].m_pos.x, m_hPositions[i].m_pos.y, m_hPositions[i].m_pos.z);
	}
}

void LYSpatialHash::calculateCollisions( float3 pos )
{
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));

	collisionCheck(pos, m_sorted_points, m_pointGridIndex, m_cellStart, m_cellEnd, m_dParams, m_numVertices);
	LYCudaHelper::copyArrayFromDevice(m_hParams, m_dParams, 0, sizeof(SimParams));
}

float3 LYSpatialHash::getForceFeedback(float3 pos)
{
	calculateCollisions(pos);
	float3 Ax = m_hParams->Ax;
	float3 Nx = m_hParams->Nx;
	float Fx = dot(Nx, (pos - Ax));
	if (Fx < 0.0f ){



		return -Nx * Fx;
	}
	return make_float3(0.0f);
}

float3 LYSpatialHash::calculateFeedbackUpdateProxy( LYVertex *pos )
{
	static float3 tgPlaneNormal = make_float3(0.0f);
	static float3 Psurface  = make_float3(0.0f);
	static bool touched = false;

	float3 colliderPos = pos->m_pos;
	float3 Pseed = make_float3(0.0f);
	float error = 9999.999f;
	float3 Ax, Nx;
	float Fx;
	float3 dPseed;

	if (!touched){
		calculateCollisions(colliderPos);
		Ax = m_hParams->Ax/m_hParams->w_tot;
		Nx = m_hParams->Nx/length(m_hParams->Nx);

		Fx = dot(Nx, colliderPos - Ax);
		if (Fx < 0.0f){
			touched = true;
			Pseed = Ax;
			Psurface = Pseed;
			pos->m_normal = Psurface;
			tgPlaneNormal = Nx;
			float3 f = (Psurface - colliderPos);
			return f;
		} else {
			pos->m_normal = colliderPos;
			touched = false;
		}
	} else {
		float dist = dot(colliderPos - Psurface, tgPlaneNormal);
		if (dist <= 0.0f)
		{
			float3 P0p = colliderPos - Psurface;
			float dist = dot(P0p, tgPlaneNormal);
			P0p = colliderPos - dist*tgPlaneNormal;
			Pseed = P0p;
			calculateCollisions(Pseed);
			Ax = m_hParams->Ax/m_hParams->w_tot;
			Nx = m_hParams->Nx/length(m_hParams->Nx);
			Psurface = Ax;
			pos->m_normal = Psurface;
			tgPlaneNormal = Nx;
			float3 f = (Psurface - colliderPos);
			return f;
		} else {
			touched = false;
			pos->m_normal = colliderPos;
		}
	}

	return make_float3(0.0f);
}
