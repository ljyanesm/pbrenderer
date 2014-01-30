#include "LYSpatialHash.h"


LYSpatialHash::LYSpatialHash(void)
{
}

LYSpatialHash::LYSpatialHash(uint vbo, size_t numVertices, uint3 gridSize) :
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
	m_params.R			= 0.2f;

	m_hParams->gridSize		= m_gridSize;
	m_hParams->numCells		= m_numGridCells;
	m_hParams->cellSize		= make_float3(0.03f, 0.03f, 0.03f);
	m_hParams->numBodies	= m_numVertices;

	m_hParams->dmin			= 0.75f;
	m_hParams->w_tot		= 0.0f;
	m_hParams->R			= 0.2f;
	m_hParams->Xc			= make_float3(INF);
	m_hParams->RMAX			= 2.0f;
	m_hParams->RMIN			= 0.2f;
	m_hParams->Ax			= make_float3(0.0f);
	m_hParams->Nx			= make_float3(0.0f);


	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));
	
	size_t classSize = m_numVertices*sizeof(LYVertex) + m_numVertices*sizeof(uint)*2 + m_numGridCells*sizeof(uint)*2;
	(classSize > 1024*1024) ? 
		printf("This mesh requires: %d MB\n", classSize / (1024*1024)) :
	(classSize > 1024) ? printf("This mesh requires: %d Kb\n", classSize / (1024)):
		printf("This mesh requires: %d Bytes\n", classSize);

	LYCudaHelper::printMemInfo();
	LYCudaHelper::allocateArray((void **)&m_sorted_points, m_numVertices*sizeof(LYVertex));

	LYCudaHelper::allocateArray((void **)&m_pointHash, m_numVertices*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_pointGridIndex, m_numVertices*sizeof(uint));

	LYCudaHelper::allocateArray((void **)&m_cellStart, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_cellEnd, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_dForceFeedback, sizeof(float3));
	LYCudaHelper::allocateArray((void **)&m_dParams, sizeof(SimParams));
	LYCudaHelper::printMemInfo();

	cudaDeviceSynchronize();
	m_dirtyPos = true;
}


LYSpatialHash::~LYSpatialHash(void)
{
	delete m_hCellEnd;
	delete m_hCellStart;
	delete m_forceFeedback;
	delete m_hParams;	

	LYCudaHelper::freeArray(m_sorted_points);
	std::cout << "m_sorted_points" << std::endl;
	LYCudaHelper::freeArray(m_pointHash);
	std::cout << "m_pointHash" << std::endl;
	LYCudaHelper::freeArray(m_pointGridIndex);
	std::cout << "m_pointGridIndex" << std::endl;
	LYCudaHelper::freeArray(m_cellEnd);
	std::cout << "m_cellEnd" << std::endl;
	LYCudaHelper::freeArray(m_cellStart);
	std::cout << "m_cellStart" << std::endl;
	LYCudaHelper::freeArray(m_dForceFeedback);
	std::cout << "m_dForceFeedback" << std::endl;
	LYCudaHelper::freeArray(m_dParams);
	std::cout << "m_dParams" << std::endl;
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

void LYSpatialHash::setInfluenceRadius(float r){
	m_params.R = r;
	this->m_hParams->R = r;
	setParameters(&m_params);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));
}

void	LYSpatialHash::update()
{
	LYVertex *dPos;

	// update constants
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	setParameters(&m_params);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));
	if (true || m_dirtyPos) {
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

void LYSpatialHash::dump()
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
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));
	collisionCheck(pos, m_sorted_points, m_pointGridIndex, m_cellStart, m_cellEnd, m_dParams, m_numVertices);
	LYCudaHelper::copyArrayFromDevice(m_hParams, m_dParams, 0, sizeof(SimParams));
}

float3 LYSpatialHash::calculateFeedbackUpdateProxy( LYVertex *pos )
{
	static float3 tgPlaneNormal = make_float3(0.0f);
	static float3 Psurface  = make_float3(0.0f);
	static bool touched = false;

	float3 colliderPos = pos->m_pos;
	float3 Pseed = make_float3(0.0f);
	float3 dP = make_float3(0.0f);
	float error = 9999.999f;
	float3 Ax, Nx;
	float Fx;

	if (!touched){
		calculateCollisions(colliderPos);
		Ax = m_hParams->Ax/m_hParams->w_tot;
		Nx = m_hParams->Nx/length(m_hParams->Nx);

		Fx = dot(Nx, colliderPos - Ax);

		if (Fx < 0.0f && dot(colliderPos - Ax, Nx) < 0.0f){
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
		if (dist <= 0)
		{
			float3 P0p = colliderPos - Psurface;
			float dist = dot(P0p, tgPlaneNormal);
			P0p = colliderPos - dist*tgPlaneNormal;
			Pseed = P0p;
			do{
				calculateCollisions(Pseed);
				Ax = m_hParams->Ax/m_hParams->w_tot;
				Nx = m_hParams->Nx/length(m_hParams->Nx);
				dP = - (Ax*Nx)/(Nx*Nx);
				Pseed += dP;
			} while (length(dP) < 0.001);
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
