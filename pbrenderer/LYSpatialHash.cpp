#include "LYSpatialHash.h"

LYSpatialHash::LYSpatialHash(uint vbo, size_t numVertices, uint3 gridSize, std::string modelName) :
	m_gridSize(gridSize),
	renderingMethod(HapticRenderingMethods::IMPLICIT_SURFACE),
	maxSearchRange(7),
	maxSearchRangeSq(maxSearchRange*maxSearchRange),
	m_maxNumCollectionElements((2*maxSearchRange+1)*(2*maxSearchRange+1)*(2*maxSearchRange+1)) // (2r+1)^3
{
	m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	m_srcVBO		= vbo;
	m_numVertices	= numVertices;
	
	m_numToolVertices = 1;

	m_forceFeedback = make_float4(0.0f);
	m_hParams	=	new SimParams();

	m_params.gridSize = m_gridSize;
	m_params.cellSize = make_float3(1.0f/gridSize.x, 1.0f/gridSize.y, 1.0f/gridSize.z);
	m_params.R			= m_params.cellSize.x*1.5f;
	neighborhoodRadius	= 0.02f;

	maxR = m_params.cellSize.x * 7.f;

	m_hParams->gridSize		= m_gridSize;
	m_hParams->cellSize		= make_float3(1.0f/gridSize.x, 1.0f/gridSize.y, 1.0f/gridSize.z);

	m_hParams->w_tot		= 0.0f;
	m_hParams->Ax			= make_float3(0.0f);
	m_hParams->Nx			= make_float3(0.0f);

	m_touched				= false;
	m_updatePositions		= false;

	m_collisionCheckType	= CollisionCheckType::BASIC;

	m_hCellStart = new uint[m_numGridCells];
	memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

	m_hCellEnd = new uint[m_numGridCells];
	memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));
	
	size_t classSize = 
		m_numVertices*sizeof(LYVertex)*2 
		+ m_numVertices*sizeof(uint)*2 
		+ m_numGridCells*sizeof(uint)*2
		+ m_numVertices*sizeof(float4);

	(classSize > 1024*1024) ? 
		printf("This mesh requires: %d MB\n", classSize / (1024*1024)) :
	(classSize > 1024) ? printf("This mesh requires: %d Kb\n", classSize / (1024)):
		printf("This mesh requires: %d Bytes\n", classSize);

	LYCudaHelper::printMemInfo();
	LYCudaHelper::allocateArray((void **)&m_sorted_points, m_numVertices*sizeof(LYVertex));
	LYCudaHelper::allocateArray((void **)&m_src_points, m_numVertices*sizeof(LYVertex));

	LYCudaHelper::allocateArray((void **)&m_point_force, m_numVertices*sizeof(float4));
	cudaMemset(m_point_force, 0, m_numVertices*sizeof(float4));

	LYCudaHelper::allocateArray((void **)&m_collisionPoints, m_numToolVertices*sizeof(glm::vec4));

	LYCudaHelper::allocateArray((void **)&m_pointHash, m_numVertices*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_pointGridIndex, m_numVertices*sizeof(uint));

	LYCudaHelper::allocateArray((void **)&m_cellStart, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_cellEnd, m_numGridCells*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&m_dParams, sizeof(SimParams));
	LYCudaHelper::allocateArray((void **)&m_dSinking, sizeof(float4));
	LYCudaHelper::allocateArray((void **)&d_CollectionCellStart, m_maxNumCollectionElements*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&d_CollectionVertices,  m_maxNumCollectionElements*sizeof(uint));
	LYCudaHelper::printMemInfo();
	
	LYCudaHelper::registerGLBufferObject(m_srcVBO, &m_vboRes);

	LYVertex *dPos = (LYVertex *) LYCudaHelper::mapGLBufferObject(&m_vboRes);

	checkCudaErrors(cudaMemcpy(m_src_points, dPos, m_numVertices*sizeof(LYVertex), cudaMemcpyDeviceToDevice));

	LYCudaHelper::unmapGLBufferObject(m_vboRes);

	collisionCheckArgs.sortedPos				= m_sorted_points;
	collisionCheckArgs.toolPos					= m_collisionPoints;
	collisionCheckArgs.forceVector				= m_forceFeedback;
	collisionCheckArgs.force					= m_point_force;
	collisionCheckArgs.gridParticleIndex		= m_pointGridIndex;
	collisionCheckArgs.cellStart				= m_cellStart;
	collisionCheckArgs.cellEnd					= m_cellEnd;
	collisionCheckArgs.dev_params				= m_dParams;
	collisionCheckArgs.numVertices				= m_numVertices;
	collisionCheckArgs.numToolVertices			= m_numToolVertices;
	collisionCheckArgs.voxSize					= 1.0f/gridSize.x;
	collisionCheckArgs.maxSearchRange			= maxSearchRange;
	collisionCheckArgs.maxSearchRangeSq			= maxSearchRangeSq;
	collisionCheckArgs.maxNumCollectionElements = m_maxNumCollectionElements;

	LYCudaHelper::allocateArray((void **)&collisionCheckArgs.totalVertices_2Step, 1*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&collisionCheckArgs.collectionCellStart, m_maxNumCollectionElements*sizeof(uint));
	LYCudaHelper::allocateArray((void **)&collisionCheckArgs.collectionVertices, m_maxNumCollectionElements*sizeof(uint));

	checkCudaErrors(cudaMemset(collisionCheckArgs.collectionVertices, 0, m_maxNumCollectionElements*sizeof(uint)));

	setParameters(&m_params);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));

	cudaDeviceSynchronize();

	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	setParameters(&m_params);

	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));

	dPos = (LYVertex *) LYCudaHelper::mapGLBufferObject(&m_vboRes);

	cudaMemcpy(m_src_points, dPos, numVertices*sizeof(LYVertex), cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize();

	StopWatchInterface *spatialHashTimer=nullptr;
	sdkCreateTimer(&spatialHashTimer);
	sdkStartTimer(&spatialHashTimer);
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
	sdkStopTimer(&spatialHashTimer);

	std::ofstream myfile("./performance/spatialHash.txt", std::ios::app);
	myfile << "GPU; " << modelName << "; " << this->m_numVertices << "; " << sdkGetAverageTimerValue(&spatialHashTimer) << std::endl;
	myfile.close();

	LYCudaHelper::unmapGLBufferObject(m_vboRes);

}


LYSpatialHash::~LYSpatialHash(void)
{
	delete m_hCellEnd;
	delete m_hCellStart;
	delete m_hParams;	

	std::cout << "Freeing Hash info ->" << std::endl;
	LYCudaHelper::freeArray(m_sorted_points);
	std::cout << "m_sorted_points" << std::endl;
	LYCudaHelper::freeArray(m_src_points);
	std::cout << "m_src_points" << std::endl;

	LYCudaHelper::freeArray(m_point_force);
	std::cout << "m_point_force" << std::endl;

	LYCudaHelper::freeArray(m_collisionPoints);
	std::cout << "m_collisionPoints" << std::endl;

	LYCudaHelper::freeArray(m_pointHash);
	std::cout << "m_pointHash" << std::endl;
	LYCudaHelper::freeArray(m_pointGridIndex);
	std::cout << "m_pointGridIndex" << std::endl;

	LYCudaHelper::freeArray(m_cellStart);
	std::cout << "m_cellStart" << std::endl;
	LYCudaHelper::freeArray(m_cellEnd);
	std::cout << "m_cellEnd" << std::endl;
	LYCudaHelper::freeArray(m_dParams);
	std::cout << "m_dParams" << std::endl;
	LYCudaHelper::freeArray(d_CollectionCellStart);
	std::cout << "d_CollectionCellStart" << std::endl;
	LYCudaHelper::freeArray(d_CollectionVertices);
	std::cout << "d_CollectionVertices" << std::endl;

	LYCudaHelper::freeArray(collisionCheckArgs.totalVertices_2Step);
	std::cout << "collisionCheckArgs.totalVertices_2Step" << std::endl;
	LYCudaHelper::freeArray(collisionCheckArgs.collectionCellStart);
	std::cout << "collisionCheckArgs.collectionCellStart" << std::endl;
	LYCudaHelper::freeArray(collisionCheckArgs.collectionVertices);
	std::cout << "collisionCheckArgs.collectionVertices" << std::endl;

	LYCudaHelper::freeArray(m_dSinking);
	std::cout << "m_dSinking" << std::endl;
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
	if (r > this->maxR) return;
	m_params.R = r;
	this->m_hParams->R = r;
	neighborhoodRadius = r;
	collisionCheckArgs.R = r;
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));
	setParameters(m_hParams);
	printf("NSize = %d\n", (int) glm::round(r/m_hParams->cellSize.x));
}

void	LYSpatialHash::update()
{
	// update constants
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	LYCudaHelper::copyArrayToDevice(m_dParams, m_hParams, 0, sizeof(SimParams));
	if (m_dirtyPos && m_updatePositions) {
		LYVertex *dPos = (LYVertex *) LYCudaHelper::mapGLBufferObject(&m_vboRes);
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

		if (m_updatePositions)
		{
			updatePositions(
				m_sorted_points,
				m_point_force,
				dPos,
				m_numVertices);

			//updateDensities(
			//	m_sorted_points,
			//	dPos,
			//	m_pointGridIndex,
			//	m_cellStart,
			//	m_cellEnd,
			//	m_dParams,
			//	m_numVertices);

			//updateProperties(
			//	m_sorted_points,
			//	dPos,
			//	m_pointGridIndex,
			//	m_cellStart,
			//	m_cellEnd,
			//	m_dParams,
			//	m_numVertices);
		}

		LYCudaHelper::unmapGLBufferObject(m_vboRes);
		m_dirtyPos = false;
	}
}


float LYSpatialHash::calculateCollisions( float3 pos )
{
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);
	collisionCheckArgs.pos = pos;
	collisionCheckArgs.collisionCheckType = this->m_collisionCheckType;
	collisionCheckArgs.forceVector = this->m_forceFeedback;

	collisionCheck(collisionCheckArgs);
	LYCudaHelper::copyArrayFromDevice(m_hParams, m_dParams, 0, sizeof(SimParams));

	return m_hParams->w_tot;
}

float3 LYSpatialHash::calculateFeedbackUpdateProxy( Collider *pos )
{
	switch (renderingMethod){
	case IMPLICIT_SURFACE:
		return implicitSurfaceApproach(pos);
		break;
	case SINKING:
		return sinkingApproach(pos);
		break;
	}

	m_forceFeedback = make_float4(0.0f);
	return make_float3(m_forceFeedback);
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

void LYSpatialHash::toggleUpdatePositions()
{
	m_updatePositions = !m_updatePositions;
	std::cout << "Updating positions: ";
	std::cout << (m_updatePositions ? std::string("Yes") : std::string("No"));
	std::cout << std::endl;
}

void LYSpatialHash::resetPositions()
{
	LYVertex *dPos = (LYVertex *) LYCudaHelper::mapGLBufferObject(&m_vboRes);
	checkCudaErrors(cudaMemcpy(dPos, m_src_points, m_numVertices*sizeof(LYVertex), cudaMemcpyDeviceToDevice));
	LYCudaHelper::unmapGLBufferObject(m_vboRes);
}

void LYSpatialHash::toggleCollisionCheckType()
{
	m_collisionCheckType = (CollisionCheckType) 
		( (m_collisionCheckType+1) % CollisionCheckType::NUM_TYPES);
}

void LYSpatialHash::toggleRenderingMethod()
{
	renderingMethod = (HapticRenderingMethods) 
		( (renderingMethod+1) % HapticRenderingMethods::NUM_METHODS);
}

const std::string LYSpatialHash::getCollisionCheckString() const
{
	switch (m_collisionCheckType)
	{
		case DYNAMIC:
			return std::string("Dynamic");
			break;
		case NAIVE:
			return std::string("Naive");
			break;
		case TWO_STEP:
			return std::string("2 Step");
			break;
		case BASIC:
			return std::string("Bruteforce");
			break;
	}
	return std::string("CollisionCheckType doesn't exist");
}

float3 LYSpatialHash::implicitSurfaceApproach(Collider * pos)
{
	float3 colliderPos = pos->hapticPosition;
	float3 Pseed = make_float3(0.0f);
	float3 dP = make_float3(0.0f);
	float3 Ax, Nx;
	float Fx;

	if (!m_touched){
		calculateCollisions(colliderPos);
		Ax = m_hParams->Ax/m_hParams->w_tot;
		float nLength = length(m_hParams->Nx);
		Nx = m_hParams->Nx/nLength;

		Fx = dot(Nx, colliderPos - Ax);

		if (Fx < 0.0f){
			m_touched = true;
			Pseed = Ax;
			pos->scpPosition = Pseed;
			pos->surfaceTgPlane = Nx;
			m_forceFeedback = make_float4((Pseed - colliderPos), 0.0f);
			if (length(m_forceFeedback) > 0.3f) m_dirtyPos = true;
			return make_float3(m_forceFeedback);
		} else {
			pos->scpPosition = colliderPos;
			m_touched = false;
		}
	} else {
		float dist = dot(colliderPos - pos->scpPosition, pos->surfaceTgPlane);
		if (dist <= 0)
		{
			Pseed = colliderPos - dist*pos->surfaceTgPlane;
			int err_iterations(0);
			int iterations(0);
			do{
				while(calculateCollisions(Pseed) < 0.001f && ++err_iterations < 4);
				Ax = m_hParams->Ax/m_hParams->w_tot;
				Nx = m_hParams->Nx;
				float dNx = length(Nx);
				Nx /= dNx;
				Fx = dot(Nx, Pseed - Ax);
				dP.x = -Fx * Nx.x;
				dP.y = -Fx * Nx.y;
				dP.z = -Fx * Nx.z;
				dP = dP/fmaxf(dNx, 0.01f);
				dP *= 0.01f;
				Pseed += dP;
			} while (length(dP) > 0.001f && ++iterations < 4);
			pos->scpPosition = Pseed;
			pos->surfaceTgPlane = Nx;
			m_forceFeedback = make_float4((Pseed - colliderPos), 0.0f);
			if (length(m_forceFeedback) > 0.03f) m_dirtyPos = true;

			return make_float3(m_forceFeedback);
		} else {
			printf("Contact lost!  ");
			Ax = pos->scpPosition;
			Nx = pos->surfaceTgPlane;
			Fx = dist;
			printf("Ax = (%f, %f, %f)  ", Ax.x, Ax.y, Ax.z);
			printf("Nx = (%f, %f, %f)  ", Nx.x, Nx.y, Nx.z);
			printf("Fx = %f\n", Fx);
			m_touched = false;
			pos->scpPosition = colliderPos;
		}
	}
	return make_float3(0.0f);
}

float3 LYSpatialHash::sinkingApproach(Collider * pos)
{
	float k = 0.001f;
	float k_h = 0.1f;
	float k_t = 0.001f;
	float gamma = 0.5f;
	float R = collisionCheckArgs.R;
	float eps = gamma * R;

	float3 force = make_float3(0.0f);
	float3 Vn = calculateOvershoot(pos->scpPosition);
	float lVn = length(Vn);
	if (lVn < eps) pos->scpPosition = pos->hapticPosition;
	else pos->scpPosition += k*Vn;
	float3 Vh = pos->hapticPosition - pos->scpPosition;

	float collisionCheck = dot(Vn, Vh);
	printf("Sinking = %f\n", collisionCheck);
	if (collisionCheck > EPS) pos->scpPosition += k_h*Vh;
	else {
		pos->scpPosition += k_t*cross(Vn, Vh);
		float lVh = length(Vh);
		if (lVh >= R*0.5f) force = (lVh - R*0.5f) * (Vh/lVh);
	}
	pos->surfaceTgPlane = Vn/lVn;
	float spring = -1.0f;
	force *= spring;

	return force;
}

float3 LYSpatialHash::calculateOvershoot(float3 scpPosition)
{
	m_hParams->w_tot = 0.0f;
	m_hParams->Ax = make_float3(0.0f);
	m_hParams->Nx = make_float3(0.0f);

	OvershootArgs args;
	args.sortedPos = m_sorted_points;
	args.influenceRadius = collisionCheckArgs.R;
	args.numVertices = m_numVertices;
	args.sinking = m_dSinking;
	args.pos = scpPosition;
	float4 Vn = make_float4(0.0f);
	LYCudaHelper::copyArrayToDevice((void **)m_dSinking, &Vn, 0, sizeof(float4));
	computeOvershoot(args);
	LYCudaHelper::copyArrayFromDevice((void**)&Vn, m_dSinking, 0, sizeof(float4));
	return make_float3(Vn);
}

const std::string LYSpatialHash::getMethodString() const
{
	switch(renderingMethod)
	{
	case LYSpaceHandler::IMPLICIT_SURFACE:
		return std::string("Implicit");
		break;
	case LYSpaceHandler::SINKING:
		return std::string("Sinking");
		break;
	}
	return std::string("No method detected!");
}
