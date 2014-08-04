#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <device_functions.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_gl_interop.h>
#include "glm/glm.hpp"
#include "defines.h"

#include "Collider.h"
#include "LYCudaHelper.cuh"
#include "LYSpaceHandler.h"
#include "LYSpatialHash.cuh"
#include "LYSpatialHash_kernel.cuh"

class LYSpatialHash : public LYSpaceHandler
{

public:
	LYSpatialHash(uint vbo, size_t numVertices, uint3 gridSize);
	~LYSpatialHash(void);

	void	update();
	void	clear();

	void	setVBO(uint vbo);
	void	setDeviceVertices(LYVertex *hostVertices);

	void	toggleCollisionCheck();

	void	setInfluenceRadius(float r);

	float	calculateCollisions(float3 pos);
	float3	calculateFeedbackUpdateProxy(Collider *pos);

	void	dump();

	void	resetPositions();
	void	toggleUpdatePositions();
	void	toggleCollisionCheckType();

	LYSpatialHash::SpaceHandlerType getType() { return LYSpatialHash::GPU_SPATIAL_HASH; }
private:

	const std::string getCollisionCheckString() const;

	cudaGraphicsResource* m_vboRes;

	float		neighborhoodRadius;		// Local neighborhood radius

	LYVertex*	m_src_points;			// Source points saved in the GPU
	LYVertex*	m_sorted_points;		// Sorted points saved in the GPU
	float4*		m_point_force;			// Forces applied to the points...

	glm::vec4*	m_collisionPoints;		// Collider 'tool' positions

	uint*		m_hCellStart;
	uint*		m_hCellEnd;

	uint*		m_cellStart;
	uint*		m_cellEnd;
	uint*		m_pointHash;
	uint*		m_pointGridIndex;
	uint		m_gridSortBits;

	uint*		d_CollectionCellStart;
	uint*		d_CollectionVertices;

	uint		m_srcVBO;
	size_t		m_numVertices;

	size_t		m_numToolVertices;		// Collider 'tool' vertices count

	uint3		m_gridSize;
	uint 		m_numGridCells;

	float4		m_forceFeedback;

	bool		m_updatePositions;
	bool		m_touched;
	bool		m_dirtyPos;


	CollisionCheckType	m_collisionCheckType;

	SimParams		m_params;
	SimParams*		m_hParams;
	SimParams*		m_dParams;

	ccConfiguration collisionCheckArgs;
	StopWatchInterface* collisionCheckTimer;

	const uint maxSearchRange;
	const uint maxSearchRangeSq;
	const uint m_maxNumCollectionElements;
};
