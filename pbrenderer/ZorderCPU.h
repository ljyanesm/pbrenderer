#pragma once
#include "defines.h"
#include "LYMesh.h"
#include "LYSpaceHandler.h"

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <device_functions.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <algorithm>
class ZorderCPU : public LYSpaceHandler
{
public:
	void update() {}
	void clear() {}
	void dump() {}

	void setDeviceVertices(LYVertex *hostVertices) {}

	float3	calculateFeedbackUpdateProxy(Collider *pos) { return make_float3(0.0f);}
	float	calculateCollisions(float3 pos) { return 0.0f;}
	void	setInfluenceRadius(float r) {}
	void	toggleUpdatePositions() {}
	void	resetPositions() {}

	ZorderCPU(LYMesh *mesh);
	~ZorderCPU(void);

private:
	void	updateStructure();
	bool	comparePoints(const LYVertex& p1, const LYVertex& p2);

	std::vector<LYVertex>	sortedPoints;
	std::vector<uint>		zIndeces;
	float4		m_forceFeedback;

	bool		m_updatePositions;
	bool		m_touched;
	bool		m_dirtyPos;

	uint		m_srcVBO;
	size_t		m_numVertices;

};

