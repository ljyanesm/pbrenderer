#pragma once
#include "defines.h"
#include "LYMesh.h"
#include "LYSpaceHandler.h"
#include <vector>
#include <unordered_map>
#include <algorithm>

class CPUHashSTL :
	public LYSpaceHandler
{
public:
	typedef std::unordered_multimap<uint, LYVertex> LYVertexLookup;
	CPUHashSTL(LYMesh* m, int numBuckets);
	~CPUHashSTL(void);

	void update();
	void clear();
	void dump();

	float3				calculateFeedbackUpdateProxy(Collider *pos);
	float				calculateCollisions(float3 pos);
	void				setInfluenceRadius(float r);
	void				toggleUpdatePositions();
	void				resetPositions();
	const LYSpaceHandler::SpaceHandlerType	getType() const { return LYSpaceHandler::CPU_SPATIAL_HASH; }

private:
	const float wendlandWeight(const float& dist) const;

	const float3 translateRange(const float3& value, const float3& lMin, const float3& lMax, const float3 &rMin, const float3 &rMax) const;
	const uint hashFunction(const uint3& p) const;
	const uint3 getGridPos(const float3& p) const;
	float					influenceRadius;
	const float				cellSize;
	const int				nBuckets;

	LYMesh*					originalMesh;
	std::vector<LYVertex>	originalPoints;
	LYVertexLookup			lookup;

	float3					m_Ax;
	float3					m_Nx;
	float					m_w_tot;

	float4					m_forceFeedback;

	bool					m_touched;
	bool					m_dirtyPos;
};

