#pragma once

#include <helper_functions.h>

#include "LYVertex.h"
#include "Collider.h"

/*
	Abstract class for the space handling techniques classes to be implemented on the GPU or the CPU
*/

class LYSpaceHandler
{
public:
	virtual ~LYSpaceHandler() {};
	virtual void update() = 0;
	virtual void clear() = 0;
	virtual void dump() = 0;

	virtual void setDeviceVertices(LYVertex *hostVertices) = 0;

	virtual float3	calculateFeedbackUpdateProxy(Collider *pos) = 0;
	virtual float calculateCollisions(float3 pos) = 0;
	virtual void	setInfluenceRadius(float r) = 0;
	virtual void	toggleUpdatePositions() = 0;
	virtual void	resetPositions() = 0;
};
