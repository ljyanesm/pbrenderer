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
	enum HapticRenderingMethods{
		IMPLICIT_SURFACE,
		SINKING,
		NUM_METHODS
	};

	enum SpaceHandlerType{
		GPU_SPATIAL_HASH,
		CPU_SPATIAL_HASH,
		CPU_Z_ORDER,
		NUM_TYPES
	};

	virtual ~LYSpaceHandler() {};
	virtual void update() = 0;
	virtual void clear() = 0;
	virtual void dump() = 0;

	const bool					getUpdatePos() const { return m_updatePositions; }
	virtual float3				calculateFeedbackUpdateProxy(Collider *pos) = 0;
	virtual float				calculateCollisions(float3 pos) = 0;
	virtual void				setInfluenceRadius(float r) = 0;
	virtual void				toggleUpdatePositions() = 0;
	virtual void				resetPositions() = 0;
	virtual const SpaceHandlerType	getType() const = 0;

protected:
	bool		m_updatePositions;
	bool		m_dirtyPos;
};
