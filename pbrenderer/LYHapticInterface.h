#pragma once
#include "defines.h"
#include "LYVertex.h"
#include "LYSpaceHandler.h"
#include "vector_functions.h"
#include "vector_types.h"
#include <glm\glm.hpp>

class LYHapticInterface
{
public:
	typedef enum {
		KEYBOARD_DEVICE = 0,
		HAPTIC_DEVICE
	}LYDEVICE_TYPE;

	virtual float3 getPosition() const = 0;
	virtual void setPosition(float3 pos) = 0;
	virtual void setSpaceHandler(LYSpaceHandler *sh) = 0;
	virtual float3 getForceFeedback(float3 pos) const = 0;
	virtual float getSpeed() const = 0;
	virtual float getSize() const = 0;
	virtual float getDamping() const = 0;
	virtual void setDamping(float d) = 0;
	virtual float getForceScale() const = 0;
	virtual void setForceScale(float d) = 0;

	virtual LYDEVICE_TYPE getDeviceType() const = 0;
	virtual uint getVBO()	const = 0;
	virtual uint getIB()	const	= 0;

protected:
	LYVertex	m_collider;
	float3		m_position;
	float3		m_direction;
	float		m_speed;
	float		m_size;
	float		m_damping;
	float		m_forceScale;

	LYDEVICE_TYPE	m_deviceType;

	uint vbo;
	uint ib;
};
