#pragma once
#include "defines.h"
#include "LYMesh.h"
#include "LYSpaceHandler.h"
#include "vector_functions.h"
#include "vector_types.h"
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>

class LYHapticInterface
{
public:
	typedef enum {
		KEYBOARD_DEVICE = 0,
		HAPTIC_DEVICE
	}LYDEVICE_TYPE;

	virtual void setPosition(float3 pos) = 0;
	virtual void setSpaceHandler(LYSpaceHandler *sh) = 0;
	virtual void setWorkspaceScale(float3 dim) = 0;
	virtual void setCameraMatrix(glm::mat4 t) = 0;
	virtual float3 calculateFeedbackUpdateProxy() = 0;
	virtual void setSize(float) = 0;
	virtual void setRelativePosition(float3 pos) = 0;
	virtual void pause(bool pause) = 0;
	virtual bool toggleForces() = 0;
	virtual void	setTimer(StopWatchInterface *timer) = 0;


	virtual glm::mat4 getHIPMatrix() const = 0;
	virtual glm::mat4 getProxyMatrix() const = 0;
	virtual float3 getPosition() const = 0;
	virtual float getSpeed() const = 0;
	virtual float getSize() const = 0;
	virtual LYDEVICE_TYPE getDeviceType() const = 0;
	virtual uint getVBO()	const = 0;
	virtual uint getIB()	const	= 0;

	virtual LYMesh* getHIPObject() const = 0;
	virtual LYMesh* getProxyObject() const = 0;
	virtual uint getProxyVBO() const = 0;
	virtual uint getProxyIB() const = 0;
	virtual uint getProxyNumVertices() const = 0;
	virtual uint getHIPVBO() const = 0;
	virtual uint getHIPIB() const = 0;
	virtual uint getHIPNumVertices() const = 0;


protected:
	LYVertex	m_collider;
	float3		m_workspaceScale;
	float3		m_relativePosition;
	float		m_speed;
	float		m_size;

	LYMesh		*m_HIPObject;
	LYMesh		*m_ProxyObject;

	glm::mat4	m_CameraMatrix;
	glm::mat4	m_HIPMatrix;
	glm::mat4	m_ProxyMatrix;

	LYDEVICE_TYPE	m_deviceType;

	uint vbo;
	uint ib;

	bool bPause;

	StopWatchInterface *m_timer;
};
