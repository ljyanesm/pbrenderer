#pragma once
#include "defines.h"
#include "Collider.h"
#include "LYPLYLoader.h"
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

	virtual void	setPosition(float3 pos);
	virtual void	setSpaceHandler( LYSpaceHandler *sh );
	virtual void	setWorkspaceScale(float3 dim);
	virtual void	setRelativePosition(float3 pos);
	virtual void	setCameraMatrix(glm::mat4 t);
	virtual void	setModelMatrix(glm::mat4 t);
	virtual void	setSize(float r);
	virtual bool	toggleForces(bool p = true);
	virtual void	setTimer(StopWatchInterface *timer);
	virtual void	setSpeed(float s);

	virtual float3	calculateFeedbackUpdateProxy();
	virtual void	pause();
	virtual void	start();
	virtual void	pause(bool p);


	virtual float3			getPosition() const;
	virtual float3			getSurfacePosition() const;
	virtual float3			getSurfaceNormal() const;
	virtual float3			getForceVector() const;
	virtual glm::mat4		getHIPMatrix() const;
	virtual glm::mat4		getProxyMatrix() const;
	virtual float			getSpeed() const;
	virtual float			getSize()	const;
	virtual LYDEVICE_TYPE	getDeviceType() const;
	virtual uint			getVBO() const;
	virtual uint			getIB() const;

	virtual LYMesh*			getHIPObject() const;
	virtual LYMesh*			getProxyObject() const;
	virtual uint			getProxyVBO() const;
	virtual uint			getProxyIB() const;
	virtual uint			getHIPVBO() const;
	virtual uint			getHIPIB() const;
	virtual size_t			getProxyNumVertices() const;
	virtual size_t			getHIPNumVertices() const;
	virtual bool			isEnabled() const;

protected:
	bool COLLISION_FORCEFEEDBACK;
	
	Collider		m_collider;
	float3			m_forceVector;
	float3			m_workspaceScale;
	float3			m_relativePosition;
	float			m_speed;
	float			m_size;

	LYMesh			*m_HIPObject;
	LYMesh			*m_ProxyObject;

	uint			vbo;
	uint			ib;

	glm::mat4		m_ViewMatrix;
	glm::mat4		m_ModelMatrix;
	glm::mat4		m_HIPMatrix;
	glm::mat4		m_ProxyMatrix;

	LYDEVICE_TYPE	m_deviceType;
	bool			bPause;


	StopWatchInterface *m_timer;
	LYSpaceHandler *m_spaceHandler;
};
