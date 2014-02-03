#pragma once
#include "defines.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector>

#include "LYVertex.h"
#include "LYHapticInterface.h"
#include "LYPLYLoader.h"

class LYKeyboardDevice :
	public LYHapticInterface
{
private:
	LYSpaceHandler *m_spaceHandler;
	bool COLLISION_FORCEFEEDBACK;
public:
	LYKeyboardDevice(LYSpaceHandler *sh, LYMesh *proxyMesh, LYMesh *hipMesh);
	~LYKeyboardDevice(void);

	void pause() 
	{ 
		COLLISION_FORCEFEEDBACK = false;
	}
	void start()
	{
		COLLISION_FORCEFEEDBACK = true;
	}

	void setSpaceHandler(LYSpaceHandler *sh);
	void setWorkspaceScale(float3 dim) { m_workspaceScale = dim; }
	void setRelativePosition(float3 pos) { m_relativePosition = pos; }
	void setCameraMatrix(glm::mat4 t) { m_CameraMatrix = t; }
	void pause(bool p) { bPause = p; }
	bool toggleForces(bool p = true) 
	{
		COLLISION_FORCEFEEDBACK = p;
		return COLLISION_FORCEFEEDBACK;
	}

	glm::mat4 getHIPMatrix() const { return m_HIPMatrix; }
	glm::mat4 getProxyMatrix() const { return m_ProxyMatrix; }
	float3 getPosition() const;
	float3 *getHIP();
	void setPosition(float3 pos);

	float3	calculateFeedbackUpdateProxy();


	float getSpeed() const;
	float getSize()	const;
	void setSize(float r);

	LYDEVICE_TYPE getDeviceType() const { return m_deviceType; }
	uint getVBO() const;
	uint getIB() const;

	virtual LYMesh* getHIPObject() const {return m_HIPObject;}
	virtual LYMesh* getProxyObject() const {return m_ProxyObject;}

	uint getProxyVBO() const { return m_ProxyObject->getVBO(); }
	uint getProxyIB() const { return m_ProxyObject->getIB(); }
	size_t getProxyNumVertices() const { return m_ProxyObject->getNumVertices(); }
	uint getHIPVBO() const { return m_HIPObject->getVBO(); }
	uint getHIPIB() const { return m_HIPObject->getIB(); }
	size_t getHIPNumVertices() const { return m_HIPObject->getNumVertices(); }

	void	setTimer(StopWatchInterface *timer) { m_timer = timer; }
	bool isEnabled() const { return COLLISION_FORCEFEEDBACK; }


};
