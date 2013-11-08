#pragma once
#include "defines.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector>

#include "LYVertex.h"
#include "LYHapticInterface.h"

class LYKeyboardDevice :
	public LYHapticInterface
{
private:
	LYSpaceHandler *m_spaceHandler;
public:
	LYKeyboardDevice(LYSpaceHandler *sh);
	~LYKeyboardDevice(void);

	void setSpaceHandler(LYSpaceHandler *sh);
	void setWorkspaceScale(float3 dim) { m_workspaceScale = dim; }
	void setRelativePosition(float3 pos) { m_relativePosition = pos; }
	void setCameraMatrix(glm::mat4 t) { m_CameraMatrix = t; }
	void pause(bool p) { bPause = p; }
	bool toggleForces() { return false;}

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
	uint getProxyNumVertices() const { return m_ProxyObject->getNumVertices(); }
	uint getHIPVBO() const { return m_HIPObject->getVBO(); }
	uint getHIPIB() const { return m_HIPObject->getIB(); }
	uint getHIPNumVertices() const { return m_HIPObject->getNumVertices(); }

	void	setTimer(StopWatchInterface *timer) { m_timer = timer; }

};
