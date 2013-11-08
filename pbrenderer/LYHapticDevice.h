#ifndef _OPEN_HAPTICS_H
#define _OPEN_HAPTICS_H

#include "defines.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector>

#include "helper_math.h"
#include <HD/hd.h>
#include <HDU/hdu.h>
#include <HDU/hduVector.h>
#include <HLU/hlu.h>

#include "LYSpaceHandler.h"
#include "LYHapticInterface.h"
#include "LYHapticState.h"

class LYHapticDevice : public LYHapticInterface
{
private:
	HHD ghHD;
	HDSchedulerHandle hUpdateDeviceCallback;

	LYHapticState* pState;
	LYSpaceHandler *m_spaceHandler;

	LYVertex collisionPosition;
	LYVertex displayPosition;
public:
	bool COLLISION_FORCEFEEDBACK;
	
	LYHapticDevice(LYSpaceHandler *sh);
	~LYHapticDevice();

	void setPosition(float3 pos);
	void setSize(float r);
	float3	calculateFeedbackUpdateProxy();
	void setWorkspaceScale(float3 dim) { m_workspaceScale = dim; }
	void setRelativePosition(float3 pos) { m_relativePosition = pos; }
	void setCameraMatrix(glm::mat4 t) { m_CameraMatrix = t; }
	void pause(bool p) 
	{ 
		bPause = p;
		if (p != bPause) 
			(p != true) ? hdStopScheduler() : hdStartScheduler(); 
	}

	glm::mat4 getHIPMatrix() const { return m_HIPMatrix; }
	glm::mat4 getProxyMatrix() const { return m_ProxyMatrix; }
	float3 getPosition() const;
	float getSpeed() const;
	float getSize() const;	
	LYDEVICE_TYPE getDeviceType() const { return m_deviceType; }
	uint getVBO()	const;
	uint getIB()	const;

	uint getProxyVBO() const { return m_ProxyObject->getVBO(); }
	uint getProxyIB() const { return m_ProxyObject->getIB(); }
	uint getProxyNumVertices() const { return m_ProxyObject->getNumVertices(); }
	uint getHIPVBO() const { return m_HIPObject->getVBO(); }
	uint getHIPIB() const { return m_HIPObject->getIB(); }
	uint getHIPNumVertices() const { return m_HIPObject->getNumVertices(); }
	virtual LYMesh* getHIPObject() const {return m_HIPObject;}
	virtual LYMesh* getProxyObject() const {return m_ProxyObject;}

	void setSpaceHandler(LYSpaceHandler *sh);

	void obtainHapticState();
	void initHD();
	void loadDevices();
	void touchTool();
	void setForces(bool c);
	bool toggleForces();
	void drawEndPoints();

	void	setTimer(StopWatchInterface *timer) { m_timer = timer; }

};

//OpenHaptics callback functions
HDCallbackCode HDCALLBACK touchMesh(void *pUserData);
HDCallbackCode HDCALLBACK copyHapticDisplayState(void *pUserData);
	
#endif _OPEN_HAPTICS_H