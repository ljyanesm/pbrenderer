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
public:
	bool COLLISION_FORCEFEEDBACK;
	
	LYHapticDevice(LYSpaceHandler *sh);
	~LYHapticDevice();
	float3 getPosition() const;
	void setPosition(float3 pos);
	
	float3 getForceFeedback(float3 pos) const;
	float3	calculateFeedbackUpdateProxy();

	float getSpeed() const;
	float getSize() const;
	void setSize(float r);
	
	LYDEVICE_TYPE getDeviceType() const { return m_deviceType; }
	uint getVBO()	const;
	uint getIB()	const;

	void setSpaceHandler(LYSpaceHandler *sh);

	void obtainHapticState();
	void initHD();
	void loadDevices();
	void touchTool();
	void setForces(bool c);
	bool toggleForces();
	void drawEndPoints();
};

//OpenHaptics callback functions
HDCallbackCode HDCALLBACK touchMesh(void *pUserData);
HDCallbackCode HDCALLBACK copyHapticDisplayState(void *pUserData);
	
#endif _OPEN_HAPTICS_H