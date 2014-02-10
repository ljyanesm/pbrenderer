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

#include "LYPLYLoader.h"
#include "LYSpaceHandler.h"
#include "LYHapticInterface.h"
#include "LYHapticState.h"

//OpenHaptics callback functions
HDCallbackCode HDCALLBACK touchMesh(void *pUserData);
HDCallbackCode HDCALLBACK copyHapticDisplayState(void *pUserData);
class LYHapticDevice : public LYHapticInterface
{
private:
	HHD ghHD;
	HDSchedulerHandle hUpdateDeviceCallback;

	LYHapticState* pState;
public:
	LYHapticDevice(LYSpaceHandler *sh, LYMesh *p, LYMesh *h);
	~LYHapticDevice();


	void obtainHapticState();
	void initHD();
	void loadDevices();
	void touchTool();
};
#endif _OPEN_HAPTICS_H