#pragma once
#include "defines.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector>

#include "LYVertex.h"
#include "LYHapticInterface.h"

class LYHapticKeyboard :
	public LYHapticInterface
{
private:
	LYSpaceHandler *m_spaceHandler;
public:
	LYHapticKeyboard(LYSpaceHandler *sh);
	~LYHapticKeyboard(void);

	void setSpaceHandler(LYSpaceHandler *sh);
	float3 getPosition() const;
	void setPosition(float3 pos);
	float3 getForceFeedback(float3 pos) const;
	float getSpeed() const;
	float getSize()	const;

	LYDEVICE_TYPE getDeviceType() const { return m_deviceType; }
	uint getVBO() const;
	uint getIB() const;
};