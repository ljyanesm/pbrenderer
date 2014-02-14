#pragma once
#include "LYVertex.h"
#include "LYHapticInterface.h"
class IOManager
{
	float4 inputPosition;		// This is the input value from the device

	//float4 surfacePosition;		// This value is to be set at a 1kHz (Haptic) rate and read at about 60fps (Graphics)
	//float4 surfaceNormal;		// This value is to be set at a 1kHz (Haptic) rate and read at about 60fps (Graphics)
	//float4 forceVector;			// Force vector for rendering purposes

	float4 wsPosition;			// Workspace center position
	float4 wsDimension;			// Workspace total dimensions
	float4 wsWorkingDimension;	// Workspace working dimensions (outside this range it moves the workspace)

	LYHapticInterface *_device;	// This will not be modifiable just to be read from
	
	StopWatchInterface *timer;

public:
	IOManager(void);
	IOManager(LYHapticInterface *_d, float4 wsDim, float4 wsWD);
	~IOManager(void);

	const float4 getSurfacePosition() const { return make_float4(_device->getSurfacePosition(), 1.0f); }
	const float4 getSurfaceNormal() const { return make_float4(_device->getSurfaceNormal(), 0.0f); }
	const float4 getForceVector() const { return make_float4(_device->getForceVector(), 0.0f); }
	const glm::mat4 getSCPPositionMatrix() const { return _device->getProxyMatrix(); }

	LYHapticInterface *getDevice() { return _device; }

	void setTimer(StopWatchInterface *t) { timer = t; _device->setTimer(timer); }
};

