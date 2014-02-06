#pragma once
#include "LYVertex.h"
#include "LYHapticInterface.h"
class IOManager
{
	float4 inputPosition;		// This is the input value from the device

	float4 surfacePosition;		// This value is to be set at a 1kHz (Haptic) rate and read at about 60fps (Graphics)
	float4 surfaceNormal;		// This value is to be set at a 1kHz (Haptic) rate and read at about 60fps (Graphics)

	float4 wsPosition;			// Workspace center position
	float4 wsDimension;			// Workspace total dimensions
	float4 wsWorkingDimension;	// Workspace working dimensions (outside this range it moves the workspace)

	LYHapticInterface *_device;	// This will not be modifiable just to be read from

public:
	IOManager(void);
	IOManager(LYHapticInterface *_d, float4 wsDim, float4 wsWD);
	~IOManager(void);

	float4 getSurfacePosition() { return surfacePosition; }
	float4 getSurfaceNormal() { return surfaceNormal; }
};

