#include "LYHapticDevice.h"

#include <iostream>
using namespace std;

LYHapticDevice::LYHapticDevice(LYSpaceHandler *sh)
{

	m_spaceHandler = sh;

	m_deviceType = LYHapticInterface::HAPTIC_DEVICE;

	m_collider =	LYVertex();
	m_speed =		0.001f;
	m_size	=		0.05f;

	std::vector<LYVertex> Vertices;
	std::vector<unsigned int> Indices;
	LYVertex v(make_float3(0.0, 0.0, 0.0),
		make_float2(0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(255, 255, 255),
		int(1));

	Vertices.push_back(v);

	Indices.push_back(0);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * 1, &Vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &ib);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 1, &Indices[0], GL_STATIC_DRAW);

	ghHD = HD_INVALID_HANDLE;

	pState = new LYHapticState();

	initHD();
	COLLISION_FORCEFEEDBACK = true;
}

LYHapticDevice::~LYHapticDevice(void)
{
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ib);
}

float3 LYHapticDevice::getPosition() const{
	return m_collider.m_pos;
}
void LYHapticDevice::setPosition(float3 pos) {
	m_position = pos;
	m_collider.m_pos = pos;
	std::vector<LYVertex> Vertices;
	Vertices.push_back(m_collider);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * 1, &Vertices[0], GL_STATIC_DRAW);
}
float3 LYHapticDevice::getForceFeedback(float3 pos) const{
	return m_spaceHandler->getForceFeedback(m_collider.m_pos);
}

float LYHapticDevice::getSpeed() const 
{
	return m_speed;
}

float LYHapticDevice::getSize() const 
{
	return m_size;
}

uint LYHapticDevice::getIB() const
{
	return ib;
}

uint LYHapticDevice::getVBO() const
{
	return vbo;
}

void LYHapticDevice::setForces(bool c)
{
	COLLISION_FORCEFEEDBACK = c;
}

void LYHapticDevice::loadDevices()
{
	HDErrorInfo error;

	ghHD = hdInitDevice(HD_DEFAULT_DEVICE);
	if (HD_DEVICE_ERROR(error = hdGetError()))
	{
		//no default device - test another name for the device.
		ghHD = hdInitDevice("Omni1");
		if (HD_DEVICE_ERROR(error = hdGetError()))
		{	
		}
	}
}

void LYHapticDevice::initHD()
{	
	loadDevices();

	hdMakeCurrentDevice(ghHD);
	hdEnable(HD_FORCE_OUTPUT);

	hUpdateDeviceCallback = hdScheduleAsynchronous(
		touchMesh, this, HD_MAX_SCHEDULER_PRIORITY);

	hdStartScheduler();
}

void LYHapticDevice::touchTool()
{
	/* Obtain a thread-safe copy of the current haptic display state. */
	hdScheduleSynchronous(copyHapticDisplayState, pState,
		HD_DEFAULT_SCHEDULER_PRIORITY);

	int currentButtons;
	hduVector3Dd position;
	hduVector3Dd force( 0,0,0 );
	HDdouble forceClamp;
	HDErrorInfo error;

	hdBeginFrame(ghHD);

	hdGetIntegerv(HD_CURRENT_BUTTONS, &currentButtons);

	if(COLLISION_FORCEFEEDBACK)
	{
		//calculate the force
		float pos[3] = {pState->position[0],pState->position[1],pState->position[2]};
		this->setPosition(make_float3(pState->position[0] * 0.01f, -pState->position[2] * 0.01f ,pState->position[1] *0.01f));
		float f[3]={0,0,0};
		float forceScale = 0.01f;
		float3 _force = this->getForceFeedback(m_collider.m_pos);
		//return the force to the haptic device
		force[0] = _force.x * forceScale;
		force[1] = _force.y * forceScale;
		force[2] = _force.z * forceScale;

		hdSetDoublev(HD_CURRENT_FORCE, force);
	}
	else
	{
		force[0] = 0;
		force[1] = 0;
		force[2] = 0;
		hdSetDoublev(HD_CURRENT_FORCE, force);
	}
	hdEndFrame(ghHD);
}

bool LYHapticDevice::toggleForces()
{
	COLLISION_FORCEFEEDBACK = !COLLISION_FORCEFEEDBACK;

	return COLLISION_FORCEFEEDBACK;
}

void LYHapticDevice::setSpaceHandler( LYSpaceHandler *sh )
{
	m_spaceHandler = sh;
}

/******************** OPEN HAPTICS CALBACK FUNCTIONS **************/

HDCallbackCode HDCALLBACK copyHapticDisplayState(void *pUserData)
{
	LYHapticState *pState = (LYHapticState *) pUserData;

	hdGetDoublev(HD_CURRENT_POSITION, pState->position);
	hdGetDoublev(HD_CURRENT_TRANSFORM, pState->transform);
	hdGetDoublev(HD_CURRENT_VELOCITY, pState->velocity);
	hdGetIntegerv(HD_UPDATE_RATE, &pState->UpdateRate);

	return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK touchMesh(void *pUserData)
{
	LYHapticDevice* haptic = (LYHapticDevice*) pUserData;

	haptic->touchTool();

	return HD_CALLBACK_CONTINUE;
}

/****************************************************************/