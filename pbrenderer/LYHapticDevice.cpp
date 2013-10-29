#include "LYHapticDevice.h"

#include <iostream>
using namespace std;

LYHapticDevice::LYHapticDevice(LYSpaceHandler *sh)
{
	m_spaceHandler		= sh;
	m_deviceType		= LYHapticInterface::HAPTIC_DEVICE;
	m_collider			= LYVertex();
	m_speed				= 0.001f;
	m_size				= 0.03f;
	m_hapticWorkspace	= make_float3(0.3f);

	m_modelMatrix = glm::mat4();

	LYVertex proxy;
	proxy.m_pos = m_collider.m_normal;

	std::vector<LYVertex> Vertices;
	std::vector<unsigned int> Indices;
	LYVertex v(make_float3(0.0, 0.0, 0.0),
		make_float2(0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(255, 255, 255),
		int(1));

	Vertices.push_back(v);
	Vertices.push_back(v);

	Indices.push_back(0);
	Indices.push_back(1);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * 2, &Vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &ib);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 2, &Indices[0], GL_STATIC_DRAW);

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
	// Multiply 

	m_position = pos;
	m_collider.m_pos = pos;
	std::vector<LYVertex> Vertices;
	Vertices.push_back(m_collider);
	
	LYVertex proxy;
	proxy.m_color = make_float3(1.0f, 0.0f, 0.0f);
	proxy.m_pos = m_collider.m_normal;
	Vertices.push_back(proxy);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * 2, &Vertices[0], GL_STATIC_DRAW);
}

float3 LYHapticDevice::getForceFeedback(float3 pos) const{
	return m_spaceHandler->getForceFeedback(m_collider.m_pos);
}

float3 LYHapticDevice::calculateFeedbackUpdateProxy()
{
	return m_spaceHandler->calculateFeedbackUpdateProxy(&m_collider);
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
		HD_MAX_SCHEDULER_PRIORITY);
	static float3 oldForce = float3();
	int currentButtons;
	hduVector3Dd position;
	hduVector3Dd force( 0,0,0 );

	hdBeginFrame(ghHD);

	hdGetIntegerv(HD_CURRENT_BUTTONS, &currentButtons);

	if(COLLISION_FORCEFEEDBACK)
	{
		float3 pos = make_float3((float) pState->position[0], (float) pState->position[1], (float) pState->position[2]);
		pos.x *= m_hapticWorkspace.x;
		pos.y *= m_hapticWorkspace.y;
		pos.z *= m_hapticWorkspace.z;
		// Find current model
		// int model = this->findCurrentModel(pos);
		// this->setPosition(pos, m_mesh->at(model)->getModelMatrix());
		this->setPosition(pos);
		float f[3]={0,0,0};
		float damping = 0.2f;
		float forceScale = 3.0f;
		float3 _force = this->calculateFeedbackUpdateProxy() * forceScale;
		force[0] = _force.x;
		force[1] = _force.y;
		force[2] = _force.z;
		//oldForce += (_force - oldForce) * damping;
		//force[0] = oldForce.x;
		//force[1] = oldForce.y;
		//force[2] = oldForce.z;

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

void LYHapticDevice::setSize( float r )
{
	m_size = r;
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