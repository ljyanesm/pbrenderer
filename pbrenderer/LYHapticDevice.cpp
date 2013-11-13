#include "LYHapticDevice.h"

#include <iostream>
using namespace std;

LYHapticDevice::LYHapticDevice(LYSpaceHandler *sh, LYMesh *proxyMesh, LYMesh *hipMesh)
{
	m_spaceHandler		= sh;
	m_deviceType		= LYHapticInterface::HAPTIC_DEVICE;
	m_collider			= LYVertex();
	m_speed				= 0.001f;
	m_size				= 0.03f;
	m_workspaceScale	= make_float3(0.3f);
	m_relativePosition	= make_float3(0.0f);

	m_ProxyObject	= proxyMesh;
	m_HIPObject		= hipMesh;

	m_HIPMatrix = glm::mat4();
	m_ProxyMatrix = glm::mat4();
	m_CameraMatrix = glm::mat4();

	LYVertex proxy;
	proxy.m_pos = m_collider.m_normal;

	std::vector<LYVertex> Vertices;
	std::vector<unsigned int> Indices;
	LYVertex hip(make_float3(0.0, 0.0, 0.0),
		make_float2(0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(0, 0, 0),
		int(0));
	LYVertex p(make_float3(0.0, 0.0, 0.0),
		make_float2(0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(255, 0, 0),
		int(1));

	Vertices.push_back(hip);
	Vertices.push_back(p);

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
	delete pState;
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ib);
}

float3 LYHapticDevice::getPosition() const{
	return m_collider.m_pos;
}

// Position for display!!
void LYHapticDevice::setPosition(float3 pos) {

	// Haptics only
	glm::mat4 inverseTransformation = glm::inverse(this->m_CameraMatrix);
	glm::vec3 p = glm::vec3(inverseTransformation * glm::vec4(pos.x, pos.y, pos.z, 1));
	m_collider.m_pos = make_float3(p.x, p.y, p.z);

	// Graphics only
	glm::mat4 finalTransformation = glm::translate(this->m_CameraMatrix, p);
	m_HIPMatrix = finalTransformation;
	p = glm::vec3(finalTransformation * glm::vec4(glm::vec3(0,0,0),1.0));
	finalTransformation = glm::translate(this->m_CameraMatrix, glm::vec3(m_collider.m_normal.x, m_collider.m_normal.y, m_collider.m_normal.z));
	m_ProxyMatrix = finalTransformation;
}

float3 LYHapticDevice::calculateFeedbackUpdateProxy()
{
	float3 force = m_spaceHandler->calculateFeedbackUpdateProxy(&m_collider);
	return force;
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
	//if (bPause) return;
	hdScheduleSynchronous(copyHapticDisplayState, pState,
		HD_MAX_SCHEDULER_PRIORITY);
	static float3 oldForce = float3();
	int currentButtons;
	hduVector3Dd position;
	hduVector3Dd force( 0,0,0 );
	sdkStartTimer(&m_timer);
	hdBeginFrame(ghHD);

	hdGetIntegerv(HD_CURRENT_BUTTONS, &currentButtons);

	if(COLLISION_FORCEFEEDBACK)
	{
		glm::vec3 pos = glm::vec3((float) pState->position[0], (float) pState->position[1], (float) pState->position[2]);
		pos.x *= m_workspaceScale.x;
		pos.y *= m_workspaceScale.y;
		pos.z *= m_workspaceScale.z;

		this->setPosition(make_float3(pos.x, pos.y, pos.z));
		float3 tmpForce = this->calculateFeedbackUpdateProxy();

		// Apply the current modelView rotation transformation (NO TRANSLATION) to the force vector
		glm::mat4 noTranslationMat = this->m_CameraMatrix;
		noTranslationMat[3][0] = 0; noTranslationMat[3][1] = 0; noTranslationMat[3][2] = 0; noTranslationMat[3][3] = 1;
		glm::vec3 p = glm::vec3(noTranslationMat * glm::vec4(tmpForce.x, tmpForce.y, tmpForce.z,1));
		tmpForce = make_float3(p.x, p.y, p.z);

		float f[3]={0,0,0};
		float damping = 0.2f;
		float forceScale = 3.0f;
		float3 _force = tmpForce * forceScale;
		force[0] = _force.x;
		force[1] = _force.y;
		force[2] = _force.z;
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
	sdkStopTimer(&m_timer);
}

bool LYHapticDevice::toggleForces(bool p)
{
	COLLISION_FORCEFEEDBACK = p;

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
	if(haptic->isEnabled()) haptic->touchTool();

	return HD_CALLBACK_CONTINUE;
}

/****************************************************************/