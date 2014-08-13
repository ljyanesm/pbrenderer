#include "LYHapticInterface.h"

void LYHapticInterface::setPosition( float3 pos )
{
	// Haptics only
	glm::mat4 inverseTransformation = glm::inverse(this->m_ViewMatrix * this->m_ModelMatrix);
	glm::vec3 p = glm::vec3(inverseTransformation * glm::vec4(pos.x, pos.y, pos.z, 1));
	m_collider.hapticPosition = make_float3(p.x, p.y, p.z);

	// Graphics only
	glm::mat4 finalTransformation = glm::translate(this->m_ModelMatrix, p);
	m_HIPMatrix = finalTransformation;
	p = glm::vec3(finalTransformation * glm::vec4(glm::vec3(0,0,0),1.0));
	finalTransformation = glm::translate(this->m_ModelMatrix, glm::vec3(m_collider.scpPosition.x, m_collider.scpPosition.y, m_collider.scpPosition.z));
	m_ProxyMatrix = finalTransformation;

	if (!bPause) {
		m_collider.scpPosition = m_collider.hapticPosition;
		m_collider.surfaceTgPlane = make_float3(0,1,0);
		m_ProxyMatrix = m_HIPMatrix;
	}

}

void LYHapticInterface::setSpaceHandler( LYSpaceHandler *sh )
{
	m_spaceHandler = sh;
	m_HIPMatrix = glm::mat4();
	m_ProxyMatrix = glm::mat4();
	m_ViewMatrix = glm::mat4();
}

void LYHapticInterface::setWorkspaceScale( float3 dim )
{
	m_workspaceScale = dim;
}

void LYHapticInterface::setRelativePosition( float3 pos )
{
	m_relativePosition = pos;
}

void LYHapticInterface::setCameraMatrix( glm::mat4 t )
{
	m_ViewMatrix = t;
}

void LYHapticInterface::setModelMatrix( glm::mat4 t )
{
	m_ModelMatrix = t;
}

void LYHapticInterface::setSize( float r )
{
	m_size = r;
}

bool LYHapticInterface::toggleForces()
{
	COLLISION_FORCEFEEDBACK = !COLLISION_FORCEFEEDBACK;
	return COLLISION_FORCEFEEDBACK;
}

void LYHapticInterface::setTimer( StopWatchInterface *timer )
{
	m_timer = timer;
}

void LYHapticInterface::setSpeed( float s )
{
	m_speed = s;
}

float3 LYHapticInterface::calculateFeedbackUpdateProxy()
{
	if (!COLLISION_FORCEFEEDBACK) return make_float3(0.0f);
	m_forceVector = m_spaceHandler->calculateFeedbackUpdateProxy(&m_collider);
	return m_forceVector;
}

void LYHapticInterface::pause()
{
	COLLISION_FORCEFEEDBACK = false;
}

void LYHapticInterface::pause( bool p )
{
	bPause = p;
}

void LYHapticInterface::start()
{
	COLLISION_FORCEFEEDBACK = true;
	bPause = false;
}

glm::mat4 LYHapticInterface::getHIPMatrix() const
{
	return m_HIPMatrix;
}

glm::mat4 LYHapticInterface::getProxyMatrix() const
{
	return m_ProxyMatrix;
}

float3 LYHapticInterface::getPosition() const
{
	return m_collider.hapticPosition;
}

float LYHapticInterface::getSpeed() const
{
	return m_speed;
}

float LYHapticInterface::getSize() const
{
	return m_size;
}

LYHapticInterface::LYDEVICE_TYPE LYHapticInterface::getDeviceType() const
{
	return m_deviceType;
}

uint LYHapticInterface::getVBO() const
{
	return vbo;
}

uint LYHapticInterface::getIB() const
{
	return ib;
}

LYMesh* LYHapticInterface::getHIPObject() const
{
	return m_HIPObject;
}

LYMesh* LYHapticInterface::getProxyObject() const
{
	return m_ProxyObject;
}

uint LYHapticInterface::getProxyVBO() const
{
	return m_ProxyObject->getVBO();
}

uint LYHapticInterface::getProxyIB() const
{
	return m_ProxyObject->getIB();
}

uint LYHapticInterface::getHIPVBO() const
{
	return m_HIPObject->getVBO();
}

uint LYHapticInterface::getHIPIB() const
{
	return m_HIPObject->getIB();
}

size_t LYHapticInterface::getProxyNumVertices() const
{
	return m_ProxyObject->getNumVertices();
}

size_t LYHapticInterface::getHIPNumVertices() const
{
	return m_HIPObject->getNumVertices();
}

bool LYHapticInterface::isEnabled() const
{
	return COLLISION_FORCEFEEDBACK;
}

float3 LYHapticInterface::getSurfacePosition() const
{
	return m_collider.scpPosition;
}

float3 LYHapticInterface::getSurfaceNormal() const
{
	return m_collider.surfaceTgPlane;
}

float3 LYHapticInterface::getForceVector() const
{
	return m_forceVector;
}
