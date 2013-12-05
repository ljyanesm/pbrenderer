#include "LYKeyboardDevice.h"


LYKeyboardDevice::LYKeyboardDevice(LYSpaceHandler *sh, LYMesh *proxyMesh, LYMesh *hipMesh)
{
	m_spaceHandler = sh;
	m_deviceType = LYHapticInterface::KEYBOARD_DEVICE;
	m_collider =	LYVertex();
	m_speed =		0.01f;
	m_size	=		0.03f;

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

	m_HIPMatrix = glm::mat4();
}


LYKeyboardDevice::~LYKeyboardDevice(void)
{
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ib);
}

float3 LYKeyboardDevice::getPosition() const{
	return m_collider.m_pos;
}

float3 *LYKeyboardDevice::getHIP() {
	return &(m_collider.m_pos);
}

void LYKeyboardDevice::setPosition(float3 pos) {
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

float3 LYKeyboardDevice::calculateFeedbackUpdateProxy()
{
	float3 force = m_spaceHandler->calculateFeedbackUpdateProxy(&m_collider);
	return force;
}

float LYKeyboardDevice::getSpeed() const 
{
	return m_speed;
}

float LYKeyboardDevice::getSize() const 
{
	return m_size;
}

uint LYKeyboardDevice::getIB() const
{
	return ib;
}

uint LYKeyboardDevice::getVBO() const
{
	return vbo;
}

void LYKeyboardDevice::setSpaceHandler( LYSpaceHandler *sh )
{
	m_spaceHandler = sh;
}

void LYKeyboardDevice::setSize( float r )
{
	m_size = r;
}
