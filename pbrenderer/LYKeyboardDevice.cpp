#include "LYKeyboardDevice.h"


LYKeyboardDevice::LYKeyboardDevice(LYSpaceHandler *sh)
{
	m_spaceHandler = sh;
	m_deviceType = LYHapticInterface::KEYBOARD_DEVICE;
	m_collider =	LYVertex();
	m_speed =		0.01f;
	m_size	=		0.03f;

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

	m_modelMatrix = glm::mat4();
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
float3 LYKeyboardDevice::getForceFeedback(float3 pos) const{
	return m_spaceHandler->getForceFeedback(this->getPosition());
}

float3 LYKeyboardDevice::calculateFeedbackUpdateProxy()
{
	return m_spaceHandler->calculateFeedbackUpdateProxy(&m_collider);
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
