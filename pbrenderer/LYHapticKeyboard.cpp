#include "LYHapticKeyboard.h"


LYHapticKeyboard::LYHapticKeyboard(void)
{
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
}


LYHapticKeyboard::~LYHapticKeyboard(void)
{
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ib);
}

float3 LYHapticKeyboard::getPosition() const{
	return m_collider.m_pos;
}
void LYHapticKeyboard::setPosition(float3 pos) {
	m_position = pos;
	m_collider.m_pos = pos;
	std::vector<LYVertex> Vertices;
	Vertices.push_back(m_collider);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * 1, &Vertices[0], GL_STATIC_DRAW);
}
float3 LYHapticKeyboard::getForceFeedback() const{
	return float3();
}

float LYHapticKeyboard::getSpeed() const 
{
	return m_speed;
}

float LYHapticKeyboard::getSize() const 
{
	return m_size;
}

uint LYHapticKeyboard::getIB() const
{
	return ib;
}

uint LYHapticKeyboard::getVBO() const
{
	return vbo;
}
