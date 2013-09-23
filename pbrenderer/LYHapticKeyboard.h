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
public:
	LYHapticKeyboard(void);
	~LYHapticKeyboard(void);

	glm::vec3 getPosition() const;
	void setPosition(glm::vec3 pos);
	glm::vec3 getForceFeedback() const;
	float getSpeed() const;
	float getSize()	const;

	uint getVBO() const;
	uint getIB() const;

private:
	LYVertex	m_collider;
	glm::vec3	m_position;
	glm::vec3	m_direction;
	float		m_speed;
	float		m_size;

	uint vbo;
	uint ib;
};
