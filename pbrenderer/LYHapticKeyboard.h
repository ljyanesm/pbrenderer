#pragma once
#include "LYHapticInterface.h"
class LYHapticKeyboard :
	public LYHapticInterface
{
public:
	LYHapticKeyboard(void);
	~LYHapticKeyboard(void);

	glm::vec3 getPosition();
	void setPosition(glm::vec3 pos);
	glm::vec3 getForceFeedback();
	float getSpeed();

private:
	glm::vec3	m_position;
	glm::vec3	m_direction;
	float		m_speed;
};

