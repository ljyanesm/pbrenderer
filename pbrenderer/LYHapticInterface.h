#pragma once
#include "LYVertex.h"
#include <glm\glm.hpp>

class LYHapticInterface
{
public:
	virtual glm::vec3 getPosition() = 0;
	virtual void setPosition(glm::vec3 pos) = 0;
	virtual glm::vec3 getForceFeedback() = 0;
	virtual float getSpeed()= 0;
};

