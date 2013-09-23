#pragma once
#include "defines.h"
#include "LYVertex.h"
#include <glm\glm.hpp>

class LYHapticInterface
{
public:
	virtual glm::vec3 getPosition() const = 0;
	virtual void setPosition(glm::vec3 pos) = 0;
	virtual glm::vec3 getForceFeedback() const = 0;
	virtual float getSpeed() const = 0;
	virtual float getSize() const = 0;

	virtual uint getVBO()	const = 0;
	virtual uint getIB()	const	= 0;
};
