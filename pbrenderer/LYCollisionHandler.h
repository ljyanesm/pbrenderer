#pragma once

#include "glm\glm.hpp"

class LYCollisionHandler
{
public:
	virtual const glm::vec3 detectCollision() const = 0;
};

