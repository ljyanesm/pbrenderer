#pragma once

#include "glm\glm.hpp"
#include "LYCollisionHandler.h"

class LYDistanceFieldCollider : public LYCollisionHandler
{
public:
	LYDistanceFieldCollider(void);
	~LYDistanceFieldCollider(void);

	const glm::vec3 detectCollision() const;
};

