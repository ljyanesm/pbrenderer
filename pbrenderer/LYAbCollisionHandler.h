#pragma once

#include "LYVertex.h"


/*
Abstract class for the collision handler, can be implemented either on GPU or CPU
*/
class LYAbCollisionHandler
{
public:
	virtual ~LYAbCollisionHandler(void);

	virtual LYVertex insertPoint(LYVertex p) = 0;
	virtual LYVertex insertPoints(LYVertex *p) = 0;
	virtual LYVertex getForce() = 0;

private:
	virtual void computeCollision() = 0;

	LYVertex force;
};

