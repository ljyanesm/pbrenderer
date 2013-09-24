#pragma once
#include "defines.h"
#include "LYVertex.h"
#include "vector_functions.h"
#include "vector_types.h"
#include <glm\glm.hpp>

class LYHapticInterface
{
public:
	virtual float3 getPosition() const = 0;
	virtual void setPosition(float3 pos) = 0;
	virtual float3 getForceFeedback() const = 0;
	virtual float getSpeed() const = 0;
	virtual float getSize() const = 0;

	virtual uint getVBO()	const = 0;
	virtual uint getIB()	const	= 0;
};
