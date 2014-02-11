#pragma once

#include "LYVertex.h"

class Collider
{
public:

	float3	hapticPosition;
	float3	scpPosition;
	float3	surfaceTgPlane;

	Collider(void);
	~Collider(void);
};

