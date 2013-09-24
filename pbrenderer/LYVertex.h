#pragma once
#include "vector_functions.h"
#include "vector_types.h"
#include <glm/glm.hpp>

struct LYVertex
{
	float3	m_pos;
	float3	m_normal;
	float3	m_color;
	float2	m_tex;
	int			m_objectID;
	
	LYVertex() 
	{
		m_pos = float3();
		m_normal = float3();
		m_color = float3();
		m_tex = float2();
		m_objectID = 0;
	}
	LYVertex(const float3& pos, const float2& tex, const float3& normal, const float3 color, int id)
	{
		m_pos		= pos;
		m_normal	= normal;
		m_color		= color;
		m_tex		= tex;
		m_objectID	= id;
	}
};