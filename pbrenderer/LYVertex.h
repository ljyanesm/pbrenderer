#pragma once
#include "vector_functions.h"
#include "vector_types.h"

struct LYVertex
{
	float3	m_pos;
	float3	m_normal;
	float3	m_color;
	float2	m_tex;
	float	m_density;
	
	LYVertex() 
	{
		m_pos = float3();
		m_normal = float3();
		m_color = float3();
		m_tex = float2();
		m_density = 0;
	}
	LYVertex(const float3& pos, const float2& tex, const float3& normal, const float3 color, float density)
	{
		m_pos		= pos;
		m_normal	= normal;
		m_color		= color;
		m_tex		= tex;
		m_density	= density;
	}
};