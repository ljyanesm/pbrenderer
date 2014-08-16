#pragma once
#include <iostream>
#include "vector_functions.h"
#include "vector_types.h"

struct LYVertex
{
	float3	m_pos;
	float3	m_normal;
	float3	m_color;
	float2	m_tex;
	float	m_density;
	__host__ __device__
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

	bool operator<(const LYVertex& rhs) const
	{
		return ( 
			m_pos.x < rhs.m_pos.x &&
			m_pos.y < rhs.m_pos.y &&
			m_pos.z < rhs.m_pos.z 
			);
	}

	bool operator==(const LYVertex& rhs) const
	{
		float eps = 0.000001f;
		return ( 
			(m_pos.x - rhs.m_pos.x) < eps &&
			(m_pos.y - rhs.m_pos.y) < eps &&
			(m_pos.z - rhs.m_pos.z) < eps 
			);
	}

	friend
		std::ostream& operator << (std::ostream& stream, const LYVertex& p) {
			return (stream << "(" << p.m_pos.x << ", " << p.m_pos.y << ", " << p.m_pos.z << ")");
	}
};