#pragma once
#include <glm/glm.hpp>

class LYVertex
{
public:
	glm::vec3	m_pos;
	glm::vec3	m_normal;
	glm::vec3	m_color;
	glm::vec2	m_tex;
	int			m_objectID;
	
	LYVertex() {}
	~LYVertex() {}

	LYVertex(const glm::vec3& pos, const glm::vec2& tex, const glm::vec3& normal, const glm::vec3 color, int id)
	{
		m_pos		= pos;
		m_normal	= normal;
		m_color		= color;
		m_tex		= tex;
		m_objectID	= id;
	}
};

