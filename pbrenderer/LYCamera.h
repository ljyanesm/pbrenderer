#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/projection.hpp>
class LYCamera
{
public:
	LYCamera(void);
	~LYCamera(void);

	void setModelView(glm::mat4 mv){ modelview = mv; }
	glm::mat4 getModelView(){ return modelview; }
	void perspProjection(int w, int h, float fov, float n, float f)
	{
		m_fov = fov;
		m_width = w;
		m_height = h;
		m_near = n;
		m_far = f;

		projection = glm::perspective(m_fov, (float) m_width / (float) m_height, m_near, m_far);
	}
	void orthoProjection(int l, int r, int bot, int up, int n, int f)
	{
		m_left		= (float) l;
		m_right		= (float) r;
		m_bottom	= (float) bot;
		m_up		= (float) up;
		m_near		= (float) n;
		m_far		= (float) f;
		projection = glm::ortho(m_left, m_right, m_bottom, m_up, m_near, m_far);
	}

	glm::mat4 getProjection() { return projection; }

private:
	glm::mat4 modelview;
	glm::mat4 projection;
	float m_fov;
	float m_near;
	float m_far;
	int m_width;
	int m_height;

	float m_left;
	float m_right;
	float m_bottom;
	float m_up;
};

