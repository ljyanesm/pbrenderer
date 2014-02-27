#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/projection.hpp>
class LYCamera
{
public:
	LYCamera(int w, int h, glm::vec3 pos, glm::vec4 lightPosition) { m_width = w ; m_height = h; position = pos; lightDir = lightPosition;}
	~LYCamera(void);

	void setViewMatrix(glm::mat4 v){ view = v; }
	glm::mat4 getViewMatrix() const { return view; }
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

	glm::mat4 getProjection() const { return projection; }

	int getWidth() const { return m_width; }
	int getHeight() const { return m_height; }
	float getFOV() const { return m_fov; }
	float getNear() const { return m_near; }
	float getFar() const { return m_far; }

	const glm::vec4 getLightDir() const { return lightDir; }

	void setPosition(const glm::vec3 pos) { position = pos; }
	const glm::vec3 getPosition() const { return position; }

private:
	glm::mat4 view;
	glm::mat4 projection;

	glm::vec3 position;
	glm::vec4 lightDir;

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

