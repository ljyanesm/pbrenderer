#pragma once
#include <glm/glm.hpp>
#include <iostream>
#include <iomanip>

#include <GL\glew.h>
#include <GL\freeglut.h>

#include "LYHapticInterface.h"
#include "LYShader.h"
#include "LYCamera.h"
#include "LYMesh.h"

typedef struct {
	unsigned int vertex_array;
	unsigned int vbo_indices;
	unsigned int num_indices;
	//Don't need these to get it working, but needed for deallocation
	unsigned int vbo_data;
} device_mesh2_t;

typedef struct {
	glm::vec3 pt;
	glm::vec2 texcoord;
} vertex2_t;


class LYScreenspaceRenderer
{
public:
	enum DisplayMode {
		DISPLAY_DEPTH = 0,
		DISPLAY_NORMAL = 1,
		DISPLAY_POSITION = 2,
		DISPLAY_COLOR = 3,
		DISPLAY_DIFFUSE = 4,
		DISPLAY_DIFFUSE_SPEC = 5,
		DISPLAY_TOTAL = 6,
		NUM_DISPLAY_MODES
	};

	enum RenderMode {
		POINTS = 0,
		TRIANGLES = 1,
		NUM_RENDER_MODES
	};
	LYScreenspaceRenderer(LYCamera *c);
	LYScreenspaceRenderer(void);
	~LYScreenspaceRenderer(void);

	void addDisplayMesh(LYMesh *mesh)
	{
		m_objects.push_back(mesh);
	}
	void displayTriangleMesh(LYMesh *mesh, DisplayMode mode);
	void displayPointMesh(LYMesh *mesh, DisplayMode mode);
	void display(DisplayMode mode);

	void setCamera(LYCamera *c);
	void setMesh(LYMesh *m);
	void setPointRadius(float r);
	void setPointScale(float s);
	void setOriented(int o);
	void setPointDiv(int d);
	void setCollider(LYHapticInterface* haptic);

	void dumpIntoPdb(std::string o);

	GLuint getDepthFBO() { return m_FBO; }

protected:
	void _initQuad();
	void _initFBO(int w, int h);
	void _setTextures();
	void _bindFBO(GLuint FBO);
	void _drawPoints(LYMesh *mesh);
	void _drawTriangles(LYMesh *mesh);
	void _drawCollider();
	void _initShaders();

private:
	LYShader	*depthShader;		//Point sprites shader to depth and color render targets.
	LYShader	*blurDepthShader;	//Point sprites shader to depth and color render targets.
	LYShader	*normalShader;		//Normals from image space depth and positions.
	LYShader	*diffuseShader;		//Diffuse and specular? lighting from normals.
	LYShader	*totalShader;		//Combination of the previous passes.

	const LYCamera*			m_camera;
	std::vector<LYMesh*>	m_objects;

	int			m_pointDiv;

	GLuint		m_depthTexture;
	GLuint		m_colorTexture;
	GLuint		m_positionTexture;
	GLuint		m_normalTexture;
	GLuint		m_blurDepthTexture;

	GLuint		m_FBO;
	GLuint		m_normalsFBO;
	GLuint		m_blurDepthFBO;

	float		m_pointRadius;
	float		m_pointScale;

	int			m_oriented;

	device_mesh2_t m_device_quad;

	LYHapticInterface const* haptic_interface;
};
