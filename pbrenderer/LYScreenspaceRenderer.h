#pragma once
#include <glm/glm.hpp>
#include <iostream>
#include <iomanip>

#include <GL\glew.h>
#include <GL\freeglut.h>

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
		DISPLAY_FRESNEL = 6,
		DISPLAY_REFLECTION = 7,
		DISPLAY_FRES_REFL = 8,
		DISPLAY_THICKNESS = 9,
		DISPLAY_REFRAC = 10,
		DISPLAY_TOTAL = 11,
		NUM_DISPLAY_MODES
	};
	LYScreenspaceRenderer(LYMesh *m, LYCamera *c);
	LYScreenspaceRenderer(void);
	~LYScreenspaceRenderer(void);

	void display(DisplayMode mode = DISPLAY_TOTAL);

	void setCamera(LYCamera *c) { m_camera = c; }
	void setMesh(LYMesh *m) { m_mesh = m; }
	void setPointRadius(float r) { m_pointRadius = r; }

	void dumpIntoPdb(std::string o);

protected:
	void _initQuad();
	void _initFBO(int w, int h);
	void _setTextures();
	void _bindFBO(GLuint FBO);
	void _drawPoints();
	void _initShaders();

private:
	LYshader *mainShader;
	LYshader *depthShader;		//Point sprites shader to depth and color render targets.
	LYshader *blurDepthShader;		//Point sprites shader to depth and color render targets.
	LYshader *normalShader;		//Normals from image space depth and positions.
	LYshader *diffuseShader;	//Diffuse and specular? lighting from normals.
	LYshader *totalShader;		//Combination of the previous passes.

	LYCamera *m_camera;
	LYMesh *m_mesh;

	GLuint m_depthTexture;
	GLuint m_colorTexture;
	GLuint m_positionTexture;
	GLuint m_normalTexture;
	GLuint m_blurDepthTexture;

	GLuint m_FBO;
	GLuint m_normalsFBO;
	GLuint m_blurDepthFBO;

	float m_pointRadius;

	device_mesh2_t m_device_quad;
};
