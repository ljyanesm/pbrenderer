#pragma once
#include "LYMesh.h"
#include "LYShader.h"
#include "LYPLYLoader.h"
#include "LYCamera.h"
class OverlayRenderer
{
	LYMesh *surfaceObject;				// Object to represent the plane and surface point of contact
	LYMesh *hapticWorkspaceObject;		// Box to show the haptic workspace

	LYMesh *vectorObject;				// Arrow object to use as vector representation!

	LYShader *normalShader;				// General shader

	LYCamera *m_camera;

	GLuint m_depthFBO;

	glm::mat4 sceneModelMatrix;
public:
	OverlayRenderer(LYPLYLoader *ply_loader, LYCamera *cam);
	~OverlayRenderer(void);

	void display() const;

	void setDepthFBO(GLuint dFBO) { m_depthFBO = dFBO; }
	void setSceneModelMatrix (glm::mat4 mMat) { sceneModelMatrix = mMat; }
private:
	void _drawTriangles(LYMesh *m_mesh) const;
};
