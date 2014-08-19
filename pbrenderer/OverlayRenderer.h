#pragma once
#include "LYMesh.h"
#include "LYShader.h"
#include "LYPLYLoader.h"
#include "LYCamera.h"
class OverlayRenderer
{
	LYMesh *surfaceObject;				// Object to represent the plane and surface point of contact
	LYMesh *cubeObject;		// Box to show the haptic workspace

	LYMesh *vectorObject;				// Arrow object to use as vector representation!

	LYShader *normalShader;				// General shader

	LYCamera *m_camera;

	GLuint m_depthFBO;

	glm::mat4 sceneViewMatrix;
	glm::mat4 SCPPositionMatrix;
	float4	surfacePosition;
	float4	surfaceNormal;
	float4	forceVector;
public:
	OverlayRenderer(LYPLYLoader *ply_loader, LYCamera *cam);
	~OverlayRenderer(void);

	void display() const;

	void setDepthFBO(GLuint dFBO) { m_depthFBO = dFBO; }
	void setSceneViewMatrix(glm::mat4 mMat) { sceneViewMatrix = mMat; }
	void setSCPPositionMatrix(glm::mat4 scpMat) { SCPPositionMatrix = scpMat; }
	void setSurfacePosition(float4 p) { surfacePosition = p; }
	void setSurfaceNormal(float4 n) {surfaceNormal = n; }
	void setForceVector(float4 f) { forceVector = f; }

};
