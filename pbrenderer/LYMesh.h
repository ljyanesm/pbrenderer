#ifndef MESH_H
#define	MESH_H

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <GL/glew.h>

#include <helper_functions.h>
#include <helper_math.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "LYVertex.h"
#include "LYTexture.h"

class LYMesh
{
#define INVALID_OGL_VALUE	0xFFFFFFFF
#define INVALID_MATERIAL	0xFFFFFFFF

public:
    LYMesh();
	LYMesh(const std::vector<LYVertex>& Vertices,
		const std::vector<unsigned int>& Indices);
    ~LYMesh();

	glm::mat4	getModelMatrix() { return modelMatrix; }
	glm::vec3	getModelCentre() { return modelCentre; }
	glm::vec3	getMinPoint() { return minP; }
	glm::vec3	getMaxPoint() { return maxP; }
	float		getScale() { return modelScale; }
	GLuint		getVBO() { return VB; }
	GLuint		getIB() { return IB; }
	size_t		getNumIndices() { return NumIndices; }	// numIndices == numVertices  for our purposes
	size_t		getNumVertices() { return m_Vertices.size(); }	// numIndices == numVertices  for our purposes
	bool		getRenderPoints(){ return m_points; }
	std::vector<LYVertex> *getVertices() { return &m_Vertices; }
	
	void setModelMatrix(glm::mat4 m) { modelMatrix = m; } 
	void setRenderMode(bool points){ m_points = points; }

private:
    void Clear();

    std::vector<LYVertex> m_Vertices;
	glm::mat4	modelMatrix;
	glm::vec3	modelCentre;
	glm::vec3	minP, maxP;
	float		modelScale;
	GLuint VB;
	GLuint IB;
	size_t NumIndices;
	size_t numVertices;
	unsigned int MaterialIndex;

	bool m_points;

	float4		*pos;
	float4		*color;
	float4		*normal;
	float4		*force;
	float		*density;

	uint nX, nY, nZ;
	uint nNX, nNY, nNZ;
	uint nR, nG, nB;
};


#endif	/* MESH_H */

