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
	float		getScale() { return modelScale; }
	GLuint		getVBO() { return VB; }
	GLuint		getIB() { return IB; }
	size_t		getNumIndices() { return m_Vertices.size(); }	// numIndices == numVertices  for our purposes
	size_t		getNumVertices() { return m_Vertices.size(); }	// numIndices == numVertices  for our purposes

	std::vector<LYVertex> *getVertices() { return &m_Vertices; }
	void setModelMatrix(glm::mat4 m) { modelMatrix = m; } 

private:
    void Clear();

    std::vector<LYVertex> m_Vertices;
	glm::mat4	modelMatrix;
	glm::vec3	modelCentre;
	float		modelScale;
	GLuint VB;
	GLuint IB;
	size_t NumIndices;
	size_t numVertices;
	unsigned int MaterialIndex;

	uint nX, nY, nZ;
	uint nNX, nNY, nNZ;
	uint nR, nG, nB;
};


#endif	/* MESH_H */

