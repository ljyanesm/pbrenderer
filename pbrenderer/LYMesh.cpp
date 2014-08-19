#include <assert.h>

#include "LYMesh.h"

LYMesh::LYMesh(const std::vector<LYVertex>& Vertices,
                          const std::vector<unsigned int>& Indices)
						  : m_points(true)
{
    this->numVertices = NumIndices = Indices.size();
	m_Vertices = Vertices;
	size_t classSize = sizeof(LYVertex) * Vertices.size() + sizeof(unsigned int) * NumIndices;
	(classSize > 1024*1024) ? 
		printf("This mesh requires: %d MB\n", classSize / (1024*1024)) :
		(classSize > 1024) ? printf("This mesh requires: %d Kb\n", classSize / (1024)):
							 printf("This mesh requires: %d Bytes\n", classSize);

	glm::vec3 max, min;
	max = glm::vec3(-99999999.9f);
	min = glm::vec3( 99999999.9f);
	int ii = 0;
	for (std::vector<LYVertex>::const_iterator i = Vertices.begin(); i != Vertices.end(); ++i, ii++){
		//pos[ii] = make_float4(i->m_pos);
		//color[ii] = make_float4(i->m_color);
		//normal[ii] = make_float4(i->m_normal);
		//force[ii] = make_float4(0.0f);
		//density[ii] = 0.0f;

		if (max.x < i->m_pos.x) max.x = i->m_pos.x;
		if (max.y < i->m_pos.y) max.y = i->m_pos.y;
		if (max.z < i->m_pos.z) max.z = i->m_pos.z;

		if (min.x > i->m_pos.x) min.x = i->m_pos.x;
		if (min.y > i->m_pos.y) min.y = i->m_pos.y;
		if (min.z > i->m_pos.z) min.z = i->m_pos.z;
	}

	modelCentre = (max+min)*0.5f;
	modelMatrix = glm::mat4();
	float maxDist = -1.0f;
	float dX, dY, dZ;
	dX = abs(max.x - min.x);
	dY = abs(max.y - min.y);
	dZ = abs(max.z - min.z);
	if (maxDist < dX) maxDist = dX;
	if (maxDist < dY) maxDist = dY;
	if (maxDist < dZ) maxDist = dZ;
	float factor = maxDist;
	minP = min;
	maxP = max;
	modelScale = factor;
	modelMatrix = glm::translate(modelMatrix, -modelCentre);

	glGenBuffers(1, &VB);
  	glBindBuffer(GL_ARRAY_BUFFER, VB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * Vertices.size(), &Vertices[0], GL_DYNAMIC_DRAW);

    glGenBuffers(1, &IB);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IB);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * NumIndices, &Indices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

LYMesh::LYMesh() : m_points(true)
{
	VB = INVALID_OGL_VALUE;
	IB = INVALID_OGL_VALUE;
	NumIndices  = 0;
	MaterialIndex = INVALID_MATERIAL;
}


LYMesh::~LYMesh()
{
    Clear();
}


void LYMesh::Clear()
{
	if (VB != INVALID_OGL_VALUE)
	{
		glDeleteBuffers(1, &VB);
	}

	if (IB != INVALID_OGL_VALUE)
	{
		glDeleteBuffers(1, &IB);
	}
}

void LYMesh::draw( RenderType renderType )
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	int vbo = this->getVBO();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
	int ib = this->getIB();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	size_t numIndices = this->getNumIndices();
	switch (renderType)
	{
	case POINTS:
		numIndices = numVertices;
		glDrawElements(GL_POINTS, numIndices, GL_UNSIGNED_INT, nullptr);
		break;
	case TRIANGLES:
		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, nullptr);
		break;
	case LINES:
		glDrawElements(GL_LINES, numIndices, GL_UNSIGNED_INT, nullptr);
		break;
	}
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}

void LYMesh::setPositions(std::vector<LYVertex>& pos)
{
	glBindBuffer(GL_ARRAY_BUFFER, VB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * pos.size(), &pos[0], GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
