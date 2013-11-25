#include <assert.h>

#include "LYMesh.h"

LYMesh::LYMesh(const std::vector<LYVertex>& Vertices,
                          const std::vector<unsigned int>& Indices)
{
    this->numVertices = NumIndices = Indices.size();
	m_Vertices = Vertices;

	printf("This mesh requeres: %d bytes\n", sizeof(LYVertex) * Vertices.size() + sizeof(unsigned int) * NumIndices);
    glGenBuffers(1, &VB);
  	glBindBuffer(GL_ARRAY_BUFFER, VB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * Vertices.size(), &Vertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &IB);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IB);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * NumIndices, &Indices[0], GL_STATIC_DRAW);
	glm::vec3 max, min;
	max = glm::vec3(-99999999.9f);
	min = glm::vec3( 99999999.9f);
	for (std::vector<LYVertex>::const_iterator i = Vertices.begin(); i != Vertices.end(); ++i){
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
	modelScale = factor;
	modelMatrix = glm::translate(modelMatrix, -modelCentre);
}

LYMesh::LYMesh()
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
