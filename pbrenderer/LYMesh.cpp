#include <assert.h>

#include "LYMesh.h"

void LYMesh::Init(const std::vector<LYVertex>& Vertices,
                          const std::vector<unsigned int>& Indices)
{
    NumIndices = Indices.size();

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

LYMesh::LYMesh( const std::string &Filename )
{
	LoadPoints(Filename);
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

int LYMesh::vertex_x(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_pos.x =		(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

int LYMesh::vertex_y(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_pos.y =		(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

int LYMesh::vertex_z(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_pos.z =		(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

int LYMesh::vertex_nx(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_normal.x =	(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

int LYMesh::vertex_ny(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_normal.y =	(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

int LYMesh::vertex_nz(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_normal.z = (float) ply_get_argument_value(argument);
	index++;
	return 1;
}

int LYMesh::vertex_r(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_color.x = (float) ply_get_argument_value(argument) / 255.f;
	index++;
	return 1;
}

int LYMesh::vertex_g(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_color.y = (float) ply_get_argument_value(argument) / 255.f;
	index++;
	return 1;
}

int LYMesh::vertex_b(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	float b = (float) ply_get_argument_value(argument) / 255.f;
	Vertices->at(index).m_color.z = b;
	index++;
	return 1;
}

bool LYMesh::LoadPoints(const std::string& Filename)
{
	Clear();
	bool ret = false;
	std::string currentLine;
	unsigned long numVertices(0);

	unsigned long nvertices;
	p_ply ply = ply_open(Filename.c_str(), NULL, 0, NULL);
	if (!ply) return 1;
	if (!ply_read_header(ply)) return 1;
	
	nvertices = 
	ply_set_read_cb(ply, "vertex", "x", vertex_x,(void *) &m_Vertices, 0);
	ply_set_read_cb(ply, "vertex", "y", vertex_y,(void *) &m_Vertices, 1);
	ply_set_read_cb(ply, "vertex", "z", vertex_z,(void *) &m_Vertices, 2);
	ply_set_read_cb(ply, "vertex", "nx", vertex_nx,(void *) &m_Vertices, 3);
	ply_set_read_cb(ply, "vertex", "ny", vertex_ny,(void *) &m_Vertices, 4);
	ply_set_read_cb(ply, "vertex", "nz", vertex_nz,(void *) &m_Vertices, 5);
	ply_set_read_cb(ply, "vertex", "red", vertex_r,(void *) &m_Vertices, 6);
	ply_set_read_cb(ply, "vertex", "green", vertex_g,(void *) &m_Vertices, 7);
	ply_set_read_cb(ply, "vertex", "blue", vertex_b,(void *) &m_Vertices, 8);

	m_Vertices.resize(nvertices);

	if (!ply_read(ply)) return 1;
	ply_close(ply);

	std::vector<unsigned int> Indices;
	for (unsigned long i = 0; i < nvertices; i++)
	{
		Indices.push_back(i);
	}
	numVertices = m_Vertices.size();
	this->Init(m_Vertices, Indices);
	return 0;
}