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
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "LYVertex.h"
#include "LYTexture.h"
#include "LYPLYLoader.h"


class LYMesh
{
#define INVALID_OGL_VALUE	0xFFFFFFFF
#define INVALID_MATERIAL	0xFFFFFFFF

	struct MeshEntry {
		MeshEntry();

		~MeshEntry();

		void Init(const std::vector<LYVertex>& Vertices,
			const std::vector<unsigned int>& Indices);

		std::vector<LYVertex> m_Vertices;
		GLuint VB;
		GLuint IB;
		size_t NumIndices;
		size_t numVertices;
		unsigned int MaterialIndex;
};

public:
    LYMesh();

    ~LYMesh();

	//bool LoadMesh(const std::string& Filename);
	bool LoadPoints(const std::string& Filename);

	const std::vector<MeshEntry> *getEntries() { return &m_Entries; }
	const std::vector<LYTexture*>  *getTextures() { return &m_Textures; }

private:
 //   bool InitFromScene(const aiScene* pScene, const std::string& Filename);
	//void InitMesh(unsigned int Index, const aiMesh* paiMesh);
 //   bool InitMaterials(const aiScene* pScene, const std::string& Filename);
    void Clear();

    std::vector<MeshEntry> m_Entries;
    std::vector<LYTexture*> m_Textures;
};


#endif	/* MESH_H */

