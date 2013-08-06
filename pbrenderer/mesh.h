/*

	Copyright 2011 Etay Meiri

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MESH_H
#define	MESH_H

#include <map>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <GL/glew.h>
#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "util.h"
#include "math_3d.h"
#include "texture.h"
#include "LYShader.h"


struct Vertex
{
    Vector3f m_pos;
    Vector2f m_tex;
    Vector3f m_normal;
	Vector3f m_color;

    Vertex() {}

    Vertex(const Vector3f& pos, const Vector2f& tex, const Vector3f& normal, const Vector3f color)
    {
        m_pos    = pos;
        m_tex    = tex;
        m_normal = normal;
		m_color	 = color;
    }
};


class Mesh
{

#define INVALID_MATERIAL 0xFFFFFFFF

	struct MeshEntry {
		MeshEntry();

		~MeshEntry();

		void Init(const std::vector<Vertex>& Vertices,
			const std::vector<unsigned int>& Indices);

		GLuint VB;
		GLuint IB;
		unsigned int NumIndices;
		unsigned int MaterialIndex;
};

public:
    Mesh();

    ~Mesh();

    bool LoadMesh(const std::string& Filename);

	const std::vector<MeshEntry> getEntries() {return m_Entries; }
	const std::vector<Texture*> getTextures() { return m_Textures; }

	void Render();
	void Render2(glm::mat4 p, glm::mat4 mv, float pr);

private:
    bool InitFromScene(const aiScene* pScene, const std::string& Filename);
	void InitMesh(unsigned int Index, const aiMesh* paiMesh);
	void InitShaders();
    bool InitMaterials(const aiScene* pScene, const std::string& Filename);
    void Clear();

    std::vector<MeshEntry> m_Entries;
    std::vector<Texture*> m_Textures;

	LYshader *mainShader;
	LYshader *depthPass;
	LYshader *blurPass;
	LYshader *renderPass;

};


#endif	/* MESH_H */

