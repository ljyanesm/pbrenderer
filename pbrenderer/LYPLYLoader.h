#pragma once
#include <vector>
#include <string>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "LYMesh.h"
#include "defines.h"
#include "rply.h"

class LYPLYLoader{
	LYPLYLoader(){}
	LYPLYLoader(LYPLYLoader const&);
	void operator= (LYPLYLoader const&);

	const aiScene *scene;

	static int LYPLYLoader::vertex_x(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nX).m_pos.x =		(float) ply_get_argument_value(argument);
		LYPLYLoader::nX++;
		return 1;
	}

	static int LYPLYLoader::vertex_y(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nY).m_pos.y =		(float) ply_get_argument_value(argument);
		LYPLYLoader::nY++;
		return 1;
	}

	static int LYPLYLoader::vertex_z(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nZ).m_pos.z =		(float) ply_get_argument_value(argument);
		LYPLYLoader::nZ++;
		return 1;
	}

	static int LYPLYLoader::vertex_nx(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nNX).m_normal.x =	(float) ply_get_argument_value(argument);
		LYPLYLoader::nNX++;
		return 1;
	}

	static int LYPLYLoader::vertex_ny(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nNY).m_normal.y =	(float) ply_get_argument_value(argument);
		LYPLYLoader::nNY++;
		return 1;
	}

	static int LYPLYLoader::vertex_nz(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nNZ).m_normal.z = (float) ply_get_argument_value(argument);
		LYPLYLoader::nNZ++;
		return 1;
	}

	static int LYPLYLoader::vertex_r(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nR).m_color.x = (float) ply_get_argument_value(argument) / 255.f;
		LYPLYLoader::nR++;
		return 1;
	}

	static int LYPLYLoader::vertex_g(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		Vertices->at(LYPLYLoader::nG).m_color.y = (float) ply_get_argument_value(argument) / 255.f;
		LYPLYLoader::nG++;
		return 1;
	}

	static int LYPLYLoader::vertex_b(p_ply_argument argument) {
		long eol;
		std::vector<LYVertex> *Vertices;
		ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
		float b = (float) ply_get_argument_value(argument) / 255.f;
		Vertices->at(LYPLYLoader::nB).m_color.z = b;
		LYPLYLoader::nB++;
		return 1;
	}

	static uint nX, nY, nZ;
	static uint nNX, nNY, nNZ;
	static uint nR, nG, nB;

public:
	static LYPLYLoader& getInstance()
	{
		static LYPLYLoader instance; 
		return instance;
	}

	LYMesh* readPolygonData(const std::string &filename)
	{
		LYMesh *ret;

		std::string currentLine;
		unsigned long numVertices(0);

		scene = aiImportFile(filename.c_str(), aiProcess_Triangulate);

		std::vector<LYVertex> m_Vertices;
		std::vector<unsigned int> m_Indices;
		m_Vertices.resize(scene->mMeshes[0]->mNumVertices);
		if(scene)
		{
			for(unsigned int i = 0; i < scene->mMeshes[0]->mNumVertices; i++)
			{
				m_Vertices.at(i).m_pos = make_float3(scene->mMeshes[0]->mVertices[i].x, scene->mMeshes[0]->mVertices[i].y, scene->mMeshes[0]->mVertices[i].z);
				m_Vertices.at(i).m_tex = make_float2(0,0);
				m_Vertices.at(i).m_normal = make_float3(scene->mMeshes[0]->mNormals[i].x, scene->mMeshes[0]->mNormals[i].y, scene->mMeshes[0]->mNormals[i].z);
				if (scene->mMeshes[0]->HasVertexColors(i)) m_Vertices.at(i).m_color = make_float3(scene->mMeshes[0]->mColors[i]->r, scene->mMeshes[0]->mColors[i]->g, scene->mMeshes[0]->mColors[i]->b);
				m_Vertices.at(i).m_objectID = i;
			}

			for (unsigned int i = 0 ; i < scene->mMeshes[0]->mNumFaces ; i++) {
				const aiFace& Face = scene->mMeshes[0]->mFaces[i];
				assert(Face.mNumIndices == 3);
				m_Indices.push_back(Face.mIndices[0]);
				m_Indices.push_back(Face.mIndices[1]);
				m_Indices.push_back(Face.mIndices[2]);
			}

			ret = new LYMesh(m_Vertices, m_Indices);
		}
		m_Vertices.resize(0);
		m_Indices.resize(0);
		return ret;
	}

	LYMesh* readPointData(const std::string &filename)
	{
		LYMesh *ret;
		std::string currentLine;
		unsigned long numVertices(0);

		unsigned long nvertices;
		p_ply ply = ply_open(filename.c_str(), NULL, 0, NULL);
		if (!ply) return 0;
		if (!ply_read_header(ply)) return 0;
		
		std::vector<LYVertex> m_Vertices;

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

		if (!ply_read(ply))
		{
			throw 1;
		}
		ply_close(ply);

		std::vector<unsigned int> Indices;
		for (unsigned long i = 0; i < nvertices; i++)
		{
			Indices.push_back(i);
		}

		ret = new LYMesh(m_Vertices, Indices);
		LYPLYLoader::nX = 0;
		LYPLYLoader::nY = 0;
		LYPLYLoader::nZ = 0;
		LYPLYLoader::nNX = 0;
		LYPLYLoader::nNY = 0;
		LYPLYLoader::nNZ = 0;
		LYPLYLoader::nR = 0; 
		LYPLYLoader::nG = 0; 
		LYPLYLoader::nB = 0;
		m_Vertices.resize(0);
		Indices.resize(0);
		return ret;
	}

};
