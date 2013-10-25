#include <assert.h>

#include "LYMesh.h"

LYMesh::MeshEntry::MeshEntry()
{
    VB = INVALID_OGL_VALUE;
    IB = INVALID_OGL_VALUE;
    NumIndices  = 0;
    MaterialIndex = INVALID_MATERIAL;
};

LYMesh::MeshEntry::~MeshEntry()
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

void LYMesh::MeshEntry::Init(const std::vector<LYVertex>& Vertices,
                          const std::vector<unsigned int>& Indices)
{
	m_Vertices = Vertices;
    NumIndices = Indices.size();

    glGenBuffers(1, &VB);
  	glBindBuffer(GL_ARRAY_BUFFER, VB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * Vertices.size(), &Vertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &IB);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IB);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * NumIndices, &Indices[0], GL_STATIC_DRAW);
}

LYMesh::LYMesh()
{
}


LYMesh::~LYMesh()
{
    Clear();
}


void LYMesh::Clear()
{
    for (unsigned int i = 0 ; i < m_Textures.size() ; i++) {
		if (m_Textures[i]) delete m_Textures[i];
    }
}

static int vertex_x(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_pos.x =		(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

static int vertex_y(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_pos.y =		(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

static int vertex_z(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_pos.z =		(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

static int vertex_nx(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_normal.x =	(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

static int vertex_ny(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_normal.y =	(float) ply_get_argument_value(argument);
	index++;
	return 1;
}

static int vertex_nz(p_ply_argument argument) {
	long eol;
	static unsigned long index = 0;
	std::vector<LYVertex> *Vertices;
	ply_get_argument_user_data(argument, (void**) &Vertices, &eol);
	Vertices->at(index).m_normal.z = (float) ply_get_argument_value(argument);
	index++;
	return 1;
}

bool LYMesh::LoadPoints(const std::string& Filename)
{
	Clear();
	bool ret = false;
	std::string currentLine;
	m_Entries.resize(1);

	std::vector<LYVertex> Vertices;
	std::vector<unsigned int> Indices;
	unsigned long numVertices(0);

	unsigned long nvertices;
	p_ply ply = ply_open(Filename.c_str(), NULL, 0, NULL);
	if (!ply) return 1;
	if (!ply_read_header(ply)) return 1;
	
	nvertices = 
	ply_set_read_cb(ply, "vertex", "x", vertex_x,(void *) &Vertices, 0);
	ply_set_read_cb(ply, "vertex", "y", vertex_y,(void *) &Vertices, 1);
	ply_set_read_cb(ply, "vertex", "z", vertex_z,(void *) &Vertices, 2);
	ply_set_read_cb(ply, "vertex", "nx", vertex_nx,(void *) &Vertices, 3);
	ply_set_read_cb(ply, "vertex", "ny", vertex_ny,(void *) &Vertices, 4);
	ply_set_read_cb(ply, "vertex", "nz", vertex_nz,(void *) &Vertices, 5);

	Vertices.resize(nvertices);

	if (!ply_read(ply)) return 1;
	ply_close(ply);
	for (unsigned long i = 0; i < nvertices; i++)
	{
		Indices.push_back(i);
	}
	m_Entries[0].numVertices = Vertices.size();
	m_Entries[0].Init(Vertices, Indices);
	return 0;
}

//bool LYMesh::LoadMesh(const std::string& Filename)
//{
//	// Release the previously loaded mesh (if it exists)
//	Clear();
//	bool Ret = false;
//	Assimp::Importer Importer;
//	std::string filename(Filename);
//	const aiScene* pScene = Importer.ReadFile(filename.c_str(), aiProcess_FlipUVs);
//
//	if (pScene) {
//		Ret = InitFromScene(pScene, filename);
//	}
//	else {
//		printf("Error parsing '%s': '%s'\n", Filename.c_str(), Importer.GetErrorString());
//	}
//	printf("The model has been loaded correctly\n");
//	return Ret;
//}
//
//bool LYMesh::InitFromScene(const aiScene* pScene, const std::string& Filename)
//{  
//    m_Entries.resize(pScene->mNumMeshes);
//    m_Textures.resize(pScene->mNumMaterials);
//
//    // Initialize the meshes in the scene one by one
//    for (unsigned int i = 0 ; i < m_Entries.size() ; i++) {
//        const aiMesh* paiMesh = pScene->mMeshes[i];
//        InitMesh(i, paiMesh);
//    }
//
//    return InitMaterials(pScene, Filename);
//}
//
//void LYMesh::InitMesh(unsigned int Index, const aiMesh* paiMesh)
//{
//    m_Entries[Index].MaterialIndex = paiMesh->mMaterialIndex;
//    
//    std::vector<LYVertex> Vertices;
//    std::vector<unsigned int> Indices;
//    const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);
//
//    for (unsigned int i = 0 ; i < paiMesh->mNumVertices ; i++) {
//        const aiVector3D* pPos      = &(paiMesh->mVertices[i]);
//        const aiVector3D* pNormal   = &(paiMesh->mNormals[i]);
//        const aiVector3D* pTexCoord = paiMesh->HasTextureCoords(0) ? &(paiMesh->mTextureCoords[0][i]) : &Zero3D;
//		const aiColor4D*  pColor	= paiMesh->HasVertexColors(0) ? &(paiMesh->mColors[0][i]) : &aiColor4D();
//
//        LYVertex v(make_float3(pPos->x, pPos->y, pPos->z),
//                 make_float2(pTexCoord->x, pTexCoord->y),
//                 make_float3(pNormal->x, pNormal->y, pNormal->z),
//				 make_float3(pColor->r, pColor->g, pColor->b),
//				 int(i));
//
//        Vertices.push_back(v);
//    }
//
//    for (unsigned int i = 0 ; i < paiMesh->mNumVertices ; i++) {
//        Indices.push_back(i);
//    }
//	m_Entries[Index].numVertices = Vertices.size();
//    m_Entries[Index].Init(Vertices, Indices);
//}
//
//bool LYMesh::InitMaterials(const aiScene* pScene, const std::string& Filename)
//{
//    // Extract the directory part from the file name
//    std::string::size_type SlashIndex = Filename.find_last_of("/");
//    std::string Dir;
//
//    if (SlashIndex == std::string::npos) {
//        Dir = ".";
//    }
//    else if (SlashIndex == 0) {
//        Dir = "/";
//    }
//    else {
//        Dir = Filename.substr(0, SlashIndex);
//    }
//
//    bool Ret = true;
//
//    // Initialize the materials
//    for (unsigned int i = 0 ; i < pScene->mNumMaterials ; i++) {
//        const aiMaterial* pMaterial = pScene->mMaterials[i];
//
//        m_Textures[i] = NULL;
//
//        if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
//            aiString Path;
//
//            if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
//                std::string FullPath = Dir + "/" + Path.data;
//                m_Textures[i] = new LYTexture(GL_TEXTURE_2D, FullPath.c_str());
//
//                if (!m_Textures[i]->Load()) {
//                    printf("Error loading texture '%s'\n", FullPath.c_str());
//                    delete m_Textures[i];
//                    m_Textures[i] = NULL;
//                    Ret = false;
//                }
//                else {
//                    printf("Loaded texture '%s'\n", FullPath.c_str());
//                }
//            }
//        }
//
//        // Load a white texture in case the model does not include its own texture
//        if (!m_Textures[i]) {
//            m_Textures[i] = new LYTexture(GL_TEXTURE_2D, "./white.png");
//
//            Ret = m_Textures[i]->Load();
//        }
//    }
//
//    return Ret;
//}