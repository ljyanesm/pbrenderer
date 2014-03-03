#include "OverlayRenderer.h"

OverlayRenderer::OverlayRenderer(LYPLYLoader *ply_loader, LYCamera *cam)
{
	normalShader = new LYShader("./shaders/regularShader.vs", "./shaders/regularShader.frag");
	vectorObject = ply_loader->getInstance().readPolygonData("arrow.ply");
	surfaceObject = ply_loader->getInstance().readPolygonData("surface.ply");

	std::vector<LYVertex> qb_vertices;
	std::vector<uint> qb_indices;

	/*
	Create a cube to make the wireframe cube...
	*/

	qb_vertices.push_back(LYVertex(make_float3( 1.0f,  1.0f,  1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 0 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3(-1.0f,  1.0f,  1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 1 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3(-1.0f, -1.0f,  1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 2 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3( 1.0f, -1.0f,  1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 3 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3( 1.0f,  1.0f, -1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 4 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3(-1.0f,  1.0f, -1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 5 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3(-1.0f, -1.0f, -1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 6 (X, Y, Z)
	qb_vertices.push_back(LYVertex(make_float3( 1.0f, -1.0f, -1.0f), 
		make_float2(0.0f), make_float3(0.0f), make_float3(0.0f), 0)); // Vertex 7 (X, Y, Z)

	qb_indices.push_back(0);
	qb_indices.push_back(1);

	qb_indices.push_back(1);
	qb_indices.push_back(2);

	qb_indices.push_back(2);
	qb_indices.push_back(3);

	qb_indices.push_back(3);
	qb_indices.push_back(0);

	qb_indices.push_back(4);
	qb_indices.push_back(5);

	qb_indices.push_back(5);
	qb_indices.push_back(6);

	qb_indices.push_back(6);
	qb_indices.push_back(7);

	qb_indices.push_back(7);
	qb_indices.push_back(4);

	qb_indices.push_back(0);
	qb_indices.push_back(4);

	qb_indices.push_back(1);
	qb_indices.push_back(5);

	qb_indices.push_back(2);
	qb_indices.push_back(6);

	qb_indices.push_back(3);
	qb_indices.push_back(7);

	cubeObject = new LYMesh(qb_vertices, qb_indices);
	m_camera = cam;
}


OverlayRenderer::~OverlayRenderer(void)
{
	delete vectorObject;
	delete surfaceObject;
	delete cubeObject;
}

void OverlayRenderer::display() const {
	
	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_depthFBO);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	int width(m_camera->getWidth());
	int height(m_camera->getHeight());
	glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, 
		GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	normalShader->useShader();
	glActiveTexture(GL_TEXTURE0);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	glm::mat4 modelView;
	glm::mat4 mvpMat;
	glm::mat4 model;
	glm::mat4 projection;

	glm::mat4 modelOrientation;

	glm::vec3 surface_normal = -glm::vec3(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z);
	if ( glm::length(surface_normal) && surface_normal != glm::vec3(0,1,0))
		modelOrientation = glm::transpose(glm::lookAt(glm::vec3(0,0,0), surface_normal, glm::vec3(0,1,0)));
	else
		modelOrientation = glm::transpose(glm::lookAt(glm::vec3(0,0,0), m_camera->getPosition(), glm::vec3(0,1,0)));

	/*
		Wire frame cube for model
	*/
	model = glm::mat4();
	modelView = m_camera->getViewMatrix() * model;
	mvpMat = m_camera->getProjection() * modelView;
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"modelViewMat"),1,GL_FALSE, &modelView[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"MVPMat"),1,GL_FALSE, &mvpMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_Persp"),1,GL_FALSE, &projection[0][0]);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(),"fragDepth"), 0);
	_drawLines(cubeObject);

	/*
	 Surface object:
		Is located at the surface point and its oriented by the surface normal
	 */

	projection = m_camera->getProjection();

	model = glm::mat4();
	model *= SCPPositionMatrix;
	if( glm::length(surface_normal)) model *= modelOrientation;
	model *= glm::scale(0.1f, 0.1f, 0.1f);
	modelView = sceneViewMatrix * model;
	mvpMat = m_camera->getProjection() * modelView;
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"modelViewMat"),1,GL_FALSE, &modelView[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"MVPMat"),1,GL_FALSE, &mvpMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_Persp"),1,GL_FALSE, &projection[0][0]);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(),"fragDepth"), 1);
	_drawTriangles(surfaceObject);

	/*
	Force vector:
		Goes from the position of the surface point with the direction of the surface normal
		and has the size of its magnitude.
	*/
	float force_mag = length(forceVector);
	model = glm::mat4();
	model *= SCPPositionMatrix;
	model *= glm::translate((-surface_normal)*0.1f);
	if( glm::length(surface_normal)) model *= modelOrientation;
	model *= glm::scale(glm::vec3(0.5f, 0.5f, 0.5f));
	model *= glm::scale(glm::vec3(1.f, 1.f, 0.5f+force_mag));
	modelView = sceneViewMatrix * model;
	mvpMat = m_camera->getProjection() * modelView;
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"modelViewMat"),1,GL_FALSE, &modelView[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"MVPMat"),1,GL_FALSE, &mvpMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_Persp"),1,GL_FALSE, &projection[0][0]);
	glUniform4fv( glGetUniformLocation(normalShader->getProgramId(), "lightDir"), 1, &m_camera->getLightDir()[0]);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(),"fragDepth"), 1);

	if ( force_mag > 0.01f) _drawTriangles(vectorObject);


	normalShader->delShader();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);

}

void OverlayRenderer::_drawTriangles(LYMesh *m_mesh) const
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	int vbo = m_mesh->getVBO();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
	int ib = m_mesh->getIB();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	size_t numIndices = m_mesh->getNumIndices();
	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}

void OverlayRenderer::_drawLines(LYMesh *m_mesh) const
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	int vbo = m_mesh->getVBO();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
	int ib = m_mesh->getIB();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	size_t numIndices = m_mesh->getNumIndices();

	glDrawElements(GL_LINES, numIndices, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}