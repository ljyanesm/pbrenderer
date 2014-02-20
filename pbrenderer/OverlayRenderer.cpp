#include "OverlayRenderer.h"

OverlayRenderer::OverlayRenderer(LYPLYLoader *ply_loader, LYCamera *cam)
{
	normalShader = new LYShader("./shaders/regularShader.vs", "./shaders/regularShader.frag");
	vectorObject = ply_loader->getInstance().readPolygonData("arrow.ply");
	surfaceObject = ply_loader->getInstance().readPolygonData("surface.ply");
	hapticWorkspaceObject = ply_loader->getInstance().readPolygonData("cube-wire.ply");
	m_camera = cam;
}


OverlayRenderer::~OverlayRenderer(void)
{
	delete vectorObject;
	delete surfaceObject;
	delete hapticWorkspaceObject;
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

	/*
	 Surface object:
		Is located at the surface point and its oriented by the surface normal
	 */
	glm::vec3 surface_normal = -glm::vec3(surfaceNormal.x, surfaceNormal.y, surfaceNormal.z);
	model = glm::mat4();
	model *= SCPPositionMatrix; //glm::translate(glm::vec3(surfacePosition.x, surfacePosition.y, surfacePosition.z));
	if( glm::length(surface_normal)) model *= glm::transpose(glm::lookAt(glm::vec3(0,0,0), surface_normal, glm::vec3(0,1,0)));
	model *= glm::scale(0.1f, 0.1f, 0.1f);
	modelView = sceneViewMatrix * model;
	mvpMat = m_camera->getProjection() * modelView;
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"modelViewMat"),1,GL_FALSE, &modelView[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"MVPMat"),1,GL_FALSE, &mvpMat[0][0]);
	_drawTriangles(surfaceObject);

	/*
	Force vector:
		Goes from the position of the surface point with the direction of the surface normal
		and has the size of its magnitude.
	*/
	float force_mag = length(forceVector);
	model = glm::mat4();
	model *= SCPPositionMatrix;
	if( glm::length(surface_normal)) model *= glm::transpose(glm::lookAt(glm::vec3(0,0,0), surface_normal, glm::vec3(0,1,0)));
	model *= glm::scale(glm::vec3(0.5f, 0.5f, 0.5f));
	model *= glm::scale(glm::vec3(1.f, 1.f, 0.5f+force_mag));
	modelView = sceneViewMatrix * model;
	mvpMat = m_camera->getProjection() * modelView;
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"modelViewMat"),1,GL_FALSE, &modelView[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"MVPMat"),1,GL_FALSE, &mvpMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_Persp"),1,GL_FALSE, &m_camera->getProjection()[0][0]);
	glUniform4fv( glGetUniformLocation(normalShader->getProgramId(), "lightDir"), 1, &m_camera->getLightDir()[0]);

	if ( force_mag > 0.01f) _drawTriangles(vectorObject);

	//model = glm::mat4();
	//model *= glm::scale(0.1f, 0.1f, 0.1f);
	//modelView = m_camera->getViewMatrix() * model;
	//mvpMat = m_camera->getProjection() * modelView;
	//glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"modelViewMat"),1,GL_FALSE, &modelView[0][0]);
	//glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"MVPMat"),1,GL_FALSE, &mvpMat[0][0]);
	//_drawTriangles(hapticWorkspaceObject);
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