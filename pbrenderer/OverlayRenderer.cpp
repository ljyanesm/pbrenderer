#include "OverlayRenderer.h"


OverlayRenderer::OverlayRenderer(LYPLYLoader *ply_loader, LYCamera *cam)
{
	normalShader = new LYShader("./shaders/regularShader.vs", "./shaders/regularShader.frag", "Normal");
	vectorObject = ply_loader->getInstance().readFile("arrow.ply");
	surfaceObject = ply_loader->getInstance().readFile("surface.ply");
	hapticWorkspaceObject = ply_loader->getInstance().readFile("cube-wire.ply");
	m_camera = cam;
}


OverlayRenderer::~OverlayRenderer(void)
{
}

void OverlayRenderer::display() const {

	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glBindTexture(GL_TEXTURE_2D,0); 
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glEnable(GL_BLEND);
	normalShader->useShader();
	glm::mat4 model =  vectorObject->getModelMatrix();
	glm::rotate(model, 180.0f, 0.0f, 0.0f, 1.0f);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"model"),1,GL_FALSE, &model[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"view"),1,GL_FALSE, &m_camera->getViewMatrix()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"projection"),1,GL_FALSE, &m_camera->getProjection()[0][0]);

		_drawTriangles(vectorObject);
	//	_drawTriangles(surfaceObject);
	//	_drawTriangles(hapticWorkspaceObject);
	normalShader->delShader();
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
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
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
	int ib = m_mesh->getIB();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	size_t numIndices = m_mesh->getNumIndices();
	glDrawElements(GL_LINE_LOOP, numIndices, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}