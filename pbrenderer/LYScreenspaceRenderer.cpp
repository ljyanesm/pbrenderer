#include "LYScreenspaceRenderer.h"

void checkFramebufferStatus(GLenum framebufferStatus) {
	switch (framebufferStatus) {
	case GL_FRAMEBUFFER_COMPLETE_EXT: break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
		printf("Attachment Point Unconnected");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
		printf("Missing Attachment");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		printf("Dimensions do not match");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
		printf("Formats");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
		printf("Draw Buffer");
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
		printf("Read Buffer");
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
		printf("Unsupported Framebuffer Configuration");
		break;
	default:
		printf("Unknown Framebuffer Object Failure");
		break;
	}
}

LYScreenspaceRenderer::LYScreenspaceRenderer():
	m_pointDiv(1)
{
	_initShaders();
	_initFBO(m_camera->getWidth(), m_camera->getHeight());
	_initQuad();
}

LYScreenspaceRenderer::LYScreenspaceRenderer(LYCamera *c) :
	m_camera(c),
	m_pointDiv(1)
{
	_initShaders();
	_initFBO(m_camera->getWidth(), m_camera->getHeight());
	_initQuad();
}


LYScreenspaceRenderer::~LYScreenspaceRenderer(void)
{
	delete depthShader;
	delete normalShader;
	delete blurDepthShader;
	delete totalShader;

	glDeleteTextures(1, &m_depthTexture);
	glDeleteTextures(1, &m_colorTexture);
	glDeleteTextures(1, &m_normalTexture);
	glDeleteTextures(1, &m_positionTexture);
	glDeleteTextures(1, &m_blurDepthTexture);

	glDeleteFramebuffers(1, &m_FBO);
	glDeleteFramebuffers(1, &m_normalsFBO);
	glDeleteFramebuffers(1, &m_blurDepthFBO);

	glDeleteVertexArrays(1, &(m_device_quad.vertex_array));
	glDeleteBuffers(1,&(m_device_quad.vbo_data));
	glDeleteBuffers(1,&(m_device_quad.vbo_indices));


}

void LYScreenspaceRenderer::_initShaders() 
{
	depthShader = new LYShader("./shaders/depth_pass.vs", "./shaders/depth_pass.frag");
	normalShader = new LYShader("./shaders/normal_pass.vs", "./shaders/normal_pass.frag");
	blurDepthShader = new LYShader("./shaders/blur_pass.vs", "./shaders/blur_pass.frag");
	totalShader = new LYShader("./shaders/shader.vs", "./shaders/shader.frag");
}

void LYScreenspaceRenderer::_setTextures() {
	glBindTexture(GL_TEXTURE_2D,0); 
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
}

void LYScreenspaceRenderer::_bindFBO(GLuint FBO) {
	glBindTexture(GL_TEXTURE_2D,0); //Bad mojo to unbind the framebuffer using the texture
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glClear(GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
}

void LYScreenspaceRenderer::_initQuad() {
	float size = 1.0f;
	vertex2_t verts [] = {  {glm::vec3(-size,size,0.0f),glm::vec2(0,1)},
	{glm::vec3(-size,-size,0.0f),glm::vec2(0,0)},
	{glm::vec3(size,-size,0.0f),glm::vec2(1,0)},
	{glm::vec3(size,size,0.0f),glm::vec2(1,1)}
	};

	unsigned short indices[] = { 0,1,2,0,2,3};

	//Allocate vertex array
	//Vertex arrays encapsulate a set of generic vertex attributes and the buffers they are bound too
	//Different vertex array per mesh.
	glGenVertexArrays(1, &(m_device_quad.vertex_array));
	glBindVertexArray(m_device_quad.vertex_array);


	//Allocate VBOs for data
	glGenBuffers(1,&(m_device_quad.vbo_data));
	glGenBuffers(1,&(m_device_quad.vbo_indices));

	//Upload vertex data
	glBindBuffer(GL_ARRAY_BUFFER, m_device_quad.vbo_data);
	glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
	//Use of stride data, Array of Structures instead of Structures of Arrays
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,sizeof(vertex2_t),0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,sizeof(vertex2_t),(void*)sizeof(glm::vec3));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	//indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(GLushort), indices, GL_STATIC_DRAW);
	m_device_quad.num_indices = 6;
	//Unplug Vertex Array
	glBindVertexArray(0);
}

void LYScreenspaceRenderer::_initFBO(int w, int h) {
	GLenum FBOstatus;

	glActiveTexture(GL_TEXTURE0);

	glGenTextures(1, &m_depthTexture);
	glGenTextures(1, &m_colorTexture);
	glGenTextures(1, &m_normalTexture);
	glGenTextures(1, &m_positionTexture);
	glGenTextures(1, &m_blurDepthTexture);

	//Depth Texture Initializations
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	//Blurred Depth Texture
	glBindTexture(GL_TEXTURE_2D, m_blurDepthTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);

	//Normal Texture Initialization
	glBindTexture(GL_TEXTURE_2D, m_normalTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , w, h, 0, GL_RGBA, GL_FLOAT,0);

	//Position Texture Initialization
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , w, h, 0, GL_RGBA, GL_FLOAT,0);

	//Color Texture Initialization
	glBindTexture(GL_TEXTURE_2D, m_colorTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F , w, h, 0, GL_RGBA, GL_FLOAT, 0);

	glBindTexture(GL_TEXTURE_2D, 0);


	glGenFramebuffers(1, &m_FBO);
	glGenFramebuffers(1, &m_normalsFBO);
	glGenFramebuffers(1, &m_blurDepthFBO);

	//Create First Framebuffer Object
	glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);

	// Instruct openGL that we won't bind a color texture with the currently binded FBO
	glReadBuffer(GL_NONE);

	GLint position_loc = glGetFragDataLocation(depthShader->getProgramId(),"out_Position");
	GLint color_loc = glGetFragDataLocation(depthShader->getProgramId(),"out_Color");
	GLenum draws [2];
	draws[position_loc] = GL_COLOR_ATTACHMENT1;
	draws[color_loc] = GL_COLOR_ATTACHMENT0;

	// attach the texture to FBO depth attachment point
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depthTexture, 0);
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, draws[position_loc], m_positionTexture, 0);
	glBindTexture(GL_TEXTURE_2D, m_colorTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, draws[color_loc], m_colorTexture, 0);

	glDrawBuffers(2, draws);

	// check FBO status
	FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(FBOstatus != GL_FRAMEBUFFER_COMPLETE) {
		printf("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use m_FBO\n");
		checkFramebufferStatus(FBOstatus);
	}

	//Create Normals FBO (FBO to store Normals Data)
	glBindFramebuffer(GL_FRAMEBUFFER, m_normalsFBO);
	glReadBuffer(GL_NONE);
	GLint normal_loc = glGetFragDataLocation(normalShader->getProgramId(),"out_Normal");
	draws[normal_loc] = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, draws);
	glBindTexture(GL_TEXTURE_2D, m_normalTexture);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_normalTexture, 0);

	// check FBO status
	FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(FBOstatus != GL_FRAMEBUFFER_COMPLETE) {
		printf("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use m_normalsFBO\n");
		checkFramebufferStatus(FBOstatus);
	}

	//Create Blurred Depth FBO (FBO to store Blurred Depth Data)
	glBindFramebuffer(GL_FRAMEBUFFER, m_blurDepthFBO);
	glReadBuffer(GL_NONE);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glBindTexture(GL_TEXTURE_2D, m_blurDepthTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_blurDepthTexture, 0);

	// check FBO status
	FBOstatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(FBOstatus != GL_FRAMEBUFFER_COMPLETE) {
		printf("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use m_blurDepthFBO\n");
		checkFramebufferStatus(FBOstatus);
	}

	// switch back to window-system-provided framebuffer
	glClear(GL_DEPTH_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void LYScreenspaceRenderer::_drawCollider()
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);
		uint vbo = haptic_interface->getVBO();
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), 0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
		glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
		uint ib = haptic_interface->getIB();
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
		int numIndices = 2;
		glDrawElements(GL_POINTS, numIndices, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

}

void LYScreenspaceRenderer::_drawPoints(LYMesh *m_mesh)
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	uint vbo = m_mesh->getVBO();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
	uint ib = m_mesh->getIB();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	size_t numIndices = m_mesh->getNumIndices();
	glDrawElements(GL_POINTS, numIndices, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}

void LYScreenspaceRenderer::_drawTriangles(LYMesh *m_mesh)
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	uint vbo = m_mesh->getVBO();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)12);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)24);
	glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(LYVertex), (const GLvoid*)32);
	uint ib = m_mesh->getIB();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);

	size_t numIndices = m_mesh->getNumIndices();
	glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
}

void LYScreenspaceRenderer::displayTriangleMesh(LYMesh *m_mesh, DisplayMode mode = DISPLAY_TOTAL)
{
	_setTextures();
	_bindFBO(m_FBO);

	//Draw Particles
	glEnable(GL_POINT_SPRITE_ARB);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	depthShader->useShader();
	glm::mat4 modelMatrix = glm::mat4();
	glm::mat4 modelViewMatrix = glm::mat4();
	glm::mat4 inverse_transposed = glm::mat4();

	modelMatrix =  m_mesh->getModelMatrix();
	modelViewMatrix = m_camera->getViewMatrix() * modelMatrix;
	inverse_transposed = glm::inverse(modelViewMatrix);
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointScale"), m_pointScale);
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointRadius"), m_pointRadius );
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &modelViewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);

	_drawTriangles(m_mesh);

	glDisable(GL_POINT_SPRITE_ARB);

	glDisable(GL_DEPTH_TEST);
}

void LYScreenspaceRenderer::displayPointMesh(LYMesh *m_mesh, DisplayMode mode = DISPLAY_TOTAL)
{
	_setTextures();
	_bindFBO(m_FBO);

	//Draw Particles
	glEnable(GL_POINT_SPRITE_ARB);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	depthShader->useShader();
	glm::mat4 modelMatrix = glm::mat4();
	glm::mat4 modelViewMatrix = glm::mat4();
	glm::mat4 inverse_transposed = glm::mat4();

	modelMatrix =  m_mesh->getModelMatrix();
	modelViewMatrix = m_camera->getViewMatrix() * modelMatrix;
	inverse_transposed = glm::inverse(modelViewMatrix);
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointScale"), m_pointScale);
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointRadius"), m_pointRadius );
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &modelViewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);

	_drawPoints(m_mesh);

	glDisable(GL_POINT_SPRITE_ARB);

	glDisable(GL_DEPTH_TEST);
}

void LYScreenspaceRenderer::display(DisplayMode mode  = DISPLAY_TOTAL)
{
	//Render Attributes to Texture
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_setTextures();
	_bindFBO(m_FBO);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Draw Particles
	glEnable(GL_POINT_SPRITE_ARB);
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	//glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	depthShader->useShader();

	glm::mat4 modelMatrix = glm::mat4();
	glm::mat4 modelViewMatrix = glm::mat4();
	glm::mat3 inverse_transposed = glm::mat3();


	//////////////////////////////////////////////////////////////////////////
	// RENDER ALL THE POINT BASED MODELS
	// The "m_objects" array contains all the meshes that need to be displayed
	// 
	//
	// FIRST PASS
	//////////////////////////////////////////////////////////////////////////
	for (std::vector<LYMesh*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		LYMesh* mesh = (*it);
		modelMatrix =  mesh->getModelMatrix();
		modelViewMatrix = m_camera->getViewMatrix() * modelMatrix;
		inverse_transposed = glm::inverse(glm::transpose(glm::mat3(modelViewMatrix)));
		glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointScale"), m_pointScale);
		glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointRadius"), m_pointRadius );
		glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &modelViewMatrix[0][0]);
		glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
		glUniformMatrix3fv(glGetUniformLocation(depthShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);

		(mesh->getRenderPoints()) ? _drawPoints(mesh) : _drawTriangles(mesh);
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//
	//  The HIP and the Proxy objects are displayed as points within the
	//
	//  FIRST PASS
	//
	//////////////////////////////////////////////////////////////////////////
	modelMatrix = haptic_interface->getHIPMatrix();
	modelViewMatrix = m_camera->getViewMatrix() * modelMatrix;
	inverse_transposed = glm::inverse(glm::transpose(glm::mat3(modelViewMatrix)));
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &modelViewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix3fv(glGetUniformLocation(depthShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);

	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointScale"), m_camera->getHeight() / tanf(m_camera->getFOV()*0.5f*(float)M_PI/180.0f) );
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointRadius"), haptic_interface->getSize()*8.0f);

	_drawPoints(haptic_interface->getHIPObject());

	modelMatrix = haptic_interface->getProxyMatrix();
	modelViewMatrix = m_camera->getViewMatrix() * modelMatrix;
	inverse_transposed = glm::inverse(glm::transpose(glm::mat3(modelViewMatrix)));
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &modelViewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix3fv(glGetUniformLocation(depthShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);

	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointScale"), m_camera->getHeight() / tanf(m_camera->getFOV()*0.5f*(float)M_PI/180.0f) );
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointRadius"), haptic_interface->getSize()*8.0f);

	_drawPoints(haptic_interface->getProxyObject());

	//////////////////////////////////////////////////////////////////////////
	// END OF FIRST PASS
	//////////////////////////////////////////////////////////////////////////


	glDisable(GL_POINT_SPRITE_ARB);

	glDisable(GL_DEPTH_TEST);


	//////////////////////////////////////////////////////////////////////////
	//
	//  START: SECOND PASS
	//  Depth buffer smoothing pass
	//
	//////////////////////////////////////////////////////////////////////////

	//Blur Depth Texture
	_setTextures();
	_bindFBO(m_blurDepthFBO);

	blurDepthShader->useShader();
	glBindVertexArray(m_device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);

	glActiveTexture(GL_TEXTURE11);
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glUniform1i(glGetUniformLocation(blurDepthShader->getProgramId(), "u_Depthtex"),11);
	glUniform1f( glGetUniformLocation(blurDepthShader->getProgramId(), "u_Far"), m_camera->getFar() );
	glUniform1f( glGetUniformLocation(blurDepthShader->getProgramId(), "u_Near"), m_camera->getNear() );
	glUniform1f(glGetUniformLocation(blurDepthShader->getProgramId(), "u_Width"), (GLfloat) 1.0 / m_camera->getWidth());
	glUniform1f(glGetUniformLocation(blurDepthShader->getProgramId(), "u_Height"), (GLfloat) 1.0 / m_camera->getHeight());

	glDrawElements(GL_TRIANGLES, m_device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindVertexArray(0);

	//////////////////////////////////////////////////////////////////////////
	//  END OF SECOND PASS
	//////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////////////////
	//
	//  START: THIRD PASS
	//  The values of the normals are calculated using the smoothed depth values
	//
	//  Normals to Texture from Depth Texture
	//
	///////////////////////////////////////////////////////////////////////////////
	_setTextures();
	_bindFBO(m_normalsFBO);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	normalShader->useShader();
	glBindVertexArray(m_device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);

	glActiveTexture(GL_TEXTURE11);
	glBindTexture(GL_TEXTURE_2D, m_blurDepthTexture);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(), "u_Depthtex"),11);
	glActiveTexture(GL_TEXTURE12);
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(), "u_Positiontex"),12);

	glUniform1f( glGetUniformLocation(normalShader->getProgramId(), "u_Far"), m_camera->getFar() );
	glUniform1f( glGetUniformLocation(normalShader->getProgramId(), "u_Near"), m_camera->getNear() );
	glUniform1f( glGetUniformLocation(normalShader->getProgramId(), "u_Width"), (GLfloat) 1.0/m_camera->getWidth() );
	glUniform1f( glGetUniformLocation(normalShader->getProgramId(), "u_Height"), (GLfloat) 1.0/m_camera->getHeight() );
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);
	glm::mat4 inverse_projectiond = glm::inverse(m_camera->getProjection());
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_InvProj"),1,GL_FALSE,&inverse_projectiond[0][0]);

	glDrawElements(GL_TRIANGLES, m_device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//////////////////////////////////////////////////////////////////////////
	// END OF THIRD PASS
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	//
	// FINAL PASS
	// Render the screen sized quad, combining all the attributes of the model
	// and applying the illumination model (Phong in this case)
	//
	// Draw Full Screen Quad
	//////////////////////////////////////////////////////////////////////////
	_setTextures();
	totalShader->useShader();
	glBindVertexArray(m_device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, m_blurDepthTexture);
	glUniform1i(glGetUniformLocation(totalShader->getProgramId(), "u_Depthtex"),6);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_normalTexture);
	glUniform1i(glGetUniformLocation(totalShader->getProgramId(), "u_Normaltex"),1);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, m_colorTexture);
	glUniform1i(glGetUniformLocation(totalShader->getProgramId(), "u_Colortex"),2);
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);
	glUniform1i(glGetUniformLocation(totalShader->getProgramId(), "u_Positiontex"),3);

	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &modelViewMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_InvTrans"),1,GL_FALSE, &inverse_transposed[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_InvProj"),1,GL_FALSE,&inverse_projectiond[0][0]);
	glUniform4fv( glGetUniformLocation(totalShader->getProgramId(), "lightDir"), 1, &m_camera->getLightDir()[0]);

	glUniform1f( glGetUniformLocation(totalShader->getProgramId(), "u_Far"), m_camera->getFar());
	glUniform1f( glGetUniformLocation(totalShader->getProgramId(), "u_Near"), m_camera->getNear());
	glUniform1f( glGetUniformLocation(totalShader->getProgramId(), "u_Aspect"), (GLfloat) m_camera->getWidth()/m_camera->getHeight());
	glUniform1i( glGetUniformLocation(totalShader->getProgramId(), "u_DisplayType"), mode);

	glDrawElements(GL_TRIANGLES, m_device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);

	//////////////////////////////////////////////////////////////////////////
	//  END OF FINAL PASS
	//////////////////////////////////////////////////////////////////////////

	// Clear the meshes array for next frame
	m_objects.clear();
}

void LYScreenspaceRenderer::dumpIntoPdb(std::string outputFilename)
{
	for(std::vector<LYMesh*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		std::ofstream outFile;
		outFile.open((outputFilename+".pdb").c_str());
		std::vector<LYVertex> *vertices = (*it)->getVertices();
		for(unsigned int i=0; i< vertices->size(); i++)
		{
			float localSCP[3] = {vertices->at(i).m_pos.x, vertices->at(i).m_pos.y, vertices->at(i).m_pos.z}; //get centre values of points

			outFile << "ATOM "<<std::setw(6)<<i+1<<" O   HOH     1    ";
			outFile << std::setw(8);
			outFile << std::setprecision(3);
			outFile.setf(std::ios::fixed);
			outFile <<std::setw(8)<<std::setprecision(3)<<localSCP[0]<<std::setw(8)<<std::setprecision(3)<<localSCP[1]<<std::setw(8)<<std::setprecision(3)<<localSCP[2];
			outFile << "  1.00 67.53           O  "<<std::endl;

		}
		outFile << "END                                                                             " << std::endl;
		outFile.close();
	}

}

void LYScreenspaceRenderer::setPointRadius( float r )
{
	m_pointRadius = r;
}

void LYScreenspaceRenderer::setCamera( LYCamera *c )
{
	m_camera = c;
}

void LYScreenspaceRenderer::setCollider(LYHapticInterface* haptic)
{
	haptic_interface = haptic;
}

void LYScreenspaceRenderer::setPointDiv( int d )
{
	m_pointDiv = d;
}

void LYScreenspaceRenderer::setPointScale( float s )
{
	m_pointScale = s;
}
