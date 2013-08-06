#include "Screenspace_Renderer.h"

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

Screenspace_Renderer::Screenspace_Renderer()
{
	_initShaders();
	_initFBO(m_camera->getWidth(), m_camera->getHeight());
	_initQuad();
}

Screenspace_Renderer::Screenspace_Renderer(Mesh *m, LYCamera *c) :
	m_mesh(m), 
	m_camera(c)
{
	_initShaders();
	_initFBO(m_camera->getWidth(), m_camera->getHeight());
	_initQuad();
}


Screenspace_Renderer::~Screenspace_Renderer(void)
{
}

void Screenspace_Renderer::_initShaders() 
{
	mainShader = new LYshader("./shaders/mainShader.vs", "./shaders/mainShader.frag", "Color");
	depthShader = new LYshader("./shaders/depth_pass.vs", "./shaders/depth_pass.frag", "Color");
	normalShader = new LYshader("./shaders/normal_pass.vs", "./shaders/normal_pass.frag", "Texcoord");
	blurDepthShader = new LYshader("./shaders/blur_pass.vs", "./shaders/blur_pass.frag", "Texcoord");
	totalShader = new LYshader("./shaders/shader.vs", "./shaders/shader.frag", "Texcoord");
}

void Screenspace_Renderer::_setTextures() {
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,0); 
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glColorMask(true,true,true,true);
	glDisable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
}

void Screenspace_Renderer::_bindFBO(GLuint FBO) {
	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D,0); //Bad mojo to unbind the framebuffer using the texture
	glBindFramebuffer(GL_FRAMEBUFFER, FBO);
	glClear(GL_DEPTH_BUFFER_BIT);
	//glColorMask(false,false,false,false);
	glEnable(GL_DEPTH_TEST);
}

void Screenspace_Renderer::_initQuad() {
	float size = 1.0f;
	vertex2_t verts [] = {  {glm::vec3(-size,size,-0.2f),glm::vec2(0,1)},
	{glm::vec3(-size,-size,-0.2f),glm::vec2(0,0)},
	{glm::vec3(size,-size,-0.2f),glm::vec2(1,0)},
	{glm::vec3(size,size,-0.2f),glm::vec2(1,1)}
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

void Screenspace_Renderer::_initFBO(int w, int h) {
	GLenum FBOstatus;

	glActiveTexture(GL_TEXTURE10);

	glGenTextures(1, &m_depthTexture);
	glGenTextures(1, &m_colorTexture);
	glGenTextures(1, &m_normalTexture);
	glGenTextures(1, &m_positionTexture);
	//glGenTextures(1, &m_backgroundTexture);
	glGenTextures(1, &m_blurDepthTexture);

	//Depth Texture Initializations
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	//Blurred Depth Texture
	glBindTexture(GL_TEXTURE_2D, m_blurDepthTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);

	//Normal Texture Initialization
	glBindTexture(GL_TEXTURE_2D, m_normalTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F , w, h, 0, GL_RGBA, GL_FLOAT,0);

	//Position Texture Initialization
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F , w, h, 0, GL_RGBA, GL_FLOAT,0);

	//Color Texture Initialization
	glBindTexture(GL_TEXTURE_2D, m_colorTexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F , w, h, 0, GL_RGBA, GL_FLOAT, 0);

	////Background Texture Initialization
	//glBindTexture(GL_TEXTURE_2D, m_backgroundTexture);

	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	//glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	//glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB , w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

	//glBindTexture(GL_TEXTURE_2D, 0);

	//m_backgroundTexData = (unsigned char *)malloc(sizeof(unsigned char) * 4 * w * h);

	glGenFramebuffers(1, &m_FBO);
	glGenFramebuffers(1, &m_normalsFBO);
	//glGenFramebuffers(1, &m_backgroundFBO);
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

	glDrawBuffers(2, draws);

	// attach the texture to FBO depth attachment point
	int test = GL_COLOR_ATTACHMENT0;
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_depthTexture, 0);
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, draws[position_loc], m_positionTexture, 0);
	glBindTexture(GL_TEXTURE_2D, m_colorTexture);    
	glFramebufferTexture(GL_FRAMEBUFFER, draws[color_loc], m_colorTexture, 0);

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

void Screenspace_Renderer::_drawPoints()
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	for (unsigned int i = 0 ; i < m_mesh->getEntries().size() ; i++) {
		glBindBuffer(GL_ARRAY_BUFFER, m_mesh->getEntries()[i].VB);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)20);
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)32);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_mesh->getEntries()[i].IB);

		const unsigned int MaterialIndex = m_mesh->getEntries()[i].MaterialIndex;

		if (MaterialIndex < m_mesh->getTextures().size() && m_mesh->getTextures()[MaterialIndex]) {
			m_mesh->getTextures()[MaterialIndex]->Bind(GL_TEXTURE0);
		}

		glDrawElements(GL_POINTS, m_mesh->getEntries()[i].NumIndices, GL_UNSIGNED_INT, 0);
	}

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

}

void Screenspace_Renderer::display(DisplayMode mode /* = PARTICLE_POINTS */)
{
	glm::mat4 inverse_transposed = glm::inverse(m_camera->getModelView());
	//Render Attributes to Texture
	glClearColor(0.75, 0.75, 0.75, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_setTextures();
	_bindFBO(m_FBO);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Draw Particles
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	depthShader->useShader();
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointScale"), m_camera->getHeight() / tanf(m_camera->getFOV()*0.5f*(float)M_PI/180.0f) );
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "pointRadius"), m_pointRadius );
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "u_Far"), m_camera->getFar() );
	glUniform1f( glGetUniformLocation(depthShader->getProgramId(), "u_Near"), m_camera->getNear() );
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_ModelView"),1,GL_FALSE,&m_camera->getModelView()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(depthShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);

	_drawPoints();
	glDisable(GL_POINT_SPRITE_ARB);

	glEnable(GL_DEPTH_TEST);

	//Blur Depth Texture
	_setTextures();
	_bindFBO(m_blurDepthFBO);

	blurDepthShader->useShader();
	glBindVertexArray(m_device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);

	glEnable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE11);
	glBindTexture(GL_TEXTURE_2D, m_depthTexture);
	glUniform1i(glGetUniformLocation(blurDepthShader->getProgramId(), "u_Depthtex"),11);
	glUniform1f( glGetUniformLocation(blurDepthShader->getProgramId(), "u_Far"), m_camera->getFar() );
	glUniform1f( glGetUniformLocation(blurDepthShader->getProgramId(), "u_Near"), m_camera->getNear() );

	glDrawElements(GL_TRIANGLES, m_device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindVertexArray(0);

	//Write Normals to Texture from Depth Texture
	_setTextures();
	_bindFBO(m_normalsFBO);

	normalShader->useShader();
	glBindVertexArray(m_device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);

	glEnable(GL_TEXTURE_2D);

	glActiveTexture(GL_TEXTURE11);
	glBindTexture(GL_TEXTURE_2D, m_blurDepthTexture);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(), "u_Depthtex"),11);
	glActiveTexture(GL_TEXTURE12);
	glBindTexture(GL_TEXTURE_2D, m_positionTexture);
	glUniform1i(glGetUniformLocation(normalShader->getProgramId(), "u_Positiontex"),12);

	glUniform1f( glGetUniformLocation(normalShader->getProgramId(), "u_Far"), m_camera->getFar() );
	glUniform1f( glGetUniformLocation(normalShader->getProgramId(), "u_Near"), m_camera->getNear() );
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_InvTrans"),1,GL_FALSE,&inverse_transposed[0][0]);
	glm::mat4 inverse_projectiond = glm::inverse(m_camera->getProjection());
	glUniformMatrix4fv(glGetUniformLocation(normalShader->getProgramId(),"u_InvProj"),1,GL_FALSE,&inverse_projectiond[0][0]);

	glDrawElements(GL_TRIANGLES, m_device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindVertexArray(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//Draw Full Screen Quad
	_setTextures();
	totalShader->useShader();
	glBindVertexArray(m_device_quad.vertex_array);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_device_quad.vbo_indices);

	glEnable(GL_TEXTURE_2D);

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
	//glActiveTexture(GL_TEXTURE4);
	//glBindTexture(GL_TEXTURE_2D, m_backgroundTexture);
	//glUniform1i(glGetUniformLocation(totalShader->getProgramId(), "u_Backgroundtex"),4);

	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_ModelView"),1,GL_FALSE, &m_camera->getModelView()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_Persp"),1,GL_FALSE,&m_camera->getProjection()[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_InvTrans"),1,GL_FALSE, &inverse_transposed[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(totalShader->getProgramId(),"u_InvProj"),1,GL_FALSE,&inverse_projectiond[0][0]);

	glUniform1f( glGetUniformLocation(totalShader->getProgramId(), "u_Far"), m_camera->getFar());
	glUniform1f( glGetUniformLocation(totalShader->getProgramId(), "u_Near"), m_camera->getNear());
	glUniform1i( glGetUniformLocation(totalShader->getProgramId(), "u_DisplayType"), mode);

	glDrawElements(GL_TRIANGLES, m_device_quad.num_indices, GL_UNSIGNED_SHORT,0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
}
