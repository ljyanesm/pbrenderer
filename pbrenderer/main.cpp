#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/projection.hpp>

#include "mesh.h"
#include "LYCamera.h"

int width = 1024;
int height = 768;

float particleRadius = 0.01f;

const float NEARP = 0.1f;
const float FARP = 100.0f;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, -3};
float camera_trans_lag[] = {0, 0, 0};
float camera_rot_lag[] = {0, 0, 0};

bool wireframe = false;
bool displayEnabled = true;
bool bPause = false;

const float inertia = 0.1f;

glm::mat4 mv;
glm::mat4 p;

Mesh* m_pMesh;
LYCamera *m_pCamera;

// initialize OpenGL
void initGL(int *argc, char **argv){
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("Point-Based Renderer");

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(-1);
	}

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
	
	m_pMesh = new Mesh();
	m_pCamera = new LYCamera();

	m_pMesh->LoadMesh("bunny-color.ply");
	glutReportErrors();
}

void reshape(int w, int h)
{
	m_pCamera->perspProjection(w, h, 60.0f, NEARP, FARP);
	p = glm::mat4();
	p = glm::perspective(60.0f, (float) w/ (float) h, NEARP, FARP);

	glViewport(0, 0, w, h);

}

void mouse(int button, int state, int x, int y)
{
	int mods;

	if (state == GLUT_DOWN)
		buttonState |= 1<<button;
	else if (state == GLUT_UP)
		buttonState = 0;

	mods = glutGetModifiers();
	if (mods & GLUT_ACTIVE_SHIFT) {
		buttonState = 2;
	} else if (mods & GLUT_ACTIVE_CTRL) {
		buttonState = 3;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	if (buttonState == 3) {
		// left+middle = zoom
		camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
	}
	else if (buttonState & 2) {
		// middle = translate
		camera_trans[0] += dx / 100.0f;
		camera_trans[1] -= dy / 100.0f;
	}
	else if (buttonState & 1) {
		// left = rotate
		camera_rot[0] += dy / 5.0f;
		camera_rot[1] += dx / 5.0f;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case ' ':
		bPause = !bPause;
		break;
	case 13:		// ENTER key

		break;
	case '\033':
	case 'q':
		exit(0);
		break;
	case 'r':
		displayEnabled = !displayEnabled;
		break;
	case 'w':
		wireframe = !wireframe;
		break;
	case '+':
		particleRadius += 0.001f;
		break;
	case '-':
		particleRadius -= 0.001f;
		break;
	case GLUT_KEY_UP:
		camera_trans[2] += 0.5f;
		break;
	case GLUT_KEY_DOWN:
		camera_trans[2] -= 0.5f;
		break;
	}
	glutPostRedisplay();
}

void special(int k, int x, int y)
{
}

void cleanup()
{
}

void idle(void)
{
	glutPostRedisplay();
}

void display()
{
	// update the simulation
	if (!bPause)
	{
	}

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	mv = glm::mat4();
	// view transform
	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}
	mv = glm::translate(glm::mat4(), camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	glm::mat4 rotX = glm::rotate(camera_rot_lag[0], 1.0f, 0.0f, 0.0f);
	glm::mat4 rotY = glm::rotate(camera_rot_lag[1], 0.0f, 1.0f, 0.0f);
	mv = mv * rotX * rotY;
	m_pCamera->setModelView(mv);
	// cube
	glColor3f(1.0, 1.0, 1.0);

	m_pMesh->Render2(p, mv, particleRadius);

	glutSwapBuffers();
	glutReportErrors();
}

int main(int argc, char **argv)
{
	initGL(&argc, argv);

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(key);
	glutSpecialFunc(special);
	glutIdleFunc(idle);

	atexit(cleanup);

	glutMainLoop();

	return EXIT_SUCCESS;
}