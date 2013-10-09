#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/projection.hpp>

#include <vector_functions.h>
#include <vector_types.h>

#include "LYHapticDevice.h"

#include "LYWorld.h"
#include "LYMesh.h"
#include "LYCamera.h"
#include "LYScreenspaceRenderer.h"
#include "LYSpatialHash.h"
#include "LYHapticKeyboard.h"
#include "LYTimer.h"

int width = 1024;
int height = 768;

float pointRadius = 0.01f;

const float NEARP = 0.1f;
const float FARP = 100.0f;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, -3};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};

bool wireframe = false;
bool displayEnabled = true;
bool bPause = false;

const float inertia = 0.1f;

LYScreenspaceRenderer::DisplayMode mode = LYScreenspaceRenderer::DISPLAY_DIFFUSE_SPEC;
bool mouseMode = 0;

glm::mat4 mv;
glm::mat4 p;

LYWorld m_pWorld;
// Information below this line has to move to LYWorld.
///////////////////////////////////////////////////////
LYScreenspaceRenderer *screenspace_renderer;
LYSpaceHandler *space_handler;
LYMesh* m_pMesh;
LYCamera *m_pCamera;
LYHapticInterface *haptic_interface;
LYHapticDevice* haptics;
///////////////////////////////////////////////////////

// Timer variables to measure performance
///////////////////////////////////////////////////////
clock_t startTimer;
clock_t endTimer;
char fps_string[120];
///////////////////////////////////////////////////////

// FPS Control variables for rendering
///////////////////////////////////////////////////////
const int FRAMES_PER_SECOND = 30;
const int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
///////////////////////////////////////////////////////

enum {M_VIEW = 0, M_MOVE};

void cudaInit(int argc, char **argv)
{    
	int devID;

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	devID = findCudaDevice(argc, (const char **)argv);
	if (devID < 0)
	{
		printf("No CUDA Capable devices found, exiting...\n");
		exit(EXIT_SUCCESS);
	}
	checkCudaErrors(cudaSetDevice(devID));
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
}


void initCUDA(int argc, char **argv)
{
	cudaInit(argc, argv);
}
// initialize OpenGL
void initGL(int *argc, char **argv){
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow(fps_string);

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(-1);
	}

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.75, 0.75, 0.75, 1);

	m_pMesh = new LYMesh();
	m_pCamera = new LYCamera(width, height);

	m_pMesh->LoadPoints("example.ply");

	screenspace_renderer = new LYScreenspaceRenderer(m_pMesh, m_pCamera);
	space_handler = new LYSpatialHash(m_pMesh->getEntries()->at(0).VB, m_pMesh->getEntries()->at(0).numVertices, make_uint3(256, 256, 256));
	haptic_interface = new LYHapticKeyboard(space_handler);
//	haptic_interface = new LYHapticDevice(space_handler);
	screenspace_renderer->setCollider(haptic_interface);

	glutReportErrors();
}

void reshape(int w, int h)
{
	m_pCamera->perspProjection(width, height, 60.0f, NEARP, FARP);
	p = glm::mat4();
	p = glm::perspective(60.0f, (float) width/ (float) height, NEARP, FARP);

	glViewport(0, 0, width, height);
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

	switch (mouseMode){
	case M_VIEW:
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
		break;
	case M_MOVE:
		float3 p = haptic_interface->getPosition();
		glm::vec4 r, v;
		if (buttonState == 1) {
			v.x = dx * haptic_interface->getSpeed();
			v.y = -dy * haptic_interface->getSpeed();
			r = v * m_pCamera->getModelView();
			p.x += r.x;
			p.y += r.y;
			p.z += r.z;
		} else {
			v.z = dy * haptic_interface->getSpeed();
			r = v * m_pCamera->getModelView();
			p.x += r.x;
			p.y += r.y;
			p.z += r.z;
		}

		haptic_interface->setPosition(p);
		break;
	}

	ox = x; oy = y;

	glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
	float3 pos;
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
	case 'p':
		mode = (LYScreenspaceRenderer::DisplayMode)
			((mode+ 1) % LYScreenspaceRenderer::NUM_DISPLAY_MODES);
		break;
	case 'r':
		displayEnabled = !displayEnabled;
		break;
	case '+':
		pointRadius += 0.001f;
		break;
	case '-':
		pointRadius -= 0.001f;
		break;
	case GLUT_KEY_UP:
		camera_trans[2] += 0.5f;
		break;
	case GLUT_KEY_DOWN:
		camera_trans[2] -= 0.5f;
		break;
	case 'D':
		screenspace_renderer->dumpIntoPdb("bunny");
		break;
	case 'f':
		space_handler->dump();
		break;
	case 'v':
		mouseMode = !mouseMode;
		break;

	case 'w':
		// Move collider up
		pos = haptic_interface->getPosition();
		pos.y += haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);

		break;
	case 'a':
		// Move collider left
		pos = haptic_interface->getPosition();
		pos.x -= haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		break;
	case 's':
		// Move collider down
		pos = haptic_interface->getPosition();
		pos.y += -haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		break;
	case 'd':
		// Move collider right
		pos = haptic_interface->getPosition();
		pos.x += haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		break;
	case 'z':
		pos = haptic_interface->getPosition();
		pos.z += haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		// Move collider in
		break;
	case 'c':
		pos = haptic_interface->getPosition();
		pos.z += -haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		// Move collider out
		break;
	case 'I':
		LYHapticInterface *new_interface;
		if ( haptic_interface->getDeviceType() == LYHapticInterface::KEYBOARD_DEVICE)
			new_interface = new LYHapticDevice(space_handler);
		else
			new_interface = new LYHapticKeyboard(space_handler);
		haptic_interface = new_interface;
		delete haptic_interface;

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

DWORD next_game_tick = GetTickCount();
int sleep_time = 0;

void display()
{
	Sleep(20);
	LYTimer t(true);
	LYTimer spaceHandler_timer(true);
	// update the simulation
	if (!bPause)
	{
		space_handler->update();
		haptic_interface->setPosition(haptic_interface->getPosition());
		if (haptic_interface->getDeviceType() == LYHapticInterface::KEYBOARD_DEVICE){
			float3 force = haptic_interface->calculateFeedbackUpdateProxy();
		}
		screenspace_renderer->setPointRadius(pointRadius);
		spaceHandler_timer.Stop();
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

	screenspace_renderer->display(mode);

	glutSwapBuffers();
	glutReportErrors();

	sprintf(fps_string, "Point-Based Rendering \t FPS: %5.3f \t SpaceHandler FPS: %f", 1000.0f / (t.Elapsed()), 1000.0f / (spaceHandler_timer.Elapsed()));
	glutSetWindowTitle(fps_string);

}

int main(int argc, char **argv)
{
	initCUDA(argc, argv);
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
