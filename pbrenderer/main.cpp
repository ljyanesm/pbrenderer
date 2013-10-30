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

#include <config4cpp/Configuration.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "LYHapticDevice.h"

#include "LYWorld.h"
#include "LYMesh.h"
#include "LYCamera.h"
#include "LYScreenspaceRenderer.h"
#include "LYSpatialHash.h"
#include "LYKeyboardDevice.h"
#include "LYTimer.h"
#include "LYPLYLoader.h"

int width = 1024;
int height = 768;

float	pointRadius = 0.01f;
float	influenceRadius = 0.2f;
int		pointDiv = 0;

const float NEARP = 1.0f;
const float FARP = 1000.0f;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -1};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -1};
float camera_rot_lag[] = {0, 0, 0};

bool wireframe = false;
bool displayEnabled = true;
bool bPause = false;

const float inertia = 0.1f;

LYScreenspaceRenderer::DisplayMode mode = LYScreenspaceRenderer::DISPLAY_DIFFUSE_SPEC;
bool mouseMode = 0;

glm::mat4 viewMatrix;
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

// Variables to load from the cfg file
///////////////////////////////////////////////////////
std::string modelFile;
LYHapticInterface::LYDEVICE_TYPE deviceType;
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


///////////////////////////////////////////////////////
// GLUT navigation variables
///////////////////////////////////////////////////////
static int GLUTmouse[2] = { 0, 0 };
static int GLUTbutton[3] = { 0, 0, 0 };
static int GLUTmodifiers = 0;
// Display variables
static int scaling = 0;
static int translating = 0;
static int rotating = 0;
static float scale = 1.0;
static float viewScale = 1.0;
static float center[3] = { 0.0, 0.0, 0.0 };
static float rotation[3] = { 0.0, 0.0, 0.0 };
static float translation[3] = { 0.0, 0.0, -30.0 };
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

	setlocale(LC_ALL, "");
	config4cpp::Configuration *cfg = config4cpp::Configuration::create();
	try
	{
		cfg->parse("./app.cfg");
		modelFile = cfg->lookupString("", "filename");
		deviceType = (LYHapticInterface::LYDEVICE_TYPE) cfg->lookupInt("", "device");
		influenceRadius = (float) cfg->lookupFloat("", "influenceRadius");
		pointRadius = (float) cfg->lookupFloat("", "pointRadius");
	}
	catch (const config4cpp::ConfigurationException &e)
	{
		printf("%s", e.c_str());
	}
	
	m_pMesh = new LYMesh();

	m_pCamera = new LYCamera(width, height);

	if (modelFile.empty()) modelFile = argv[1];
	m_pMesh->LoadPoints(modelFile);

	screenspace_renderer = new LYScreenspaceRenderer(m_pCamera);
	space_handler = new LYSpatialHash(m_pMesh->getVBO(), (uint) m_pMesh->getNumVertices(), make_uint3(256, 256, 256));
	if (deviceType == LYHapticInterface::KEYBOARD_DEVICE) haptic_interface = new LYKeyboardDevice(space_handler);
	if (deviceType == LYHapticInterface::HAPTIC_DEVICE) haptic_interface = new LYHapticDevice(space_handler);
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
	// Invert y coordinate
	y = height - y;

	// Process mouse button event
	rotating = (button == GLUT_LEFT_BUTTON);
	scaling = (button == GLUT_MIDDLE_BUTTON);
	translating = (button == GLUT_RIGHT_BUTTON);

	// Remember button state 
	int b = (button == GLUT_LEFT_BUTTON) ? 0 : ((button == GLUT_MIDDLE_BUTTON) ? 1 : 2);
	GLUTbutton[b] = (state == GLUT_DOWN) ? 1 : 0;

	// Remember modifiers 
	GLUTmodifiers = glutGetModifiers();

	// Remember mouse position 
	GLUTmouse[0] = x;
	GLUTmouse[1] = y;

	glutPostRedisplay();
}

void motion(int x, int y)
{
	// Invert y coordinate
	y = height - y;

	// Process mouse motion event
	if (rotating) {
		// Rotate model
		rotation[0] += -0.5f * (y - GLUTmouse[1]);
		rotation[2] +=  0.5f * (x - GLUTmouse[0]);
	}
	else if (scaling) {
		// Scale window
		scale *= exp(2.0f * (float) (y - GLUTmouse[1]) / (float) height);
	}
	else if (translating) {
		// Translate window
		translation[0] += 10.0f * (float) (x - GLUTmouse[0]) / (float) width;
		translation[1] += 10.0f * (float) (y - GLUTmouse[1]) / (float) height;
	}

	// Remember mouse position 
	GLUTmouse[0] = x;
	GLUTmouse[1] = y;
	
	glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int x, int y)
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
	case '[':
		pointDiv += 1;
		screenspace_renderer->setPointDiv(1 << pointDiv);
		break;
	case ']':
		pointDiv -= 1;
		if (pointDiv < 0) pointDiv = 0;
		screenspace_renderer->setPointDiv(1 << pointDiv);
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
	case 'f':
		space_handler->dump();
		break;
	case 'v':
		mouseMode = !mouseMode;
		break;
	case ',':
		influenceRadius -= 0.01f;
		space_handler->setInfluenceRadius(influenceRadius);
		haptic_interface->setSize(influenceRadius);
		break;
	case '.':
		influenceRadius += 0.01f;
		space_handler->setInfluenceRadius(influenceRadius);
		haptic_interface->setSize(influenceRadius);
		break;

	case 'w':
		// Move collider up
		translation[3] += 1.0f;
		break;
	case 'a':
		// Move collider left
		pos = haptic_interface->getPosition();
		pos.x -= haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		break;
	case 's':
		translation[3] -= 1.0f;
		break;
	case 'd':
		// Move collider right
		pos = haptic_interface->getPosition();
		pos.x += haptic_interface->getSpeed();
		haptic_interface->setPosition(pos);
		break;
	case 'z':
		viewScale += 0.01f;
		break;
	case 'c':
		viewScale -= 0.01f;
		break;
	case 'I':
		LYHapticInterface *new_interface;
		if ( haptic_interface->getDeviceType() == LYHapticInterface::KEYBOARD_DEVICE)
			new_interface = new LYHapticDevice(space_handler);
		else
			new_interface = new LYKeyboardDevice(space_handler);
		haptic_interface = new_interface;
		delete haptic_interface;

		break;
	}

	// Remember mouse position 
	GLUTmouse[0] = x;
	GLUTmouse[1] = height - y;

	// Remember modifiers 
	GLUTmodifiers = glutGetModifiers();

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
	viewMatrix = glm::mat4();
	viewMatrix = glm::scale(viewMatrix, glm::vec3(viewScale));
	viewMatrix = glm::translate(viewMatrix, glm::vec3(translation[0], translation[1], 0.0f));

	m_pCamera->setViewMatrix(viewMatrix);

	glm::mat4 modelMatrix = glm::mat4();
	modelMatrix = glm::translate(modelMatrix,  glm::vec3(translation[0], translation[1], translation[2]));
	modelMatrix = glm::scale(modelMatrix, glm::vec3(scale));
	modelMatrix = glm::rotate(modelMatrix, rotation[0], glm::vec3(1,0,0));
	modelMatrix = glm::rotate(modelMatrix, rotation[1], glm::vec3(0,1,0));
	modelMatrix = glm::rotate(modelMatrix, rotation[2], glm::vec3(0,0,1));
	modelMatrix = glm::translate(modelMatrix, -m_pMesh->getModelCentre());
	m_pMesh->setModelMatrix(modelMatrix);
	screenspace_renderer->display(m_pMesh, mode);

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
