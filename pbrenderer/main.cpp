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

int width = 1600;
int height = 900;

float	pointRadius = 0.01f;
float	influenceRadius = 0.2f;
int		pointDiv = 0;

const float NEARP = 0.1f;
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
glm::mat4 modelViewMatrix;
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
StopWatchInterface *hapticTimer = NULL;
StopWatchInterface *graphicsTimer = NULL;

const int FRAMES_PER_SECOND = 30;
const int SKIP_TICKS = 1000 / FRAMES_PER_SECOND;
///////////////////////////////////////////////////////

enum {M_VIEW = 0, M_MOVE};


FILE* performanceFile = NULL;
bool print_to_file = false;

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
	sdkCreateTimer(&hapticTimer);
	sdkCreateTimer(&graphicsTimer);
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
	

	m_pCamera = new LYCamera(width, height);

	if (modelFile.empty()) modelFile = argv[1];
	m_pMesh = new LYMesh(modelFile);

	glm::vec3 centre = -m_pMesh->getModelCentre();

	//scale = m_pMesh->getScale();
	screenspace_renderer = new LYScreenspaceRenderer(m_pCamera);
	space_handler = new LYSpatialHash(m_pMesh->getVBO(), (uint) m_pMesh->getNumVertices(), make_uint3(256, 256, 256));
	if (deviceType == LYHapticInterface::KEYBOARD_DEVICE) haptic_interface = new LYKeyboardDevice(space_handler);
	if (deviceType == LYHapticInterface::HAPTIC_DEVICE) haptic_interface = new LYHapticDevice(space_handler);
	screenspace_renderer->setCollider(haptic_interface);
	screenspace_renderer->setPointRadius(pointRadius);
	haptic_interface->setTimer(hapticTimer);

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
	{
		buttonState |= 1<<button;
	}
	else if (state == GLUT_UP)
	{
		buttonState = 0;
	}

	mods = glutGetModifiers();

	if (mods & GLUT_ACTIVE_SHIFT)
	{
		buttonState = 2;
	}
	else if (mods & GLUT_ACTIVE_CTRL)
	{
		buttonState = 3;
	}

	ox = x;
	oy = y;

	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - ox);
	dy = (float)(y - oy);

	switch (M_VIEW)
	{
	case M_VIEW:
		if (buttonState == 3)
		{
			// left+middle = zoom
			camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
		}
		else if (buttonState & 2)
		{
			// middle = translate
			camera_trans[0] += dx / 100.0f;
			camera_trans[1] -= dy / 100.0f;
		}
		else if (buttonState & 1)
		{
			// left = rotate
			camera_rot[0] += dy / 5.0f;
			camera_rot[1] += dx / 5.0f;
		}

		break;

	case M_MOVE:
		{
			float translateSpeed = 0.003f;
			float3 p = haptic_interface->getPosition();

			if (buttonState==1)
			{
				float v[3];
				v[0] = dx*translateSpeed;
				v[1] = -dy*translateSpeed;
				v[2] = 0.0f;
				glm::vec4 r = modelViewMatrix * glm::vec4(v[0], v[1], v[2],1.0);
				p.x += r.x;
				p.y += r.y;
				p.z += r.z;
			}
			else if (buttonState==2)
			{
				float v[3];
				v[0] = 0.0f;
				v[1] = 0.0f;
				v[2] = dy*translateSpeed;
				glm::vec4 r = modelViewMatrix*glm::vec4(v[0], v[1], v[2], 1.0);
				p.x += r.x;
				p.y += r.y;
				p.z += r.z;
			}

			haptic_interface->setPosition(p);
		}
		break;
	}

	ox = x;
	oy = y;

	glutPostRedisplay();
}
// commented out to remove unused parameter warnings in Linux
float3 devPosition;
void key(unsigned char key, int x, int y)
{
	float3 pos = make_float3(0.0);
	switch (key)
	{
	case ' ':
		bPause = !bPause;
		haptic_interface->toggleForces();
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
		screenspace_renderer->setPointRadius(pointRadius);
		break;
	case '-':
		pointRadius -= 0.001f;
		screenspace_renderer->setPointRadius(pointRadius);
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
	case '/':
		print_to_file = !print_to_file;
		break;

	case 'w':
		// Move collider up
		pos.y += haptic_interface->getSpeed();
		break;
	case 'a':
		// Move collider left
		pos.x -= haptic_interface->getSpeed();
		break;
	case 's':
		pos.y -= haptic_interface->getSpeed();
		break;
	case 'd':
		// Move collider right
		pos.x += haptic_interface->getSpeed();
		break;
	case 'z':
		pos.z += haptic_interface->getSpeed();
		break;
	case 'c':
		pos.z -= haptic_interface->getSpeed();
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
	devPosition += pos;
	glutPostRedisplay();
}

void special(int k, int x, int y)
{
}

void cleanup()
{
	sdkDeleteTimer(&graphicsTimer);
	sdkDeleteTimer(&hapticTimer);
	fclose(performanceFile);
}

void idle(void)
{
	glutPostRedisplay();
}

DWORD next_game_tick = GetTickCount();
int sleep_time = 0;
static int hapticFPS = 0;
void display()
{
	LYTimer t(true);
	Sleep(20);
	// update the simulation
	haptic_interface->pause(bPause);
	if (!bPause)
	{
		space_handler->update();
		if (haptic_interface->getDeviceType() == LYHapticInterface::KEYBOARD_DEVICE){
			haptic_interface->setPosition(devPosition);
			float3 force = haptic_interface->calculateFeedbackUpdateProxy();
		}
	}
	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	viewMatrix = glm::mat4();

	for (int c = 0; c < 3; ++c)
	{
		camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
		camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
	}

	glm::mat4 modelMatrix = glm::mat4();
	modelMatrix *= glm::translate(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	modelMatrix *= glm::rotate(camera_rot_lag[0], glm::vec3(1,0,0));
	modelMatrix *= glm::rotate(camera_rot_lag[1], glm::vec3(0,1,0));
	modelMatrix *= glm::scale(glm::vec3(1));
	m_pMesh->setModelMatrix(modelMatrix);
	haptic_interface->setCameraMatrix(modelMatrix);

	haptic_interface->setWorkspaceScale(make_float3(0.05f, 0.05f, 0.05f));
	glm::mat4 viewMat = glm::scale(glm::mat4(), glm::vec3(15./m_pMesh->getScale()));
	viewMat = glm::translate(viewMat, glm::vec3(0,0,-15));
	m_pCamera->setViewMatrix(viewMat);

	screenspace_renderer->display(m_pMesh, mode);

	glutSwapBuffers();
	glutReportErrors();
	float averageTime = sdkGetAverageTimerValue(&hapticTimer);
	sprintf(fps_string, "Point-Based Rendering \t FPS: %5.3f \t SpaceHandler ms: %f", 1000.0f / (t.Elapsed()), averageTime);
	static int measureNum = 0;

	if (print_to_file){
		fprintf(performanceFile, "%d %f\n", measureNum++, averageTime);
	}

	glutSetWindowTitle(fps_string);
	hapticFPS++;
	if (hapticFPS >= 20){
		sdkResetTimer(&hapticTimer);
		hapticFPS = 0;
	}
}

int main(int argc, char **argv)
{
	performanceFile = fopen("performance.txt", "w");
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
