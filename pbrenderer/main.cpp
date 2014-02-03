#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cstdio>
#include <cstdlib>
#define _USE_MATH_DEFINES
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
#include "LYPLYLoader.h"

int width = 1600;
int height = 900;

float	pointRadius = 0.01f;
float	pointScale = 0.01f;
float	influenceRadius = 0.2f;
int		pointDiv = 0;

const float NEARP = 0.1f;
const float FARP = 1000.0f;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, 0};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, 0};
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

LYMesh* m_CubeObj;
LYshader* regularShader;

LYWorld m_pWorld;
// Information below this line has to move to LYWorld.
///////////////////////////////////////////////////////
LYPLYLoader *m_plyLoader;
uint LYPLYLoader::nX = 0;
uint LYPLYLoader::nY = 0;
uint LYPLYLoader::nZ = 0;
uint LYPLYLoader::nNX = 0; 
uint LYPLYLoader::nNY = 0; 
uint LYPLYLoader::nNZ = 0;
uint LYPLYLoader::nR = 0; 
uint LYPLYLoader::nG = 0; 
uint LYPLYLoader::nB = 0;
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


float global_point_scale = 0.01f;
float local_point_scale = 0.01f;

float3 devPosition;
int loadedModel = 0;
float objectScale = 0.0;
enum {M_VIEW = 0, M_MOVE};


FILE* performanceFile = NULL;
bool print_to_file = false;

namespace fs = ::boost::filesystem;
std::vector<fs::path> modelFiles;

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const std::string& ext, std::vector<fs::path>& ret)
{  
	if (!fs::exists(root)) return;

	if (fs::is_directory(root))
	{
		fs::recursive_directory_iterator it(root);
		fs::recursive_directory_iterator endit;
		while(it != endit)
		{
			if (fs::is_regular_file(*it) && it->path().extension() == ext && it->path().filename() != "proxy.ply" && it->path().filename() != "hip.ply")
			{

				ret.push_back(it->path().filename());
			}
			++it;
		}
	}
}

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
	loadedModel = 2;
	get_all(".", ".ply", modelFiles);
	if (modelFile.empty()) modelFile = argv[1];
	if (!modelFiles.empty() && !modelFile.empty()) modelFile = modelFiles.at(loadedModel).filename().string();

	m_pCamera = new LYCamera(width, height);
	screenspace_renderer = new LYScreenspaceRenderer(m_pCamera);


	m_pMesh = m_plyLoader->getInstance().readFile(modelFile);
	global_point_scale = m_pMesh->getScale();
	space_handler = new LYSpatialHash(m_pMesh->getVBO(), (uint) m_pMesh->getNumVertices(), make_uint3(128, 128, 128));

	if (deviceType == LYHapticInterface::KEYBOARD_DEVICE) 
		haptic_interface = new LYKeyboardDevice(space_handler, 
		m_plyLoader->getInstance().readFile("proxy.ply"), 
		m_plyLoader->getInstance().readFile("hip.ply"));
	
	if (deviceType == LYHapticInterface::HAPTIC_DEVICE) 
		haptic_interface = new LYHapticDevice(space_handler, 
		m_plyLoader->getInstance().readFile("proxy.ply"), 
		m_plyLoader->getInstance().readFile("hip.ply"));
	
	screenspace_renderer->setCollider(haptic_interface);
	screenspace_renderer->setPointRadius(pointRadius);
	haptic_interface->setTimer(hapticTimer);


	m_CubeObj = m_plyLoader->getInstance().readFile("cube.ply");
	m_CubeObj->setRenderMode(true);
	regularShader = new LYshader("./shaders/depth_pass.vs", "./shaders/depth_pass.frag", "Color");

	glutReportErrors();
}

void reshape(int w, int h)
{
	m_pCamera->perspProjection(width, height, 60.0f, NEARP, FARP);
	p = glm::mat4();
	p = glm::perspective(60.0f, (float) width/ (float) height, NEARP, FARP);
	pointScale = m_pCamera->getHeight() / tanf(m_pCamera->getFOV()*0.5f*(float)M_PI/180.0f) ;
	screenspace_renderer->setPointScale(pointScale);
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
			objectScale += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
			//camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
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

void key(unsigned char key, int x, int y)
{
	LYMesh *tmpModel = m_pMesh;
	LYSpaceHandler *tmpSpace = space_handler;
	float3 pos = make_float3(0.0);
	switch (key)
	{
	case ' ':
		bPause = haptic_interface->toggleForces(!bPause);
		break;
	case 13:		// ENTER key

		break;
	case '\033':
	case 'q':
		exit(0);
		break;
	case '[':
		haptic_interface->pause();
		loadedModel = loadedModel--%modelFiles.size();
		modelFile = modelFiles.at(loadedModel).string();
		printf("Loading new object: %d - %s\n\n", loadedModel, modelFile.c_str());
		while (true)
		{
			try{
				m_pMesh = m_plyLoader->getInstance().readFile(modelFile);
				global_point_scale = m_pMesh->getScale();
				break;	// Exit the loop
			}
			catch (int e){
				printf("The object %d - %s could not be loaded, loading next model... and %d\n", loadedModel, modelFile.c_str(), e);
				loadedModel = loadedModel--%modelFiles.size();
				modelFile = modelFiles.at(loadedModel).string();
			}
		}
		space_handler = new LYSpatialHash(m_pMesh->getVBO(), m_pMesh->getNumVertices(), make_uint3(256));
		haptic_interface->setSpaceHandler(space_handler);
		haptic_interface->start();
		delete tmpModel;
		delete tmpSpace;
		break;
	case ']':
		haptic_interface->pause();
		loadedModel = ++loadedModel%modelFiles.size();
		modelFile = modelFiles.at(loadedModel).string();
		printf("Loading new object: %d - %s\n\n", loadedModel, modelFile.c_str());
		while (true)
		{
			try{
				m_pMesh = m_plyLoader->getInstance().readFile(modelFile);
				global_point_scale = m_pMesh->getScale();
				break;
			}
			catch (int e){
				printf("The object %d - %s could not be loaded, loading next model... and %d\n", loadedModel, modelFile.c_str(), e);
				loadedModel = ++loadedModel%modelFiles.size();
				modelFile = modelFiles.at(loadedModel).string();
			}
		}
		space_handler = new LYSpatialHash(m_pMesh->getVBO(), m_pMesh->getNumVertices(), make_uint3(256));
		haptic_interface->setSpaceHandler(space_handler);
		haptic_interface->start();
		delete tmpModel;
		delete tmpSpace;
		break;
	case 'p':
		mode = (LYScreenspaceRenderer::DisplayMode)
			((mode+ 1) % LYScreenspaceRenderer::NUM_DISPLAY_MODES);
		break;
	case 'r':
		displayEnabled = !displayEnabled;
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

	case '+':
		pointRadius += 0.001f;
		screenspace_renderer->setPointRadius(pointRadius);
		break;
	case '-':
		pointRadius -= 0.001f;
		screenspace_renderer->setPointRadius(pointRadius);
		break;
	case '>':
		pointScale *= 1.1f;
		screenspace_renderer->setPointScale(pointScale);
		break;
	case '<':
		pointScale *= 0.9f;
		screenspace_renderer->setPointScale(pointScale);
		break;
	case ',':
		// Make the radius 10% bigger
		influenceRadius *= 1.1f;
		space_handler->setInfluenceRadius(influenceRadius);
		haptic_interface->setSize(influenceRadius);
		std::cout << "Influence Radius: " << influenceRadius << std::endl;
		break;
	case '.':
		// Make the radius  10% smaller
		influenceRadius *= 0.9f;
		space_handler->setInfluenceRadius(influenceRadius);
		haptic_interface->setSize(influenceRadius);
		std::cout << "Influence Radius: " << influenceRadius << std::endl;
		break;
	case ':':
		global_point_scale += 0.01f;
		break;
	case '@':
		global_point_scale -= 0.01f;
		break;
	case ';':
		local_point_scale += 0.1f;
		break;
	case '\'':
		local_point_scale -= 0.1f;
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
	printf("Cleaning up!\n");
	sdkDeleteTimer(&graphicsTimer);
	sdkDeleteTimer(&hapticTimer);
	delete m_plyLoader;
	delete haptic_interface;
	delete haptics;
	delete screenspace_renderer;
	delete space_handler;
	delete m_pMesh;
	delete m_pCamera;
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
	sdkStartTimer(&graphicsTimer);
	Sleep(20);
	// update the simulation
	if (bPause)
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

	/*/////////////////////////////////////////////////////////////////////////////////////////
	Setup the model matrix
	/////////////////////////////////////////////////////////////////////////////////////////*/
	glm::vec3 modelCentre(m_pMesh->getModelCentre());

	glm::mat4 modelMatrix;
	modelMatrix *= glm::scale(glm::vec3(global_point_scale));
	modelMatrix *= glm::translate(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
	modelMatrix *= glm::rotate(camera_rot_lag[0], glm::vec3(1,0,0));
	modelMatrix *= glm::rotate(camera_rot_lag[1], glm::vec3(0,1,0));
	modelMatrix *= glm::translate(-modelCentre);
	m_pMesh->setModelMatrix(modelMatrix);
	//////////////////////////////////////////////////////////////////////////////////////////

	/*/////////////////////////////////////////////////////////////////////////////////////////
	Setup the haptic dimensions for collision detection and force response:
	Find the object that is closest to the HIP and use it for the cameraMatrix computation.
	/////////////////////////////////////////////////////////////////////////////////////////*/
	float3 devicePosition = haptic_interface->getPosition();
	haptic_interface->setCameraMatrix(modelMatrix);		// Camera matrix is the final transformation of the screen!
	haptic_interface->setWorkspaceScale(make_float3(0.05f, 0.05f, 0.05f));
	//////////////////////////////////////////////////////////////////////////////////////////

	/*/////////////////////////////////////////////////////////////////////////////////////////
	Setup the view matrix
	/////////////////////////////////////////////////////////////////////////////////////////*/
	glm::mat4 viewMat = glm::mat4();

	viewMat *= glm::lookAt(glm::vec3(0,0,15), glm::vec3(0,0,0), glm::vec3(0,1,0));
	viewMat *= glm::scale(glm::vec3(local_point_scale));
	m_pCamera->setViewMatrix(viewMat);
	//////////////////////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////
	screenspace_renderer->addDisplayMesh(m_pMesh);
	screenspace_renderer->addDisplayMesh(m_CubeObj);

	screenspace_renderer->display(mode);
	sdkStopTimer(&graphicsTimer);
	float displayTimer = sdkGetAverageTimerValue(&graphicsTimer);
	glutSwapBuffers();
	glutReportErrors();
	float averageTime = sdkGetAverageTimerValue(&hapticTimer);
	sprintf(fps_string, "Point-Based Rendering - %s - Graphic FPS: %5.3f        Haptic FPS: %f", modelFile.c_str(), 1000.0f / (displayTimer), 1000.0f / averageTime);
	static int measureNum = 0;

	if (print_to_file){
		fprintf(performanceFile, "%d %f\n", measureNum++, averageTime);
	}

	glutSetWindowTitle(fps_string);
	hapticFPS++;
	if (hapticFPS >= 20){
		sdkResetTimer(&graphicsTimer);
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
