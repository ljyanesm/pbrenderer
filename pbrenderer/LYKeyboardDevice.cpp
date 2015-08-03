#include "LYKeyboardDevice.h"


LYKeyboardDevice::LYKeyboardDevice(LYSpaceHandler *sh, LYMesh *proxyMesh, LYMesh *hipMesh)
{
	m_spaceHandler = sh;
	m_deviceType = LYHapticInterface::KEYBOARD_DEVICE;
	m_collider =	Collider();
	m_speed =		0.1f;
	m_size	=		0.03f;

	m_forceVector		= make_float3(0.0f);

	m_workspaceScale	= make_float3(0.3f);
	m_relativePosition	= make_float3(0.0f);

	m_ProxyObject	= proxyMesh;
	m_HIPObject		= hipMesh;

	m_HIPMatrix = glm::mat4();
	m_ProxyMatrix = glm::mat4();
	m_ViewMatrix = glm::mat4();

	std::vector<LYVertex> Vertices;
	std::vector<unsigned int> Indices;
	LYVertex hip(make_float3(0.0, 0.0, 0.0),
		make_float2(0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(0, 0, 0),
		int(0));
	LYVertex p(make_float3(0.0, 0.0, 0.0),
		make_float2(0.0, 0.0),
		make_float3(0.0, 0.0, 0.0),
		make_float3(255, 0, 0),
		int(1));

	Vertices.push_back(hip);
	Vertices.push_back(p);

	Indices.push_back(0);
	Indices.push_back(1);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(LYVertex) * 2, &Vertices[0], GL_STATIC_DRAW);

	glGenBuffers(1, &ib);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * 2, &Indices[0], GL_STATIC_DRAW);
	std::cout << "Using keyboard device!" << std::endl;
	m_HIPMatrix = glm::mat4();
}


LYKeyboardDevice::~LYKeyboardDevice(void)
{
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ib);
}

bool LYKeyboardDevice::isOk() const
{
	return true;
}
