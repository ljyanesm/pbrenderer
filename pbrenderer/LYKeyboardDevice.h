#pragma once
#include "defines.h"
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <vector>

#include "LYVertex.h"
#include "LYHapticInterface.h"
#include "LYPLYLoader.h"

class LYKeyboardDevice :
	public LYHapticInterface
{
public:
	LYKeyboardDevice(LYSpaceHandler *sh, LYMesh *proxyMesh, LYMesh *hipMesh);
	~LYKeyboardDevice(void);

	bool isOk() const;
};
