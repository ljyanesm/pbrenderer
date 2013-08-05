#ifndef __LYSHADER_H_
#define __LYSHADER_H_

#include <GL\glew.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdarg>

class LYshader{
	unsigned int vs,fs,program;
	void loadFile(const char* fn,std::string& str);
	unsigned int loadShader(std::string& source,unsigned int mode);
	public:
		LYshader();
		LYshader(const char* vss,const char* fss, const char *attribLoc);
		~LYshader();
		void useShader();
		void delShader();
		unsigned int getProgramId();
};

#endif
