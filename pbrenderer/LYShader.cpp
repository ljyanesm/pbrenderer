#include "LYShader.h"

void LYshader::loadFile(const char* fn,std::string& str)
{
	std::ifstream in(fn);
	if(!in.is_open())
	{
		std::cout << "The file " << fn << " cannot be opened\n";
		return;
	}
	char tmp[300];
	while(!in.eof())
	{
		in.getline(tmp,300);
		str+=tmp;
		str+='\n';
	}
}

unsigned int LYshader::loadShader(std::string& source,unsigned int mode)
{
	unsigned int id;
	id=glCreateShader(mode);
	
	const char* csource=source.c_str();
	
	glShaderSource(id,1,&csource,NULL);
	glCompileShader(id);
	char error[1000];
	glGetShaderInfoLog(id,1000,NULL,error);
	std::cout << "Compile status: \n" << error << std::endl;
	return id;
}

LYshader::LYshader(const char* vss,const char* fss, const char *attribLoc)
{
	std::string source;
	loadFile(vss,source);
	vs=loadShader(source,GL_VERTEX_SHADER);
	source="";
	loadFile(fss,source);
	fs=loadShader(source,GL_FRAGMENT_SHADER);
	
	program=glCreateProgram();
	glAttachShader(program,vs);
	glAttachShader(program,fs);
	
	glBindAttribLocation(program, 0, "Position");
	glBindAttribLocation(program, 1, attribLoc);

	glLinkProgram(program);
	glUseProgram(program);	
}

LYshader::~LYshader()
{
	glDetachShader(program,vs);
	glDetachShader(program,fs);
	glDeleteShader(vs);
	glDeleteShader(fs);
	glDeleteProgram(program);
}

void LYshader::useShader()
{
	glUseProgram(program);
}

unsigned int LYshader::getProgramId()
{
	return program;
}


void LYshader::delShader()
{
	glUseProgram(0);
}
		
