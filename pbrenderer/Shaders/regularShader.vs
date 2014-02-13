#version 330

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec3 Color;
layout (location = 3) in vec2 TexCoord;

uniform mat4 MVPMat;
uniform mat4 modelViewMat;
uniform mat4 u_Persp;

out vec3 N;
out vec4 inColor;
out vec3 Pos;
out vec3 fs_posEye;
void main()
{
	N = Normal;
	inColor = vec4(Color, 1.0f);
    gl_Position = MVPMat * vec4(Position, 1.0f);
	Pos = vec3(gl_Position.xyz);
	fs_posEye = vec3(modelViewMat * vec4(Position, 1.0f));
}
