#version 420

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec2 Texture;
layout (location = 3) in vec3 Color;

uniform mat4 mv;
uniform mat4 p;

out vec3 vsColor;
void main()
{
	vec4 pos = vec4(Position, 1.0);
	gl_Position = pos;
	vsColor = Color;
}
