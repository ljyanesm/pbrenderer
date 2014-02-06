#version 330

layout(location = 0) in vec3 vert;
/*layout(location = 1) in vec2 tex;
layout(location = 2) in vec3 nrm;
layout(location = 3) in vec3 col;*/

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    gl_Position = projection * view * model * vec4(vert, 1.0);
}
