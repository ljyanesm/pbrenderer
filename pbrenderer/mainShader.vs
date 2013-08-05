#version 420

layout (location = 0) in vec3 Position;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec2 Texture;
layout (location = 3) in vec3 Color;

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

uniform mat4 mv;
uniform mat4 p;

out vec3 vsColor;

out vec3 fs_PosEye;
out vec4 fs_Position;
out vec4 fs_Color;

void main()
{
    vec3 posEye = vec3(mv * vec4(Position.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

	fs_PosEye = posEye;
	fs_Position = vec4(Position.xyz, 1.0);

	vec4 pos = vec4(Position, 1.0);
	gl_Position = pos * p * mv;
	vsColor = Color;
}
