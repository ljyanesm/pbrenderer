#version 420

uniform mat4x4 u_ModelView;
uniform mat4x4 u_Persp;
uniform mat4x4 u_InvTrans;

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

layout (location = 0) in vec3 Position;
layout (location = 3) in vec3 Color;

out vec3 fs_PosEye;
out vec4 fs_Position;
out vec4 fs_Color;

void main(void) {
	vec4 pos = u_ModelView * vec4(Position.xyz, 1.0f);
	vec3 posEye = pos.xyz;
	float dist = length(posEye);
	gl_PointSize = pointRadius * (pointScale/dist);
	
	fs_PosEye = posEye;
	fs_Position = pos;
	fs_Color = vec4(Color.xyz,1.0f);
	gl_Position = u_Persp * pos;
}