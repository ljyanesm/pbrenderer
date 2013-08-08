#version 330

uniform float u_Width;
uniform float u_Height;
uniform mat4 u_Persp;
uniform mat4 u_InvTrans;
uniform mat4 u_InvProj;

uniform sampler2D u_Depthtex;
uniform sampler2D u_Positiontex;

uniform float u_Far;
uniform float u_Near;

in vec2 fs_Texcoord;

out vec4 out_Normal;

//Depth used in the Z buffer is not linearly related to distance from camera
//This restores linear depth

float linearizeDepth(float exp_depth, float near, float far) {
    return	(2 * near) / (far + near -  exp_depth * (far - near)); 
}

vec3 uvToEye(vec2 texCoord, float depth){
	float x = texCoord.x * 2.0 - 1.0;
	float y = texCoord.y * 2.0 - 1.0;
	vec4 clipPos = vec4(x , y, depth, 1.0f);
	vec4 viewPos = u_InvProj * clipPos;
	return viewPos.xyz / viewPos.w;
}

vec3 getEyePos(in vec2 texCoord){
	float exp_depth = texture(u_Depthtex,fs_Texcoord).r;
    float lin_depth = linearizeDepth(exp_depth,u_Near,u_Far);
    return uvToEye(texCoord,lin_depth);
}

void main()
{       
    //Get Depth Information about the Pixel
	vec2 texel_x = vec2(u_Width, 0.0);
	vec2 texel_y = vec2(0.0, u_Height);

    float exp_depth = texture(u_Depthtex,fs_Texcoord).r;
	if (exp_depth > 0.99)
	{
		discard;
		return;
	}
	vec3 posEye = uvToEye(fs_Texcoord, exp_depth);
	vec3 position = uvToEye(fs_Texcoord, exp_depth);
	
	//Compute Gradients of Depth and Cross Product Them to Get Normal

	vec3 ddx = getEyePos(fs_Texcoord+texel_x);
	vec3 ddx2 = getEyePos(fs_Texcoord-texel_x);
	if (abs(ddx.z) > abs(ddx2.z)){
		ddx = ddx2;
	}

	vec3 ddy = getEyePos(fs_Texcoord+texel_y);
	vec3 ddy2 = getEyePos(fs_Texcoord-texel_y);
	if (abs(ddy.z) > abs(ddy2.z)){
		ddy = ddy2;
	}

	vec3 N = cross(ddx,ddy);
	out_Normal = vec4(normalize(N), 1.0);
	out_Normal = vec4(normalize(cross(dFdx(position.xyz), dFdy(position.xyz))), 1.0f);
}