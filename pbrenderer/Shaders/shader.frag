#version 420

//ENUMS
#define	DISPLAY_DEPTH 0
#define	DISPLAY_NORMAL 1
#define	DISPLAY_POSITION 2
#define	DISPLAY_COLOR 3
#define	DISPLAY_DIFFUSE 4
#define	DISPLAY_DIFFUSE_SPEC 5
#define	DISPLAY_TOTAL 6

uniform mat4 u_ModelView;
uniform mat4 u_Persp;
uniform mat4 u_InvTrans;
uniform mat4 u_InvProj;

uniform sampler2D u_Depthtex;
uniform sampler2D u_Normaltex;
uniform sampler2D u_Colortex;
uniform sampler2D u_Positiontex;
uniform sampler2D u_Backgroundtex;
uniform sampler2D u_Thicktex;

uniform samplerCube u_Cubemaptex;

uniform float u_Far;
uniform float u_Near;
uniform float u_Aspect;
uniform int u_DisplayType;

uniform vec4 lightDir;

in vec2 fs_Texcoord;
in vec3 fs_Position;

out vec4 out_Color;

//Depth used in the Z buffer is not linearly related to distance from camera
//This restores linear depth

float linearizeDepth(float exp_depth, float near, float far) {
    return	(2.0 * near) / (far + near -  exp_depth * (far - near)); 
}

vec3 uvToEye(vec2 texCoord, float depth){
	float fovy = radians(60.0);
	float aspect = u_Aspect;
	float invFocalLenX   = tan(fovy * 0.5) * aspect;
	float invFocalLenY   = tan(fovy * 0.5);
	
	float x = texCoord.x * 2.0 - 1.0;
	float y = texCoord.y * -2.0 + 1.0;
	vec4 clipPos = vec4(x , y, depth, 1.0);
	vec4 viewPos = u_InvProj * clipPos;
	return viewPos.xyz / viewPos.w;
}


void main()
{
    //Get Texture Information about the Pixel
    vec3 N = texture(u_Normaltex,fs_Texcoord).xyz;
    float exp_depth = texture(u_Depthtex,fs_Texcoord).r;
    float lin_depth = linearizeDepth(exp_depth,u_Near,u_Far);
    vec3 Color = texture(u_Colortex,fs_Texcoord).xyz;
    vec3 BackColor = vec3(0.75, 0.75, 0.75);
	vec3 position = texture(u_Positiontex,fs_Texcoord).xyz;
	float thickness = 1.0;
    vec3 incident = normalize(lightDir.xyz);
    vec3 viewer = normalize(-position.xyz);
    
    //Blinn-Phong Shading Coefficients
    vec3 H = normalize(incident + viewer);
    float specular = pow(max(0.0f, dot(H,N)),15.0f);
    float diffuse = max(0.0f, dot(incident, N));

    incident = -incident;
    H = normalize(incident + viewer);
    specular += pow(max(0.0f, dot(H,N)),15.0f);
    diffuse += max(0.0f, dot(incident, N));

    //Background Only Pixels
    if(exp_depth > 0.99999999){
		out_Color = vec4(1.0f, 1.0f, 1.0f, 0.0f);
		return;
	}
        
    //Background Refraction
    vec4 refrac_color = vec4(0.75, 0.75, 0.75, 0.0);
    
    vec3 final_color = refrac_color.rgb;
    
	switch(u_DisplayType) {
		case(DISPLAY_DEPTH):
			out_Color = vec4(lin_depth,lin_depth,lin_depth,1.0f);
			break;
		case(DISPLAY_NORMAL):
			out_Color = vec4(abs(N.xyz),1.0f);
			break;
		case(DISPLAY_POSITION):
			out_Color = vec4( abs(position)/ u_Far,1.0f);
			break;
		case(DISPLAY_COLOR):
			out_Color = vec4(Color.rgb,1.0f);
			break;
		case(DISPLAY_DIFFUSE):
			out_Color = vec4(Color.rgb * diffuse, 1.0f);
			break;
		case(DISPLAY_DIFFUSE_SPEC):
			out_Color = vec4(Color.rgb * diffuse + specular * vec3(1.0f), 1.0f);
			break;
		case(DISPLAY_TOTAL):
			out_Color = vec4(final_color.rgb * diffuse + specular * vec3(1.0f), 1.0f);
			break;
	}
	return;
}