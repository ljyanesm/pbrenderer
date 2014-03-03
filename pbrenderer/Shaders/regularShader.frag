#version 330

uniform mat4 MVPMat;
uniform mat4 modelViewMat;
uniform mat4 u_Persp;

uniform int  fragDepth;

uniform vec4 lightDir;

in vec3 Pos;
in vec3 N;
in vec4 inColor;
in vec3 fs_posEye;
out vec4 gl_FragColor;


void main()
{
    vec3 incident = normalize(lightDir.xyz);
    vec3 viewer = normalize(Pos.xyz);

    //calculate depth
    vec4 pixelPos = vec4(fs_posEye + normalize(N),1.0f);
    vec4 clipSpacePos = u_Persp * pixelPos;
    if (fragDepth != 1) gl_FragDepth = clipSpacePos.z / clipSpacePos.w;

    vec3 H = normalize(incident + viewer);
    float specular = pow(max(0.0f, dot(H,N)),50.0f);
    float diffuse = max(0.0f, dot(incident, N));

	vec4 finalColor = vec4(inColor.rgb * diffuse + specular * vec3(1.0f), 1.0f);
//	vec4 finalColor = vec4(vec3(1.0) * diffuse + specular * vec3(1.0f), 1.0f);
//	vec4 finalColor = vec4(gl_FragDepth);
	gl_FragColor = finalColor;
}
