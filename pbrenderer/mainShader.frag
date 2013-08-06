#version 420
uniform mat4 p;

in vec3 vsColor;

in vec3 fs_PosEye;
in vec4 fs_Position;
in vec4 fs_Color;
in vec2 fs_texCoord;

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

out vec4 color;

float linearizeDepth(float exp_depth, float near, float far) {
    return	(2.0 * near) / (far + near -  exp_depth * (far - near)); 
}

void main()
{
    vec3 N;
    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

	vec4 pixelPos = vec4(fs_PosEye + normalize(N)*pointRadius,1.0);
    vec4 clipSpacePos = p * pixelPos;
	float depth = clipSpacePos.z / clipSpacePos.w;
	
	float lin_depth = linearizeDepth(depth,0.1,100);


	color = vec4(vsColor, 1.0);
}
