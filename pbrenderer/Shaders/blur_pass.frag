#version 420

uniform sampler2D u_Depthtex;

uniform float u_Far;
uniform float u_Near;

in vec2 fs_Texcoord;

out vec4 out_Depth;

//Depth used in the Z buffer is not linearly related to distance from camera
//This restores linear depth
float linearizeDepth(float exp_depth, float near, float far) {
    return	(2.0 * near) / (far + near -  exp_depth * (far - near)); 
}

void main()
{       
    //Get Depth Information about the Pixel
    float exp_depth = texture(u_Depthtex,fs_Texcoord).r;
    float lin_depth = linearizeDepth(exp_depth,u_Near,u_Far);
    float blurRadius = (1.0f/lin_depth) * 0.0001;
    int windowWidth = 5;
    float sum = 0;
    float wsum = 0;
    
    if(exp_depth >= 0.9999){
		out_Depth = vec4(exp_depth);
		return;
    }
    
    for(int x = -windowWidth; x < windowWidth; x++){
		for(int y = -windowWidth; y < windowWidth; y++){
			vec2 samp = vec2(fs_Texcoord.s + x*blurRadius, fs_Texcoord.t + y*blurRadius);
			float sampleDepth = texture(u_Depthtex, samp).r;
			
			if(sampleDepth < 0.9999){
				//Spatial
				float r = length(vec2(x,y)) * 0.01;
				float w = exp(- (r*r));
			
				sum += sampleDepth * w ;
				wsum += w ;
			}
		}
    }
    
    if(wsum > 0.0001f){
		sum = sum/wsum;
    }
    
    out_Depth = vec4(sum);
}