#ifndef MODELINFO_H
#define MODELINFO_H
//#define BOOST_FILESYSTEM_VERSION 3
#define BOOST_SYSTEM_DYN_LINK
//#define BOOST_FILESYSTEM_NO_DEPRECATED 
#include <boost/filesystem.hpp>
#include <string>

class ModelInfo{
public:
	std::string modelPath;
	float global_point_scale;
	float local_point_scale;
	float pointRadius;
	float pointScale;
};

#endif