#include "IOManager.h"


IOManager::IOManager(void)
{
}

IOManager::IOManager( LYHapticInterface *_d, float4 wsDim, float4 wsWD ) :
_device(_d),
wsDimension(wsDim),
wsWorkingDimension(wsWD)
{

}


IOManager::~IOManager(void)
{
}
