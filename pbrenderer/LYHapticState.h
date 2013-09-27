
#ifndef _HAPTIC_DISPLAY_STATE_H
#define _HAPTIC_DISPLAY_STATE_H

#include <HD/hd.h>
#include <HDU/hdu.h>
#include <HDU/hduVector.h>
//
//
//#pragma comment(lib, "hd.lib")
//#pragma comment(lib, "hdu.lib")

typedef struct
{
    hduVector3Dd position;
	hduVector3Dd velocity;
    HDdouble transform[16];
	HDint UpdateRate;
} LYHapticState;

#endif _HAPTIC_DISPLAY_STATE_H