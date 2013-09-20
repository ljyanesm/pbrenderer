#pragma once
#include <Windows.h>
class LYTimer
{
public:
	explicit LYTimer(bool start_immediately = false);
	void Start(bool reset = false);
	void Stop();
	long long Elapsed();
private:

	long long milliseconds_now();

	long long start, stop;
	bool running;
};