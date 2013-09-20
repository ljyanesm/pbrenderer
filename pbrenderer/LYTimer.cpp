#include "LYTimer.h"

LYTimer::LYTimer(bool start_immediately)
	: start(0), stop(0), running(false)
{
	if (start_immediately)
	{
		Start(true);
	}
}

long long LYTimer::milliseconds_now() {
	static LARGE_INTEGER s_frequency;
	static BOOL s_use_qpc = QueryPerformanceFrequency(&s_frequency);
	if (s_use_qpc) {
		LARGE_INTEGER now;
		QueryPerformanceCounter(&now);
		return (1000LL * now.QuadPart) / s_frequency.QuadPart;
	} else {
		return GetTickCount();
	}
}

void LYTimer::Start(bool reset)
{
	if (!running)
	{
		if (reset)
		{
			start = milliseconds_now();
		}
		running = true;
	}
}
void LYTimer::Stop()
{
	if (running)
	{
		stop = milliseconds_now();
		running = false;
	}
}
long long LYTimer::Elapsed()
{
	return (running ? milliseconds_now() : stop) - start;
}