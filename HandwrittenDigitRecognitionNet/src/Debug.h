#pragma once
#include <iostream>
#include <string>

namespace Debug
{
	inline void Print(std::string m)
	{
		std::cerr << m << std::endl;
	}

	inline void Print(int m)
	{
		std::cerr << m << std::endl;
	}
}