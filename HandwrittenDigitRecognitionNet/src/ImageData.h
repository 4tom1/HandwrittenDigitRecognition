#pragma once
#include "Debug.h"
#include "MyMath.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


namespace imagedata
{
	struct ImageData {
		int label;
		uint8_t pixels[28][28]; // 784 pixels for 28x28 image
	};

	math::Matrix ImageDataToMatrix(const ImageData& imageData);
	size_t CountLines(std::ifstream& file);
	std::vector<ImageData> ReadCSV(const std::string& filename);
	void WriteImageDataToPNG(const std::string& filename, const ImageData& imageData);
	int GetImageDataFormPNG(const std::string& filename, ImageData& outImageData);
}