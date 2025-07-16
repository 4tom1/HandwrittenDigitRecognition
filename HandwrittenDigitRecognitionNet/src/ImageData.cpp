#include "ImageData.h"
#include "MyProgressBar.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"



using namespace imagedata;

math::Matrix imagedata::ImageDataToMatrix(const ImageData& imageData)
{
	int height = 28;
	int width = 28;

	math::Matrix m(height*width, 1);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			m.At(y * width + x, 0) = imageData.pixels[x][y] / 255.0f;
		}
	}

	return m;
}

size_t imagedata::CountLines(std::ifstream& file)
{
	std::string line;
	size_t lineCount = 0;

	while (std::getline(file, line)) {
		++lineCount;
	}

	file.clear();
	file.seekg(0);

	return lineCount;
}

std::vector<ImageData> imagedata::ReadCSV(const std::string& filename) {

	std::ifstream file(filename);

	if (!file.is_open()) {
		throw std::runtime_error("Cannot find a file: " + filename);
	}

	std::string line;
	std::vector<ImageData> dataset;

	size_t fileLines = CountLines(file) - 1;

	// Skip first line
	std::getline(file, line);

	int currentLine = 0;

	MyProgressBar bar("Reading a .csv file...");

	// File iteration
	while (std::getline(file, line)) {



		std::stringstream ss(line);
		std::string token;
		ImageData data;

		// Read label
		std::getline(ss, token, ',');
		data.label = std::stoi(token);

		// Read pixels
		int height = 28;
		int width = 28;

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				std::getline(ss, token, ',');
				data.pixels[x][y] = std::stoi(token);
			}
		}

		dataset.push_back(data);

		currentLine++;

		if (!(bool)(currentLine % 1000))
		{
			bar.Update((float)currentLine / fileLines * 100);
		}
	}

	return dataset;
}

void imagedata::WriteImageDataToPNG(const std::string& filename, const ImageData& imageData)
{
	int height = 28;
	int width = 28;

	std::vector<uint8_t> rawImageData(width * height);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			rawImageData[y * width + x] = imageData.pixels[x][y];
		}
	}

	int success = stbi_write_png(filename.c_str(), width, height, 1, rawImageData.data(), width);
	if (!success) {
		throw std::runtime_error("Failed to write PNG: " + filename);
	}
}

int imagedata::GetImageDataFormPNG(const std::string& filename, ImageData& outImageData)
{
	int width, height, channels;

	// Load image with one channel (grayscale)
	unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 1);

	if (!data) {
		throw std::runtime_error("Failed to load image: " + filename);
	}

	if (height != 28 || width != 28)
	{
		return -1;
	}

	outImageData.label = -1;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			unsigned char pixelValue = data[y * width + x];
			outImageData.pixels[x][y] = static_cast<uint8_t>(pixelValue);
		}
	}

	stbi_image_free(data);

	return 0;
}