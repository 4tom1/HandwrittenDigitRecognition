#include "FileUtils.h"
#include "MyMath.h"
#include "NeuralNet.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>


bool FileExists(const std::string& filename) {
	return std::filesystem::exists(filename);
}

void SaveNetToBinaryFile(const std::string& filename, const neuralnet::NetSize& netSize, const neuralnet::NetData& netData) {
	
	if (FileExists(filename)) {
		throw std::runtime_error("Cannot overwrite a file: " + filename);
	}

	std::ofstream file(filename, std::ios::binary);

	// Save number of layers
	size_t layers = netSize.size();
	file.write(reinterpret_cast<const char*>(&layers), sizeof(size_t));

	// Save each netSize value
	for (size_t i = 0; i < layers; i++) {
		file.write(reinterpret_cast<const char*>(&netSize[i]), sizeof(size_t));
	}

	// Save weights
	for (const auto& matrix : netData.weightsLayers) {
		size_t rows = matrix.rows();
		size_t cols = matrix.cols();
		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				float value = matrix.At(r, c);
				file.write(reinterpret_cast<const char*>(&value), sizeof(float));
			}
		}
	}

	// Save biases
	for (const auto& matrix : netData.biasesLayers) {
		size_t rows = matrix.rows();
		for (size_t r = 0; r < rows; r++) {
			float value = matrix.At(r, 0);
			file.write(reinterpret_cast<const char*>(&value), sizeof(float));
		}
	}

	file.close();
}

void LoadNetFromBinaryFile(const std::string& filename, neuralnet::NetSize& netSize, neuralnet::NetData& netData)
{
	std::ifstream file(filename, std::ios::binary);
	
	if (!file.is_open()) {
		throw std::runtime_error("Cannot find a file: " + filename);
	}

	// Read netSize size
	size_t layers = 0;
	file.read(reinterpret_cast<char*>(&layers), sizeof(size_t));
	netSize.resize(layers);

	size_t layersbW = layers - 1;

	// Read each netSize value
	for (size_t i = 0; i < layers; i++) {
		file.read(reinterpret_cast<char*>(&netSize[i]), sizeof(size_t));
	}

	// Preparing weightsLayer's matrixes
	for (size_t i = 0; i < layersbW; i++) {
		netData.weightsLayers.emplace_back(netSize[i + 1], netSize[i]);
	}

	// Preparing biasesLayer's vectors/matrixes
	for (size_t i = 0; i < layersbW; i++) {
		netData.biasesLayers.emplace_back(netSize[i+1], 1);
	}

	for (size_t layer = 0; layer < layersbW; layer++) {
		size_t rows = netData.weightsLayers[layer].rows();
		size_t cols = netData.weightsLayers[layer].cols();
		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				float value = 0.0f;
				file.read(reinterpret_cast<char*>(&value), sizeof(float));
				netData.weightsLayers[layer].At(r, c) = value;
			}
		}
	}

	for (size_t layer = 0; layer < layersbW; layer++) {
		size_t rows = netData.biasesLayers[layer].rows();
		for (size_t r = 0; r < rows; r++) {
			float value = 0.0f;
			file.read(reinterpret_cast<char*>(&value), sizeof(float));
			netData.biasesLayers[layer].At(r, 0) = value;
		}
	}

	file.close();
}