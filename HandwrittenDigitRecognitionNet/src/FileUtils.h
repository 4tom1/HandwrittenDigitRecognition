#pragma once
#include "NeuralNet.h"
#include <string>
#include <vector>

bool FileExists(const std::string& filename);
void SaveNetToBinaryFile(const std::string& filename, const neuralnet::NetSize& netSize, const neuralnet::NetData& netData);
void LoadNetFromBinaryFile(const std::string& filename, neuralnet::NetSize& netSize, neuralnet::NetData& netData);