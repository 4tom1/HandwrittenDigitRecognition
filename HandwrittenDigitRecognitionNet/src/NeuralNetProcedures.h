#pragma once

#include "NeuralNet.h"

namespace neuralnet
{
	void Main(neuralnet::NeuralNet& net, 
		const std::string& trainDataFilePath, 
		const std::string& testDataFilePath, 
		const std::string& netFilePath, 
		const std::string& testPicturesFolderPath);

	void BuildingProcedure(neuralnet::NeuralNet& net, 
		const std::string& trainDataFilePath, 
		const std::string& testDataFilePath, 
		const std::string& netFilePath);

	bool LoadingProcedure(neuralnet::NeuralNet& net, const std::string& netFilePath);
	bool EvaluateProcedure(neuralnet::NeuralNet& net, const std::string& testPicturesFolderPath);
}