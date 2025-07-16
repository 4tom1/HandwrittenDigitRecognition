#include "NeuralNetProcedures.h"
#include "FileUtils.h"



void neuralnet::Main(neuralnet::NeuralNet& net, const std::string& trainDataFilePath, const std::string& testDataFilePath, const std::string& netFilePath, const std::string& testPicturesFolderPath)
{
	std::string input;

	std::cout << "1. Build a new net.\n2. Test a saved net.\n3. Use a saved net to predict a number form .png file.\n>> ";
	std::cin >> input;

	if (input == "1")
	{
		BuildingProcedure(net, trainDataFilePath, testDataFilePath, netFilePath);
	}

	else if (input == "2")
	{
		while (LoadingProcedure(net, netFilePath));
		net.Test(testDataFilePath);
	}

	else
	{
		while (LoadingProcedure(net, netFilePath));
		while (EvaluateProcedure(net, testPicturesFolderPath));
	}
}

void neuralnet::BuildingProcedure(neuralnet::NeuralNet& net, const std::string& trainDataFilePath, const std::string& testDataFilePath, const std::string& netFilePath)
{
	std::string input;

	net.Build(trainDataFilePath, testDataFilePath);

	std::cout << "Do you want to save it? y/n: ";
	std::cin >> input;

	if (input == "y")
	{
		std::cout << "Name of new net: ";
		std::cin >> input;

		net.Save(netFilePath + input);
	}
}

bool neuralnet::LoadingProcedure(neuralnet::NeuralNet& net, const std::string& netFilePath)
{
	std::string input;

	std::cout << "Which net do you want load?: ";

	std::cin >> input;

	std::string filename = netFilePath + input;

	if (!FileExists(filename))
	{
		Debug::Print("A file: " + input + " doesn't exist.");

		return true;
	}

	net.Load(filename);

	return false;
}

bool neuralnet::EvaluateProcedure(neuralnet::NeuralNet& net, const std::string& testPicturesFolderPath)
{
	std::string input;

	std::cout << "Type a file path to .png image and get prediction: ";

	std::cin >> input;

	if (!FileExists(testPicturesFolderPath + input))
	{
		Debug::Print("A file: " + input + " doesn't exist.");

		return true;
	}

	int prediction = net.EvaluatePicture(testPicturesFolderPath + input);

	if (prediction == -1)
	{
		Debug::Print("A file: " + input + " cannot be processed. Check if it is .png file and if it has 28x28 pixels.");

		return true;
	}

	std::cout << "Your digit is: " << prediction << std::endl;
	std::cout << "Do you want check another digit? y/n: ";

	std::cin >> input;

	if (input == "n")
	{
		return false;
	}

	return true;
}