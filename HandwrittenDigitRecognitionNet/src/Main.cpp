#include "NeuralNetProcedures.h"
#include "FileUtils.h"

const std::string testPicturesFolderPath = "E:/HandwrittenDigitRecognition/TestPictures/";
const std::string testDataFilePath = "E:/HandwrittenDigitRecognition/data/mnist_test.csv";
const std::string trainDataFilePath = "E:/HandwrittenDigitRecognition/data/mnist_train.csv";
const std::string netFilePath = "E:/HandwrittenDigitRecognition/SavedNet/";

int main()
{
	neuralnet::NetSize netSize = { 28 * 28, 16, 16, 10 };
	float eta = 0.001f;
	float biasInitVal = 0.01f;
	size_t epochs = 10;
	
	neuralnet::NeuralNet net {
		netSize,
		eta,
		biasInitVal,
		epochs
	};
	
	neuralnet::Main(net, trainDataFilePath, testDataFilePath, netFilePath, testPicturesFolderPath);
}