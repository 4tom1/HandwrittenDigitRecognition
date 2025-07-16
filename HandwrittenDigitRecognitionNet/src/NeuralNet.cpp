#include "NeuralNet.h"
#include "FileUtils.h"
#include "MyProgressBar.h"
#include <filesystem>
#include <random>

using namespace neuralnet;

void NeuralNet::NetDataInit()
{
	size_t layersbW = netSize.size() - 1;

	// Preparing weightsLayer's matrixes
	for (size_t i = 0; i < layersbW; i++) {
		netData.weightsLayers.emplace_back(netSize[i + 1], netSize[i]);
	}

	// Preparing biasesLayer's vectors/matrixes
	for (size_t i = 0; i < layersbW; i++) {
		netData.biasesLayers.emplace_back(netSize[i + 1], 1);
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	// Fill weights with random floats in [0.0, 1.0] * sqr(2 / Nin)
	for (size_t layer = 0; layer < layersbW; layer++) {
		size_t rows = netData.weightsLayers[layer].rows();
		size_t cols = netData.weightsLayers[layer].cols();

		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				float value = static_cast<float>(2) / static_cast<float>(netSize[layer]);
				value = dist(gen) * std::sqrt(value);
				netData.weightsLayers[layer].At(r, c) = value;
			}
		}
	}

	// Fill biases
	for (size_t layer = 0; layer < layersbW; layer++) {
		size_t rows = netData.biasesLayers[layer].rows();

		for (size_t r = 0; r < rows; r++) {
			netData.biasesLayers[layer].At(r, 0) = biasInitVal;
		}
	}
}

int NeuralNet::EvaluatePicture(const std::string& filename)
{
	int prediction = -1;

	imagedata::ImageData imageData;

	// picture correctness check
	if (!imagedata::GetImageDataFormPNG(filename, imageData))
	{
		math::Matrix input = imagedata::ImageDataToMatrix(imageData);
		math::Matrix output = Feedforward(input);

		prediction = Evaluate(output);
	}

	return prediction;
}

void NeuralNet::Build(const std::string& trainDataFilePath, const std::string& testDataFilePath)
{	
	std::vector<imagedata::ImageData> trainData;
	std::vector<imagedata::ImageData> testData;

	trainData = imagedata::ReadCSV(trainDataFilePath);
	testData = imagedata::ReadCSV(testDataFilePath);

	Train(trainData);
	Test(testData);
}

void NeuralNet::Test(const std::string& testDataFilePath)
{
	std::vector<imagedata::ImageData> testData;
	testData = imagedata::ReadCSV(testDataFilePath);

	Test(testData);
}

void NeuralNet::Save(const std::string& filename)
{
	Debug::Print("Saving...");

	SaveNetToBinaryFile(filename, netSize, netData);

	Debug::Print("Saved.");
}

void NeuralNet::Load(const std::string& filename)
{
	Debug::Print("Loading...");

	LoadNetFromBinaryFile(filename, netSize, netData);

	Debug::Print("Loaded.");
}

int NeuralNet::Evaluate(const math::Matrix& outputActivation)
{
	int maxIndex = 0;
	float maxValue = outputActivation.At(0, 0);

	if (outputActivation.rows() == 1) {
		size_t cols = outputActivation.cols();
		for (int i = 1; i < cols; i++) {
			float value = outputActivation.At(0, i);
			if (value > maxValue)
			{
				maxValue = value;
				maxIndex = i;
			}
		}
	}
	else if (outputActivation.cols() == 1) {
		size_t rows = outputActivation.rows();
		for (int i = 1; i < rows; i++) {
			float value = outputActivation.At(i, 0);
			if (value > maxValue)
			{
				maxValue = value;
				maxIndex = i;
			}
		}
	}
	else {
		throw std::runtime_error("Output activation must be a vector (either 1 row or 1 column)");
	}

	return maxIndex;
}

float NeuralNet::CrossEntropyLoss(const math::Matrix& yTrue, const math::Matrix& yPred)
{
	if (yTrue.rows() != yPred.rows()) {
		throw std::invalid_argument("Size of yTrue and yPred must be equal.");
	}

	float epsilon = 1e-9f; // to avoid log(0)
	float loss = 0.0f;

	for (size_t i = 0; i < yTrue.rows(); ++i) {
		loss -= yTrue.At(i, 0) * std::log(yPred.At(i, 0) + epsilon);
	}

	return loss;
}

math::Matrix NeuralNet::LabelToYTrueMatrix(int label, int numClasses)
{
	if (label < 0 || label >= numClasses) {
		throw std::invalid_argument("Label out of range.");
	}

	math::Matrix yTrue(numClasses, 1); // Column vector: (numClasses x 1)
	yTrue.At(label, 0) = 1.0f;
	
	return yTrue;
}

void NeuralNet::Feedforward(math::Matrix& a, std::vector<math::Matrix>& activations)
{
	size_t hiddenLayers = netSize.size() - 2; // -2 because first l is input layer and last layer is output
	
	activations.push_back(a);

	for (size_t layer = 0; layer < hiddenLayers; layer++)
	{
		a = math::ReLU(netData.weightsLayers[layer] * a + netData.biasesLayers[layer]);
		activations.push_back(a);
	}

	a = math::Softmax(netData.weightsLayers.back() * a + netData.biasesLayers.back());
	activations.push_back(a);
}

math::Matrix NeuralNet::Feedforward(math::Matrix& a)
{
	size_t hiddenLayers = netSize.size() - 2; // -2 because first l is input layer and last layer is output

	for (size_t layer = 0; layer < hiddenLayers; layer++)
	{
		a = math::ReLU(netData.weightsLayers[layer] * a + netData.biasesLayers[layer]);
	}

	a = math::Softmax(netData.weightsLayers.back() * a + netData.biasesLayers.back());

	return a;
}

void NeuralNet::Backpropagation(
	const math::Matrix& yTrue,
	const std::vector<math::Matrix>& activations,
	std::vector<math::Matrix>& weightsDeltaOut,
	std::vector<math::Matrix>& biasesDeltaOut
)
{
	size_t layers = netSize.size() - 1;

	math::Matrix dZ = activations[layers] - yTrue;
	math::Matrix dW = dZ * activations[layers - 1].Transpose();
	math::Matrix db = dZ;

	weightsDeltaOut.push_back(dW);
	biasesDeltaOut.push_back(db);

	for (size_t layer = layers - 1; layer > 0; layer--)
	{
		const math::Matrix& a = activations[layer];
		const math::Matrix& W = netData.weightsLayers[layer];

		dZ = (W.Transpose() * dZ).HadamardProduct(math::ReLUPrime(a));
		dW = dZ * activations[layer - 1].Transpose();
		db = dZ;

		weightsDeltaOut.push_back(dW);
		biasesDeltaOut.push_back(db);
	}

	std::reverse(weightsDeltaOut.begin(), weightsDeltaOut.end());
	std::reverse(biasesDeltaOut.begin(), biasesDeltaOut.end());
}

void NeuralNet::Update(const std::vector<math::Matrix>& weightsDelta, const std::vector<math::Matrix>& biasesDelta)
{
	size_t layersbW = netSize.size() - 1;
	for (size_t i = 0; i < layersbW; i++) {
		netData.weightsLayers[i] = netData.weightsLayers[i] - (weightsDelta[i] * eta);
		netData.biasesLayers[i] = netData.biasesLayers[i] - (biasesDelta[i] * eta);
	}
}

void NeuralNet::Train(std::vector<imagedata::ImageData>& data)
{
	NetDataInit();
	MyProgressBar bar("Training...");

	for (size_t epoch = 0; epoch < epochs; epoch++) {
		ShuffleData(data);

		for (size_t imageIndex = 0; imageIndex < data.size(); imageIndex++) {
			std::vector<math::Matrix> activations;
			std::vector<math::Matrix> weightsDelta;
			std::vector<math::Matrix> biasesDelta;
			math::Matrix yTrue = LabelToYTrueMatrix(data[imageIndex].label, netSize.back());
			math::Matrix input = imagedata::ImageDataToMatrix(data[imageIndex]);

			Feedforward(input, activations);
			Backpropagation(yTrue, activations, weightsDelta, biasesDelta);
			Update(weightsDelta, biasesDelta);

			if (!static_cast<bool>((imageIndex + 1) % 1000))
			{
				size_t images_p = epoch * data.size() + (imageIndex + 1);
				size_t max = epochs * data.size();

				bar.Update(static_cast<double>(images_p) / static_cast<double>(max) * 100.0);
			}
		}
	}
}

void NeuralNet::Test(std::vector<imagedata::ImageData>& data)
{
	MyProgressBar bar("Testing...");

	size_t correctAnswers = 0;
	float lossSum = 0.0f;

	for (size_t imageIndex = 0; imageIndex < data.size(); imageIndex++) {
		math::Matrix yTrue = LabelToYTrueMatrix(data[imageIndex].label, netSize.back());
		math::Matrix input = imagedata::ImageDataToMatrix(data[imageIndex]);
		math::Matrix output = Feedforward(input);
		int prediction = Evaluate(output);

		if (prediction == data[imageIndex].label) {
			correctAnswers++;
		}

		lossSum += CrossEntropyLoss(yTrue, output);

		if (!static_cast<bool>((imageIndex + 1) % 1000))
		{
			bar.Update(static_cast<float>(imageIndex + 1) * 100.0f / data.size());
		}
	}

	float accuracy = static_cast<float>(correctAnswers) / static_cast<float>(data.size());
	float loss = lossSum / data.size();

	std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;
	std::cout << "Loss: " << loss << std::endl;
}

template<typename T>
void NeuralNet::ShuffleData(std::vector<T>& data)
{
	static std::random_device rd;
	static std::default_random_engine rng(rd());
	std::shuffle(data.begin(), data.end(), rng);
}