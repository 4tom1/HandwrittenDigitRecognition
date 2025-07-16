#pragma once
#include "ImageData.h"
#include "MyMath.h"
#include <vector>

namespace neuralnet
{
	using NetSize = std::vector<size_t>;

	struct NetData
	{
		std::vector<math::Matrix> weightsLayers;
		std::vector<math::Matrix> biasesLayers;
	};

	class NeuralNet
	{
		public:

		NeuralNet(NetSize netSize, float eta, float biasInitVal, size_t epochs)
		{
			this->netSize = netSize;
			this->eta = eta;
			this->biasInitVal = biasInitVal;
			this->epochs = epochs;
		}

		void Build(const std::string& trainDataFilePath, const std::string& testDataFilePath);
		int EvaluatePicture(const std::string& filename);
		void Save(const std::string& filename);
		void Load(const std::string& filename);
		void Test(const std::string& testDataFilePath);
		void NetDataInit();

		void PrintRandomNetData()
		{
			Debug::Print("Weights");
			netData.weightsLayers[1].Print();
			
			Debug::Print("Biases");
			netData.biasesLayers.back().Print();
		}

		private:

		int Evaluate(const math::Matrix& outputActivation);
		float CrossEntropyLoss(const math::Matrix& yTrue, const math::Matrix& yPred);
		math::Matrix LabelToYTrueMatrix(int label, int numClasses);
		void Feedforward(math::Matrix& a, std::vector<math::Matrix>& activations);
		math::Matrix Feedforward(math::Matrix& a);
		void Backpropagation(
			const math::Matrix& yTrue,
			const std::vector<math::Matrix>& activations,
			std::vector<math::Matrix>& weightsDeltaOut,
			std::vector<math::Matrix>& biasesDeltaOut);
		void Update(const std::vector<math::Matrix>& weightsLayersDelta, const std::vector<math::Matrix>& biasesLayersDelta);
		void Train(std::vector<imagedata::ImageData>& data);
		void Test(std::vector<imagedata::ImageData>& data);

		template<typename T>
		void ShuffleData(std::vector<T>& data);

		float eta;
		float biasInitVal;
		size_t epochs;
		NetSize netSize;
		NetData netData;
	};
}