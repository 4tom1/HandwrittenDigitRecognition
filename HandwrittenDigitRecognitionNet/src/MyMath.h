#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

namespace math
{
	class Matrix {
		private:
			std::vector<float> data;
			size_t m_rows, m_cols;

		public:
			// Constructors
			Matrix() = default;

			Matrix(size_t rows, size_t cols, float value = 0.0f) {
				data.resize(rows * cols, value);
				this->m_rows = rows;
				this->m_cols = cols;
			}

			// Accessors
			size_t rows() const { return m_rows; }
			size_t cols() const { return m_cols; }

			float& At(size_t i, size_t j) { return data.at(i * m_cols + j); }
			float  At(size_t i, size_t j) const { return data.at(i * m_cols + j); }

			// Matrix addition
			Matrix operator+(const Matrix& m) const {
				if (rows() != m.rows() || cols() != m.cols())
					throw std::invalid_argument("Matrix dimensions must match for addition");

				Matrix result(rows(), cols());
				for (size_t i = 0; i < rows(); i++)
					for (size_t j = 0; j < cols(); j++)
						result.At(i, j) = this->At(i, j) + m.At(i, j);

				return result;
			}

			// Matrix substraction
			Matrix operator-(const Matrix& m) const {
				if (rows() != m.rows() || cols() != m.cols())
					throw std::invalid_argument("Matrix dimensions must match for addition");

				Matrix result(rows(), cols());
				for (size_t i = 0; i < rows(); i++)
					for (size_t j = 0; j < cols(); j++)
						result.At(i, j) = this->At(i, j) - m.At(i, j);

				return result;
			}

			// Matrix multiplication
			Matrix operator*(const Matrix &m) const {

				if (cols() != m.rows()) {
					throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
				}

				Matrix result(rows(), m.cols());

				for (size_t i = 0; i < rows(); i++) {
					for (size_t j = 0; j < m.cols(); j++) {
						for (size_t k = 0; k < cols(); k++) {
							result.At(i, j) += this->At(i, k) * m.At(k, j);
						}
					}
				}

				return result;
			}

			// Scalar multiplication
			Matrix operator*(float scalar) const {
				Matrix result(rows(), cols());
				for (size_t i = 0; i < rows(); i++)
					for (size_t j = 0; j < cols(); j++)
						result.At(i, j) = this->At(i, j) * scalar;
				return result;
			}

			// Transpose
			Matrix Transpose() const {
				Matrix result(m_cols, m_rows);
				for (size_t i = 0; i < m_rows; i++) {
					for (size_t j = 0; j < m_cols; j++) {
						result.At(j, i) = this->At(i, j); // swap i <=> j
					}
				}
				return result;
			}

			Matrix HadamardProduct(const Matrix& m) const {
				if (cols() != m.cols() || rows() != m.rows()) {
					throw std::invalid_argument("Matrix dimensions do not match for Hadamard product.");
				}

				Matrix result(rows(), cols());
				for (size_t i = 0; i < rows(); i++)
					for (size_t j = 0; j < cols(); j++)
						result.At(i, j) = this->At(i, j) * m.At(i, j);
				return result;
			}

			// Print
			void Print(std::ostream& os = std::cout) const {
				for (size_t i = 0; i < rows(); i++) {
					for (size_t j = 0; j < cols(); j++) {
						float val = this->At(i, j);
						os << std::setw(8) << val << " ";
					}
					os << "\n";
				}
			}
	};

	inline float max(math::Matrix m)
	{
		float maxVal = m.At(0, 0);

		for (size_t i = 0; i < m.rows(); ++i)
		{
			for (size_t j = 0; j < m.cols(); ++j)
			{
				float val = m.At(i, j);
				if (val > maxVal)
				{
					maxVal = val;
				}
			}
		}

		return maxVal;
	}

	inline float ReLU(float x)
	{
		return std::max(x, 0.0f);
	}

	inline math::Matrix ReLU(const math::Matrix& x)
	{
		size_t rows = x.rows();
		size_t cols = x.cols();

		math::Matrix result(rows, cols);

		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				result.At(r, c) = ReLU(x.At(r, c));
			}
		}

		return result;
	}

	inline float ReLUPrime(float x)
	{
		return float(x > 0);
	}

	inline math::Matrix ReLUPrime(const math::Matrix& x)
	{
		size_t rows = x.rows();
		size_t cols = x.cols();

		math::Matrix result(rows, cols);

		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				result.At(r, c) = ReLUPrime(x.At(r, c));
			}
		}

		return result;
	}

	inline math::Matrix Softmax(const math::Matrix& z)
	{
		size_t rows = z.rows();
		size_t cols = z.cols();

		math::Matrix result(rows, cols);

		float maxVal = max(z);
		float expSum = 0.0f;

		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				result.At(r, c) = std::exp(z.At(r, c) - maxVal);
				expSum += result.At(r, c);
			}
		}

		for (size_t r = 0; r < rows; r++) {
			for (size_t c = 0; c < cols; c++) {
				result.At(r, c) /= expSum;
			}
		}

		return result;
	}
}