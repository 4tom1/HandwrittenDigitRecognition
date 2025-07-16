#include <indicators/progress_bar.hpp>

class MyProgressBar
{

	public:

		MyProgressBar(const std::string& postFixText)
		{
			progressBar = new indicators::ProgressBar{
				indicators::option::BarWidth{50},
				indicators::option::Start{"["},
				indicators::option::Fill{"#"},
				indicators::option::Lead{"#"},
				indicators::option::Remainder{" "},
				indicators::option::End{"]"},
				indicators::option::PostfixText{postFixText.c_str()},
				indicators::option::ForegroundColor{indicators::Color::unspecified},
				indicators::option::ShowPercentage{true},
				indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
			};
		}

		~MyProgressBar()
		{
			delete progressBar;
		}

		inline void Update(int progress)
		{
			progressBar->set_progress(progress);
		}

		inline indicators::ProgressBar& GetBar() { return *progressBar; }

		private:

		indicators::ProgressBar* progressBar;
};