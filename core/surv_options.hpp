#pragma once

#include "video_options.hpp"

struct SurvOptions : public VideoOptions
{
	SurvOptions() : VideoOptions()
	{
		using namespace boost::program_options;
		options_.add_options()("record,r", value<bool>(&record)->default_value(false)->implicit_value(true),
							   "For launching local surveillance recording.")(
			"segment-duration", value<unsigned int>(&segment_duration)->default_value(15),
			"Set approximate length of each video segment in seconds for surveillance recording. Default is 15 seconds.")(
			"hls-directory", value<std::string>(&hls_directory),
			"Path to store HLS video segments and playlist.");
	}

	bool record;
	unsigned int segment_duration;
	std::string hls_directory;

	virtual bool Parse(int argc, char *argv[]) override
	{
		if (VideoOptions::Parse(argc, argv) == false)
			return false;

		return true;
	}
	virtual void Print() const override
	{
		VideoOptions::Print();
		std::cerr << "    record: " << record << std::endl;
		std::cerr << "    segment-duration: " << segment_duration << std::endl;
		std::cerr << "    hls-directory: " << hls_directory << std::endl;
	}
};
