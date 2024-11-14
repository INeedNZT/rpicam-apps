#pragma once

#include "video_options.hpp"

struct SurvOptions : public VideoOptions
{
	SurvOptions() : VideoOptions()
	{
		const char *home_directory = std::getenv("HOME");

		using namespace boost::program_options;
		options_.add_options()("record,r", value<bool>(&record)->default_value(false)->implicit_value(true),
							   "For launching local surveillance recording.")(
			"segment-duration", value<unsigned int>(&segment_duration)->default_value(15),
			"Set approximate length of each video segment in seconds for surveillance recording.")(
			"web-root-directory", value<std::string>(&web_root_directory)->default_value("/var/www/static"),
			"Path to the root directory of the Web server.")(
			"web-log-directory",
			value<std::string>(&web_log_directory)->default_value(std::string(home_directory) + "/web_logs"),
			"Path to store Web server log files.")(
			"footage-directory",
			value<std::string>(&footage_directory)->default_value(std::string(home_directory) + "/footage"),
			"Path to store playback of surveillance footage.")(
			"web-host", value<std::string>(&web_host)->default_value("127.0.0.1"),
			"Set the host address for the web server.")(
			"web-port", value<int>(&web_port)->default_value(8000),
			"Set the port for the web server.")(
			"max-connections", value<int>(&max_connections)->default_value(100),
			"Set the maximum number of connections to the server.")
			;
	}

	bool record;
	unsigned int segment_duration;
	std::string web_root_directory;
	std::string web_log_directory;
	std::string footage_directory;
	std::string web_host;
	int web_port;
	int max_connections;

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
		std::cerr << "    web-root-directory: " << web_root_directory << std::endl;
		std::cerr << "    web-log-directory: " << web_log_directory << std::endl;
		std::cerr << "    footage-directory: " << footage_directory << std::endl;
		std::cerr << "    web-host: " << web_host << std::endl;
		std::cerr << "    web-port: " << web_port << std::endl;
		std::cerr << "    max-connections: " << max_connections << std::endl;
	}
};
