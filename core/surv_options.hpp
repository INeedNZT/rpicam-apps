#pragma once

#include <ctime>

#include "video_options.hpp"

#define THUMB_NAME "thumbnail.jpg"
#define THUMB_WIDTH 640
#define THUMB_HEIGHT 480
#define PLAYLIST_NAME "playlist.m3u8"
#define FOOTAGE_PREFIX "/footage"
#define EVENT_PREFIX "/events"
#define EVENT_LOG_FILE "log.txt"

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
			"Set the host address for the web server.")("web-port", value<int>(&web_port)->default_value(8000),
														"Set the port for the web server.")(
			"max-connections", value<int>(&max_connections)->default_value(100),
			"Set the maximum number of connections to the server.")(
			"footage-date-format", value<std::string>(&footage_date_format)->default_value("%Y-%m-%d"),
			"Set the date format used in the footage file names.")(
			"playlist-time-format", value<std::string>(&playlist_time_format)->default_value("%H:%M:%S"),
			"Set the time format used in the playlist.")(
			"event-directory",
			value<std::string>(&event_directory)->default_value(std::string(home_directory) + "/events"),
			"Path to store the security event logs and snapshots.")(
			"event-interval", value<unsigned int>(&event_interval)->default_value(300),
			"Set the time interval in seconds between events before the next event check.")(
			"save-rate", value<unsigned int>(&save_rate)->default_value(30),
			"Set the frame rate for saving an event, including log and snapshot image.")(
			"alert-config-file", value<std::string>(&alert_config_file),
			"Set the file name for configuring the security alert.")(
			"retention-cycle", value<unsigned int>(&retention_cycle)->default_value(3),
			"Set the retention cycle in days for storage.")
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
	std::string footage_date_format;
	std::string playlist_time_format;

	std::string event_directory;
	unsigned int event_interval;
	unsigned int save_rate;

	std::string alert_config_file;

	unsigned int retention_cycle;

	static int64_t sys_start_timestamp;

	static inline std::string ToTimeStr(time_t time_seconds, std::string time_format)
	{
		std::tm tm = *std::localtime(&time_seconds);

		char buffer[16];
		strftime(buffer, sizeof(buffer), time_format.c_str(), &tm);
		return std::string(buffer);
	}

	static inline int64_t GetSysTimestamp(int64_t timestamp_us)
	{
		return sys_start_timestamp + timestamp_us;
	}

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
		std::cerr << "    footage-date-format: " << footage_date_format << std::endl;
		std::cerr << "    playlist-time-format: " << playlist_time_format << std::endl;
		std::cerr << "    event-directory: " << event_directory << std::endl;
		std::cerr << "    event-interval: " << event_interval << std::endl;
		std::cerr << "    save-rate: " << save_rate << std::endl;
		std::cerr << "    alert-config-file: " << alert_config_file << std::endl;
		std::cerr << "    retention-cycle: " << retention_cycle << std::endl;
	}
};
