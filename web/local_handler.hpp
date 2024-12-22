#pragma once

#include <sstream>
#include <string>
#include <vector>

#include "core/surv_options.hpp"

struct day_surv_footage
{
	std::string thumb_path;
	std::time_t date;
	std::string date_str;

	day_surv_footage(const std::string &thumb_path, std::time_t date) : thumb_path(thumb_path), date(date), date_str("")
	{
	}

	std::string to_json(std::string date_format) const
	{
		std::ostringstream json;
		json << "{\"thumb_path\": \"" << thumb_path << "\", \"date\": \"" << date << "\", \"date_str\": \""
			 << SurvOptions::ToTimeStr(date, date_format) << "\"}";
		return json.str();
	}
};

struct hour_playlist
{
	std::string thumb_path;
	std::string m3u8_path;
	std::time_t start_time;
	std::string start_time_str;

	hour_playlist(const std::string &thumb_path, const std::string &m3u8_path, std::time_t start_time)
		: thumb_path(thumb_path), m3u8_path(m3u8_path), start_time(start_time), start_time_str("")
	{
	}

	std::string to_json(std::string time_format) const
	{
		std::ostringstream json;
		json << "{\"thumb_path\": \"" << thumb_path << "\", \"m3u8_path\": \"" << m3u8_path << "\", \"start_time\": \""
			 << start_time << "\", \"start_time_str\": \"" << SurvOptions::ToTimeStr(start_time, time_format) << "\"}";
		return json.str();
	}
};

enum class event_type
{
	None,
	Motion,
	FaceRecognition
};

struct event
{
	std::string event_id;
	event_type type = event_type::None;
	std::time_t start_time;
	std::string start_date_str;
	std::string start_time_str;

	std::string to_json(std::string date_format, std::string time_format) const
	{
		std::ostringstream json;
		json << "{\"event_id\": \"" << event_id << "\", \"type\": \"" << static_cast<int>(type)
			 << "\", \"start_time\": \"" << start_time << "\", \"start_date_str\": \""
			 << SurvOptions::ToTimeStr(start_time, date_format) << "\", \"start_time_str\": \""
			 << SurvOptions::ToTimeStr(start_time, time_format) << "\"}";
		return json.str();
	}
};

struct event_log
{
	event_type type = event_type::None;
	std::string snapshot_path;
	std::time_t log_time;
	std::string log_time_str;

	std::string to_json(std::string time_format) const
	{
		std::ostringstream json;
		json << "{\"type\": \"" << static_cast<int>(type) << "\", "
			 << "\"snapshot_path\": \"" << snapshot_path << "\", "
			 << "\"log_time\": \"" << log_time << "\", "
			 << "\"log_time_str\": \"" << SurvOptions::ToTimeStr(log_time, time_format) << "\"}";
		return json.str();
	}
};

template <typename T, typename... Args>
static std::string toJSON(const std::vector<T> &vec, const Args &...args)
{
	std::ostringstream json;
	json << "[";
	for (size_t i = 0; i < vec.size(); ++i)
	{
		json << vec[i].to_json(args...);

		if (i != vec.size() - 1)
		{
			json << ", ";
		}
	}
	json << "]";
	return json.str();
}

std::vector<day_surv_footage> getDaySurvFootage(std::string footage_root_dir);
std::vector<hour_playlist> getHourPlaylistByDate(std::string footage_root_dir, std::time_t date);
std::vector<event> getEventListByDate(std::string event_root_dir, std::time_t date);
std::vector<event_log> getEventLogsById(std::string event_root_dir, std::string event_id);
