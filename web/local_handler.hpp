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

template <typename T>
static std::string toJSON(const std::vector<T> &vec, const std::string &format = "")
{
	std::ostringstream json;
	json << "[";
	for (size_t i = 0; i < vec.size(); ++i)
	{
		if constexpr (std::is_same<T, day_surv_footage>::value || std::is_same<T, hour_playlist>::value)
		{
			json << vec[i].to_json(format);
		}
		else
		{
			json << vec[i].to_json();
		}
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
