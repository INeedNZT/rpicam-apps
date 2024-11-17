#include <filesystem>

#include "local_handler.hpp"

void completeEndlist(const std::filesystem::path &m3u8_file_path)
{
	std::ifstream m3u8_file(m3u8_file_path);
	if (!m3u8_file.is_open())
		throw std::runtime_error("Failed to open m3u8 file");

	std::stringstream file_content;
	file_content << m3u8_file.rdbuf();
	std::string content = file_content.str();
	m3u8_file.close();

	if (content.find("#EXT-X-ENDLIST") == std::string::npos)
	{
		std::ofstream m3u8_file_out(m3u8_file_path, std::ios::app);
		if (!m3u8_file_out.is_open())
			throw std::runtime_error("Failed to open m3u8 file for appending");

		m3u8_file_out << "\n#EXT-X-ENDLIST\n";
		m3u8_file_out.close();
	}
}

std::vector<day_surv_footage> getDaySurvFootage(std::string footage_root_dir)
{
	std::vector<day_surv_footage> day_footage;
	for (const auto &entry : std::filesystem::directory_iterator(footage_root_dir))
	{
		if (entry.is_directory())
		{
			std::string folder_name = entry.path().filename().string();
			std::string thumb_path = std::string(FOOTAGE_PREFIX) + "/" + folder_name + "/" + THUMB_NAME;

			std::time_t timestamp = std::stoll(folder_name);
			day_footage.push_back({ thumb_path, timestamp });
		}
	}

	return day_footage;
}

std::vector<hour_playlist> getHourPlaylistByDate(std::string footage_root_dir, std::time_t date)
{
	std::vector<hour_playlist> playlists;
	std::filesystem::path date_directory = std::filesystem::path(footage_root_dir) / std::to_string(date);

	if (!std::filesystem::exists(date_directory) || !std::filesystem::is_directory(date_directory))
		throw std::runtime_error("Directory does not exist or is not a valid directory.");

	for (const auto &hour_entry : std::filesystem::directory_iterator(date_directory))
	{
		std::filesystem::path thumb_file_path;
		std::filesystem::path m3u8_file_path;

		if (hour_entry.is_directory())
		{
			std::filesystem::path hour_directory = hour_entry.path();
			std::time_t start_time = std::stoll(hour_directory.filename().string());

			std::filesystem::path thumb_file_path;
			std::filesystem::path m3u8_file_path;

			for (const auto &file_entry : std::filesystem::directory_iterator(hour_directory))
			{
				if (file_entry.is_regular_file())
				{
					if (file_entry.path().extension() == ".jpg")
					{
						thumb_file_path = file_entry.path();
					}
					else if (file_entry.path().extension() == ".m3u8")
					{
						m3u8_file_path = file_entry.path();
					}
				}
			}

			if (thumb_file_path.empty())
				throw std::runtime_error("No thumb file (.jpg) found in the directory: " + hour_directory.string());

			if (m3u8_file_path.empty())
				throw std::runtime_error("No m3u8 file found in the directory: " + hour_directory.string());

			completeEndlist(m3u8_file_path);

			std::string thumb_file_name = std::string(FOOTAGE_PREFIX) + "/" + date_directory.filename().string() + "/" +
										  hour_directory.filename().string() + "/" +
										  thumb_file_path.filename().string();
			std::string m3u8_file_name = std::string(FOOTAGE_PREFIX) + "/" + date_directory.filename().string() + "/" +
										 hour_directory.filename().string() + "/" + m3u8_file_path.filename().string();

			playlists.push_back({ thumb_file_name, m3u8_file_name, start_time });
		}
	}

	return playlists;
}
