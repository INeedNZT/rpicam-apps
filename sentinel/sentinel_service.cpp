#include <cstdio>
#include <filesystem>
#include <jpeglib.h>

#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "sentinel_service.hpp"

#define WARNING_SENDED 1
#define DANGER_SENDED 2

void SentinelService::Start()
{
	if (running_)
		return;

	loadAlertConfig();

	running_ = true;

	std::filesystem::create_directories(event_root_dir_);

	event_loop_thread_ = std::thread(&SentinelService::run, this);
}

void SentinelService::Stop()
{
	if (!running_)
		return;

	running_ = false;
	cv_.notify_one();

	event_loop_thread_.join();
}

void SentinelService::RecordEvent(EventItem &event_item)
{
	CompletedRequestPtr completed_request = event_item.completed_request;
	FrameBuffer *buffer = completed_request->buffers[event_item.stream];
	BufferReadSync r(app_, buffer);
	libcamera::Span<uint8_t> span = r.Get()[0];

	// Cache frame buffer and release completedRequestPtr, DO NOT block the video stream
	frame_copy_.assign(span.data(), span.data() + span.size());
	event_item.frame_sequence = completed_request->sequence;
	event_item.completed_request.reset();

	auto ts = completed_request->metadata.get(controls::SensorTimestamp);
	int64_t timestamp_ns = ts ? *ts : buffer->metadata().timestamp;
	event_item.timestamp_us = timestamp_ns / 1000;

	event_item_queue_.push(std::move(event_item));
	cv_.notify_one();
}

void SentinelService::run()
{
	while (running_)
	{
		try
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cv_.wait(lock, [this] { return !event_item_queue_.empty() || !running_; });

			if (!running_)
				break;

			EventItem item = std::move(event_item_queue_.front());
			event_item_queue_.pop();

			if (time_offset_ == 0)
				time_offset_ = item.timestamp_us;

			if (!item.motion_detected && (item.detected_boxes.empty() || item.scores.empty()))
				continue;

			// An event is happening, start recording the timestamp
			if (event_start_time_ == 0)
				event_start_time_ = item.timestamp_us;

			if (event_start_time_ == item.timestamp_us || item.frame_sequence % event_save_rate_ == 0)
			{
				logEvent(item.motion_detected, item.detected_boxes, item.scores, item.timestamp_us);
				StreamInfo stream_info = app_->GetStreamInfo(item.stream);
				std::shared_ptr<uint8_t[]> jpeg_buffer_ptr;
				size_t jpeg_buffer_size = 0;
				saveSnapshot(item.motion_detected, item.detected_boxes, item.scores, item.timestamp_us, stream_info,
							 jpeg_buffer_ptr, jpeg_buffer_size);
#if LIBCURL_PRESENT
				if (event_start_time_ == item.timestamp_us || event_notif_flag_ == WARNING_SENDED)
				{
					alert al;
					al.type = alert_type::None;

					if (item.motion_detected)
						al.type = alert_type::Motion;
					if (item.detected_boxes.size() && item.scores.size())
						al.type = alert_type::FaceRecognition;

					if (event_notif_flag_ == WARNING_SENDED && al.type != alert_type::FaceRecognition)
						continue;

					std::string date_format = app_->GetOptions()->footage_date_format;
					std::string time_format = app_->GetOptions()->playlist_time_format;
					int64_t sys_timestamp = SurvOptions::GetSysTimestamp(event_start_time_ - time_offset_);
					time_t sys_time_sec = static_cast<time_t>(sys_timestamp / 1000000);
					al.time_str = SurvOptions::ToTimeStr(sys_time_sec, date_format) + " " +
								  SurvOptions::ToTimeStr(sys_time_sec, time_format);
					al.jpeg_buffer_ptr = jpeg_buffer_ptr;
					al.jpeg_buffer_size = jpeg_buffer_size;
					email_service_.SendAlert(al);

					if (al.type == alert_type::Motion)
						event_notif_flag_ = WARNING_SENDED;
					if (al.type == alert_type::FaceRecognition)
						event_notif_flag_ = DANGER_SENDED;
				}
#endif
			}

			if (item.timestamp_us - event_start_time_ >= event_interval_sec_ * 1000000)
			{
				// End event and clear resource
				event_dir_.clear();
				log_file_.close();
				event_start_time_ = 0;
				event_notif_flag_ = 0;
			}
			else if (event_start_time_ != 0)
			{
				// Extend time if there is an event
				event_start_time_ = item.timestamp_us;
			}
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Sentinel Error: *** " << e.what() << " ***");
		}
	}
}

static void drawRectangle(uint8_t *yuv420_buffer, const std::vector<float> &box, int width, int height, float score,
						  int thickness)
{
	uint8_t v = 240 - (240 - 64) * (1 - score) / (1 - 0.6f); // expand the weight based on confidence score

	int x1 = static_cast<int>(box[0] * width);
	int y1 = static_cast<int>(box[1] * height);
	int x2 = static_cast<int>(box[2] * width);
	int y2 = static_cast<int>(box[3] * height);

	int y_size = width * height;
	int u_size = (width / 2) * (height / 2);

	uint8_t *u_data = yuv420_buffer + y_size;
	uint8_t *v_data = u_data + u_size;

	for (int t = 0; t < thickness; ++t)
	{
		// Top border
		for (int x = x1 - t; x <= x2 + t; ++x)
		{
			if (y1 - t >= 0 && y1 - t < height && x >= 0 && x < width)
			{
				int y_index = (y1 - t) * width + x;
				int u_index = ((y1 - t) / 2) * (width / 2) + (x / 2);
				int v_index = u_index;

				yuv420_buffer[y_index] = 128;

				u_data[u_index] = 128;
				v_data[v_index] = v;
			}
		}

		// Bottom border
		for (int x = x1 - t; x <= x2 + t; ++x)
		{
			if (y2 + t >= 0 && y2 + t < height && x >= 0 && x < width)
			{
				int y_index = (y2 + t) * width + x;
				int u_index = ((y2 + t) / 2) * (width / 2) + (x / 2);
				int v_index = u_index;

				yuv420_buffer[y_index] = 128;

				u_data[u_index] = 128;
				v_data[v_index] = v;
			}
		}

		// Left border
		for (int y = y1 - t; y <= y2 + t; ++y)
		{
			if (x1 - t >= 0 && x1 - t < width && y >= 0 && y < height)
			{
				int y_index = y * width + (x1 - t);
				int u_index = (y / 2) * (width / 2) + ((x1 - t) / 2);
				int v_index = u_index;

				yuv420_buffer[y_index] = 128;

				u_data[u_index] = 128;
				v_data[v_index] = v;
			}
		}

		for (int y = y1 - t; y <= y2 + t; ++y)
		{
			if (x2 + t >= 0 && x2 + t < width && y >= 0 && y < height)
			{
				int y_index = y * width + (x2 + t);
				int u_index = (y / 2) * (width / 2) + ((x2 + t) / 2);
				int v_index = u_index;

				yuv420_buffer[y_index] = 128;

				u_data[u_index] = 128;
				v_data[v_index] = v;
			}
		}
	}
}

void SentinelService::logEvent(bool motion_detected, std::vector<std::vector<float>> &detected_boxes,
							   std::vector<float> &scores, int64_t timestamp_us)
{
	int64_t sys_timestamp = SurvOptions::GetSysTimestamp(timestamp_us - time_offset_);
	time_t sys_time_sec = static_cast<time_t>(sys_timestamp / 1000000);

	if (event_dir_.empty())
	{
		event_dir_ = event_root_dir_ + "/" + std::to_string(sys_time_sec);
		std::filesystem::create_directories(event_dir_);
	}

	std::string old_dir = event_dir_;
	if (motion_detected && event_dir_.find("_m") == std::string::npos)
		event_dir_ += "_m";
	if (!detected_boxes.empty() && !scores.empty() && event_dir_.find("_f") == std::string::npos)
		event_dir_ += "_f";
	std::filesystem::rename(old_dir, event_dir_);

	if (!log_file_.is_open())
	{
		std::string log_path = event_dir_ + "/" + EVENT_LOG_FILE;
		log_file_.open(log_path, std::ios::out | std::ios::app);
		if (!log_file_)
			throw std::runtime_error("Failed to create event log file");
	}

	log_file_ << sys_time_sec << ":" << (motion_detected ? "m" : "")
			  << ((!detected_boxes.empty() && !scores.empty()) ? "f" : "") << "\n";
	log_file_.flush();
}

static void YUV420_to_JPEG(const uint8_t *input, const StreamInfo &info, const int quality, const unsigned int restart,
						   uint8_t *&jpeg_buffer, unsigned long &jpeg_len)
{
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	cinfo.image_width = info.width;
	cinfo.image_height = info.height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_YCbCr;
	cinfo.restart_interval = restart;

	jpeg_set_defaults(&cinfo);
	cinfo.raw_data_in = TRUE;
	jpeg_set_quality(&cinfo, quality, TRUE);
	jpeg_mem_dest(&cinfo, &jpeg_buffer, &jpeg_len);

	jpeg_start_compress(&cinfo, TRUE);

	const uint8_t *Y = input;
	const uint8_t *U = Y + info.stride * info.height;
	const uint8_t *V = U + (info.stride / 2) * (info.height / 2);

	const uint8_t *Y_max = Y + info.stride * info.height;
	const uint8_t *U_max = U + (info.stride / 2) * (info.height / 2);
	const uint8_t *V_max = V + (info.stride / 2) * (info.height / 2);

	JSAMPROW y_rows[16];
	JSAMPROW u_rows[8];
	JSAMPROW v_rows[8];

	for (uint8_t *Y_row = (uint8_t *)Y, *U_row = (uint8_t *)U, *V_row = (uint8_t *)V;
		 cinfo.next_scanline < info.height;)
	{
		for (int i = 0; i < 16; i++, Y_row += info.stride)
			y_rows[i] = std::min(Y_row, (uint8_t *)Y_max);

		for (int i = 0; i < 8; i++, U_row += (info.stride / 2), V_row += (info.stride / 2))
			u_rows[i] = std::min(U_row, (uint8_t *)U_max), v_rows[i] = std::min(V_row, (uint8_t *)V_max);

		JSAMPARRAY rows[] = { y_rows, u_rows, v_rows };
		jpeg_write_raw_data(&cinfo, rows, 16);
	}

	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}

void SentinelService::saveSnapshot(bool motion_detected, std::vector<std::vector<float>> &detected_boxes,
								   std::vector<float> &scores, int64_t timestamp_us, StreamInfo stream_info,
								   std::shared_ptr<uint8_t[]> &jpeg_buffer_ptr, size_t &jpeg_buffer_size)
{
	for (size_t i = 0; i < detected_boxes.size(); ++i)
	{
		drawRectangle(frame_copy_.data(), detected_boxes[i], stream_info.width, stream_info.height, scores[i], 2);
	}

	FILE *fp = nullptr;
	uint8_t *jpeg_buffer = nullptr;
	unsigned long jpeg_len = 0;

	int64_t sys_timestamp = SurvOptions::GetSysTimestamp(timestamp_us - time_offset_);
	time_t sys_time_sec = static_cast<time_t>(sys_timestamp / 1000000);
	std::string filename = event_dir_ + "/" + std::to_string(sys_time_sec) + ".jpg";

	try
	{
		fp = fopen(filename.c_str(), "wb");
		YUV420_to_JPEG((uint8_t *)(frame_copy_.data()), stream_info, 90, 0, jpeg_buffer, jpeg_len);
		jpeg_buffer_ptr.reset(jpeg_buffer, std::default_delete<uint8_t[]>());
		jpeg_buffer_size = jpeg_len;

		fwrite(jpeg_buffer_ptr.get(), jpeg_len, 1, fp);
		fclose(fp);
		fp = nullptr;
	}
	catch (std::exception const &e)
	{
		if (fp)
			fclose(fp);
		throw;
	}
}

void SentinelService::loadAlertConfig()
{
	std::string filename = app_->GetOptions()->alert_config_file;
	boost::property_tree::ptree root;
	boost::property_tree::read_json(filename, root);

	for (auto const &key_and_value : root)
	{
		if (key_and_value.first == "email_service")
		{
			boost::property_tree::ptree const &email_params = key_and_value.second;
#if LIBCURL_PRESENT
			email_service_.LoadConfig(email_params);
#endif
		}
	}
}

std::unique_ptr<SentinelService> SentinelService::Create(RPiCamEncoder<SurvOptions> *app)
{
	return std::make_unique<SentinelService>(app);
}
