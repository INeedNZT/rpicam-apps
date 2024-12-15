#include <cstdio>
#include <filesystem>
#include <jpeglib.h>

#include "sentinel_service.hpp"

void SentinelService::Start()
{
	if (running_)
	{
		return;
	}

	running_ = true;

	event_loop_thread_ = new std::thread(&SentinelService::run, this);
}

void SentinelService::Stop()
{
	if (!running_)
	{
		return;
	}

	running_ = false;

	if (event_loop_thread_ && event_loop_thread_->joinable())
	{
		event_loop_thread_->join();
	}

	delete event_loop_thread_;
	event_loop_thread_ = nullptr;
}

void SentinelService::RecordEvent(EventItem &&event_item)
{
	event_item_queue_.push(std::move(event_item));
	cv_.notify_one();
}

void SentinelService::run()
{
	while (running_)
	{
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [this] { return !event_item_queue_.empty(); });
		EventItem item = std::move(event_item_queue_.front());
		event_item_queue_.pop();
		// logEvent();
		saveSnapshot(item.completed_request, item.stream, item.detected_boxes, item.scores);
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

void SentinelService::logEvent()
{
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

void SentinelService::saveSnapshot(CompletedRequestPtr &completed_request, Stream *stream,
								   std::vector<std::vector<float>> &detected_boxes, std::vector<float> &scores)
{
	FrameBuffer *buffer = completed_request->buffers[stream];
	BufferReadSync r(app_, buffer);
	libcamera::Span<uint8_t> span = r.Get()[0];

	auto ts = completed_request->metadata.get(controls::SensorTimestamp);
	int64_t timestamp_us = ts ? *ts : buffer->metadata().timestamp / 1000;

	if (time_offset_ == 0)
		time_offset_ = timestamp_us;

	int64_t sys_timestamp = SurvOptions::GetSysTimestamp(timestamp_us - time_offset_);
	time_t sys_time_sec = static_cast<time_t>(sys_timestamp / 1000000);

	frame_copy_.assign(span.data(), span.data() + span.size());

	StreamInfo lores_info = app_->GetStreamInfo(stream);

	for (size_t i = 0; i < detected_boxes.size(); ++i)
	{
		drawRectangle(frame_copy_.data(), detected_boxes[i], lores_info.width, lores_info.height, scores[i], 2);
	}

	FILE *fp = nullptr;
	uint8_t *jpeg_buffer = nullptr;
	unsigned long jpeg_len = 0;
	std::string filename = event_dir_ + "/" + std::to_string(sys_time_sec) + ".jpg";

	try
	{
		std::filesystem::create_directories(event_dir_);
		fp = fopen(filename.c_str(), "wb");
		YUV420_to_JPEG((uint8_t *)(frame_copy_.data()), lores_info, 90, 0, jpeg_buffer, jpeg_len);
		fwrite(jpeg_buffer, jpeg_len, 1, fp);
		fclose(fp);
		fp = nullptr;
	}
	catch (std::exception const &e)
	{
		if (fp)
			fclose(fp);
		free(jpeg_buffer);
		throw;
	}
}

std::unique_ptr<SentinelService> SentinelService::Create(RPiCamEncoder<SurvOptions> *app)
{
	return std::make_unique<SentinelService>(app);
}
