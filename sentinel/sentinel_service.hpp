#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "core/completed_request.hpp"
#include "core/rpicam_encoder.hpp"

#include "core/surv_options.hpp"
#include "alert_service.hpp"

using Stream = libcamera::Stream;
using FrameBuffer = libcamera::FrameBuffer;

struct EventItem
{
	EventItem() : stream(nullptr) {}
	EventItem(CompletedRequestPtr &b, Stream *s, bool md, const std::vector<std::vector<float>> &vdb,
			  const std::vector<float> &vs)
		: completed_request(b), stream(s), motion_detected(md), detected_boxes(vdb), scores(vs)
	{
	}
	EventItem(EventItem &&other)
	{
		completed_request = std::move(other.completed_request);
		stream = other.stream;
		motion_detected = other.motion_detected;
		detected_boxes = std::move(other.detected_boxes);
		scores = std::move(other.scores);
		other.stream = nullptr;
		other.motion_detected = false;
		other.detected_boxes.clear();
		other.scores.clear();
	}
	EventItem &operator=(EventItem &&other)
	{
		completed_request = std::move(other.completed_request);
		stream = other.stream;
		motion_detected = other.motion_detected;
		detected_boxes = std::move(other.detected_boxes);
		scores = std::move(other.scores);
		other.stream = nullptr;
		other.motion_detected = false;
		other.detected_boxes.clear();
		other.scores.clear();
		return *this;
	}
	CompletedRequestPtr completed_request;
	Stream *stream;
	bool motion_detected;
	std::vector<std::vector<float>> detected_boxes;
	std::vector<float> scores;
};

class SentinelService
{
public:
	SentinelService(RPiCamEncoder<SurvOptions> *app)
		: running_(false), time_offset_(0), event_start_time_(0), event_notif_flag_(0), event_dir_(""),
		  event_root_dir_(app->GetOptions()->event_directory), event_save_rate_(app->GetOptions()->save_rate),
		  event_interval_sec_(app->GetOptions()->event_interval), app_(app)
	{
	}

	virtual ~SentinelService() = default;

	void Start();
	void Stop();

	void RecordEvent(EventItem &event_item);

	static std::unique_ptr<SentinelService> Create(RPiCamEncoder<SurvOptions> *app);

private:
	bool running_;
	int64_t time_offset_;
	int64_t event_start_time_;
	int64_t event_notif_flag_;
	std::string event_dir_;
	std::string event_root_dir_;
	unsigned int event_save_rate_;
	unsigned int event_interval_sec_;
	std::ofstream log_file_;

	RPiCamEncoder<SurvOptions> *app_;

	std::mutex mutex_;
	std::condition_variable cv_;
	std::queue<EventItem> event_item_queue_;
	std::thread *event_loop_thread_;
	std::vector<uint8_t> frame_copy_;

#if LIBCURL_PRESENT
	EmailService email_service_;
#endif

	void run();
	void logEvent(bool motion_detected, std::vector<std::vector<float>> &detected_boxes, std::vector<float> &scores,
				  int64_t timestamp_us);
	void saveSnapshot(bool motion_detected, std::vector<std::vector<float>> &detected_boxes, std::vector<float> &scores,
					  int64_t timestamp_us, StreamInfo stream_info, std::shared_ptr<uint8_t[]> &jpeg_buffer_ptr,
					  size_t &jpeg_buffer_size);
	void loadAlertConfig();
};
