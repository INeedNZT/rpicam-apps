#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

#include "core/completed_request.hpp"
#include "core/rpicam_encoder.hpp"

#include "core/surv_options.hpp"

using Stream = libcamera::Stream;
using FrameBuffer = libcamera::FrameBuffer;

struct EventItem
{
	EventItem() : stream(nullptr) {}
	EventItem(CompletedRequestPtr &b, Stream *s, const std::vector<std::vector<float>> &vdb,
			  const std::vector<float> &vs)
		: completed_request(b), stream(s), detected_boxes(vdb), scores(vs)
	{
	}
	EventItem(EventItem &&other)
	{
		completed_request = std::move(other.completed_request);
		stream = other.stream;
		detected_boxes = std::move(other.detected_boxes);
		scores = std::move(other.scores);
		other.stream = nullptr;
		other.detected_boxes.clear();
		other.scores.clear();
	}
	EventItem &operator=(EventItem &&other)
	{
		completed_request = std::move(other.completed_request);
		stream = other.stream;
		detected_boxes = std::move(other.detected_boxes);
		scores = std::move(other.scores);
		other.stream = nullptr;
		other.detected_boxes.clear();
		other.scores.clear();
		return *this;
	}
	CompletedRequestPtr completed_request;
	Stream *stream;
	std::vector<std::vector<float>> detected_boxes;
	std::vector<float> scores;
};

class SentinelService
{
public:
	SentinelService(RPiCamEncoder<SurvOptions> *app)
		: running_(false), event_dir_(app->GetOptions()->event_directory),
		  event_interval_sec_(app->GetOptions()->event_interval), time_offset_(0), app_(app)
	{
	}

	virtual ~SentinelService() = default;

	void Start();
	void Stop();

	void RecordEvent(EventItem &&event_item);

	static std::unique_ptr<SentinelService> Create(RPiCamEncoder<SurvOptions> *app);

private:
	bool running_;

	std::string event_dir_;
	unsigned int event_interval_sec_;
	int64_t time_offset_;

	RPiCamEncoder<SurvOptions> *app_;

	std::mutex mutex_;
	std::condition_variable cv_;

	std::queue<EventItem> event_item_queue_;

	std::thread *event_loop_thread_;

	std::vector<uint8_t> frame_copy_;

	void run();
	void logEvent();
	void saveSnapshot(CompletedRequestPtr &completed_request, Stream *stream,
					  std::vector<std::vector<float>> &detected_boxes, std::vector<float> &scores);
};
