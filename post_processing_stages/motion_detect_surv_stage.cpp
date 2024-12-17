// This file is mostly identical to motion_detect_stage.cpp. To ensure compatibility with subsequent face detection, 
// the resolution must be set to 320x240, and the ROI (Region of Interest) is set to the entire image.
// In addition, the concept of frame_period has been removed, and detection will be performed on every frame.

#include <libcamera/stream.h>

#include "core/rpicam_app.hpp"

#include "post_processing_stages/post_processing_stage.hpp"

using Stream = libcamera::Stream;

class MotionDetectSurvStage : public PostProcessingStage
{
public:
	MotionDetectSurvStage(RPiCamApp *app) : PostProcessingStage(app) {}

	char const *Name() const override;

	void Read(boost::property_tree::ptree const &params) override;

	void Configure() override;

	bool Process(CompletedRequestPtr &completed_request) override;

private:
	struct Config
	{
		int hskip, vskip;
		float difference_m;
		int difference_c;
		float region_threshold;
		bool verbose;
	} config_;
	Stream *stream_;
	unsigned lores_stride_;
	unsigned int width_, height_;
	unsigned int region_threshold_;
	std::vector<uint8_t> previous_frame_;
	bool first_time_;
	bool motion_detected_;
	std::mutex mutex_;
};

#define NAME "motion_detect_surv"

char const *MotionDetectSurvStage::Name() const
{
	return NAME;
}

void MotionDetectSurvStage::Read(boost::property_tree::ptree const &params)
{
	config_.hskip = params.get<int>("hskip", 2);
	config_.vskip = params.get<int>("vskip", 2);
	config_.difference_m = params.get<float>("difference_m", 0.1);
	config_.difference_c = params.get<int>("difference_c", 10);
	config_.region_threshold = params.get<float>("region_threshold", 0.005);
	config_.verbose = params.get<int>("verbose", 0);
}

void MotionDetectSurvStage::Configure()
{
	StreamInfo info;
	stream_ = app_->LoresStream(&info);
	if (!stream_)
		return;

	config_.hskip = std::max(config_.hskip, 1);
	config_.vskip = std::max(config_.vskip, 1);
	info.width /= config_.hskip;
	info.height /= config_.vskip;
	lores_stride_ = info.stride * config_.vskip;

	width_ = info.width;
	height_ = info.height;
	region_threshold_ = config_.region_threshold * width_ * height_;

	if (config_.verbose)
		LOG(1, "Lores: " << info.width << "x" << info.height << " threshold: " << region_threshold_);

	previous_frame_.resize(width_ * height_);
	first_time_ = true;
	motion_detected_ = false;
}

bool MotionDetectSurvStage::Process(CompletedRequestPtr &completed_request)
{
	if (!stream_)
		return false;

	BufferReadSync r(app_, completed_request->buffers[stream_]);
	libcamera::Span<uint8_t> buffer = r.Get()[0];
	uint8_t *image = buffer.data();

	std::lock_guard<std::mutex> lock(mutex_);

	if (first_time_)
	{
		first_time_ = false;
		for (unsigned int y = 0; y < height_; y++)
		{
			uint8_t *new_value_ptr = image + y * lores_stride_;
			uint8_t *old_value_ptr = &previous_frame_[0] + y * width_;
			for (unsigned int x = 0; x < width_; x++, new_value_ptr += config_.hskip)
				*(old_value_ptr++) = *new_value_ptr;
		}

		completed_request->post_process_metadata.Set("motion_detect.result", motion_detected_);

		return false;
	}

	bool motion_detected = false;
	unsigned int regions = 0;

	for (unsigned int y = 0; y < height_; y++)
	{
		uint8_t *new_value_ptr = image + y * lores_stride_;
		uint8_t *old_value_ptr = &previous_frame_[0] + y * width_;
		for (unsigned int x = 0; x < width_; x++, new_value_ptr += config_.hskip)
		{
			int new_value = *new_value_ptr;
			int old_value = *old_value_ptr;
			*(old_value_ptr++) = new_value;
			regions += std::abs(new_value - old_value) > config_.difference_m * old_value + config_.difference_c;
			motion_detected = regions >= region_threshold_;
		}
	}

	if (config_.verbose && motion_detected != motion_detected_)
		LOG(1, "Motion " << (motion_detected ? "detected" : "stopped"));

	motion_detected_ = motion_detected;
	completed_request->post_process_metadata.Set("motion_detect.result", motion_detected);

	return false;
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new MotionDetectSurvStage(app);
}

static RegisterStage reg(NAME, &Create);
