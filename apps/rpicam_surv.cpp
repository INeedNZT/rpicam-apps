#include <chrono>
#include <filesystem>

#include "core/rpicam_encoder.hpp"
#include "output/surv_output.hpp"

#include "core/surv_options.hpp"
#include "sentinel/sentinel_service.hpp"
#include "web/web_server.hpp"

using namespace std::placeholders;

// Some keypress/signal handling.

class RPiCamSurvApp : public RPiCamEncoder<SurvOptions>
{
public:
	RPiCamSurvApp()
		: RPiCamEncoder<SurvOptions>(), web_server_(nullptr), sentinel_service_(nullptr), cleaner_running_(false),
		  enable_email_(false)
	{
	}

	void *GetValue() override { return &enable_email_; }

	void SetValue(void *v) override
	{
		if (v != nullptr)
		{
			enable_email_ = *static_cast<bool *>(v);
		}
	}

	void StartWebServer()
	{
		if (!web_server_)
		{
			web_server_ = WebServer::Create(this);
			web_server_->Start();
		}
	}

	void StopWebServer()
	{
		if (web_server_)
		{
			web_server_->Stop();
			web_server_.reset();
		}
	}

	void SendFrameData(void *mem, size_t size, int64_t timestamp_us, bool keyframe)
	{
		if (web_server_)
			web_server_->RecvFrameData(mem, size);
	}

	void StartSentinel()
	{
		if (!sentinel_service_)
		{
			enable_email_ = GetOptions()->enable_email_alerts;
			sentinel_service_ = SentinelService::Create(this);
			sentinel_service_->Start();
		}
	}

	void StopSentinel()
	{
		if (sentinel_service_)
		{
			sentinel_service_->Stop();
			sentinel_service_.reset();
		}
	}

	void InvokeSentinel(CompletedRequestPtr &completed_request, Stream *stream)
	{
		bool motion_detected = false;
		std::vector<std::vector<float>> detected_boxes;
		std::vector<float> detected_scores;

		completed_request->post_process_metadata.Get("motion_detect.result", motion_detected);
		completed_request->post_process_metadata.Get("face_detect.boxes", detected_boxes);
		completed_request->post_process_metadata.Get("face_detect.scores", detected_scores);

		EventItem item(completed_request, stream, motion_detected, detected_boxes, detected_scores);
		sentinel_service_->RecordEvent(item);
	}

	void StartDiskCleaner()
	{
		cleaner_running_ = true;
		disk_cleaner_thread_ = std::thread(&RPiCamSurvApp::cleanupCycle, this);
	}

	void StopDiskCleaner()
	{
		cleaner_running_ = false;
		cv_.notify_one();
		disk_cleaner_thread_.join();
	}

private:
	std::unique_ptr<WebServer> web_server_;
	std::unique_ptr<SentinelService> sentinel_service_;
	std::thread disk_cleaner_thread_;
	std::mutex mtx_;
	std::condition_variable cv_;
	bool cleaner_running_;
	bool enable_email_;

	void cleanupCycle()
	{
		SurvOptions const *options = static_cast<SurvOptions *>(options_.get());
		unsigned int days = options->retention_cycle;
		std::string event_directory = options->event_directory;
		std::string footage_directory = options->footage_directory;

		do
		{
			try
			{
				auto now = std::chrono::system_clock::now();
				auto now_time_t = std::chrono::system_clock::to_time_t(now);
				std::tm tm_now = *std::localtime(&now_time_t);

				tm_now.tm_hour = 0;
				tm_now.tm_min = 5;
				tm_now.tm_sec = 0;
				tm_now.tm_mday += 1;
				std::time_t midnight_timestamp = std::mktime(&tm_now);

				tm_now.tm_mday -= (days + 1);
				std::time_t retention_timestamp = std::mktime(&tm_now);

				for (const auto &dir : { event_directory, footage_directory })
				{
					for (const auto &entry : std::filesystem::directory_iterator(dir))
					{
						std::string folder_name = entry.path().filename().string();
						size_t pos = folder_name.find('_');
						if (pos != std::string::npos)
							folder_name.erase(pos);

						std::time_t folder_timestamp = static_cast<time_t>(std::stoll(folder_name));
						if (folder_timestamp < retention_timestamp)
						{
							std::filesystem::remove_all(entry.path());
							LOG(1, "Periodically cleanup folders " << entry);
						}
					}
				}

				auto midnight_time_point = std::chrono::system_clock::from_time_t(midnight_timestamp);
				auto diff_seconds = std::chrono::duration_cast<std::chrono::seconds>(midnight_time_point - now).count();

				if (diff_seconds > 0)
				{
					std::unique_lock<std::mutex> lock(mtx_);
					if (!cv_.wait_for(lock, std::chrono::seconds(diff_seconds), [this] { return !cleaner_running_; }))
					{
						// Not timeout, means cleaner_running_ set to false
						break;
					}
				}
			}
			catch (const std::exception &e)
			{
				LOG_ERROR("Disk Cleaner Error: *** " << e.what() << " ***");
			}
		} while (cleaner_running_);
	}
};

// The main even loop for the application.
static void event_loop(RPiCamSurvApp &app)
{
	SurvOptions const *options = app.GetOptions();
	std::unique_ptr<Output> output = std::unique_ptr<Output>(SurvOutput::Create(options));

	auto encode_output_ready_callback = std::bind(
		[&app, &output](auto &&...args)
		{
			app.SendFrameData(std::forward<decltype(args)>(args)...);
			output.get()->OutputReady(std::forward<decltype(args)>(args)...);
		},
		_1, _2, _3, _4);
	app.SetEncodeOutputReadyCallback(encode_output_ready_callback);
	app.SetMetadataReadyCallback(std::bind(&Output::MetadataReady, output.get(), _1));

	app.OpenCamera();
	app.ConfigureVideo(RPiCamEncoder<>::FLAG_VIDEO_NONE);
	app.StartEncoder();
	app.StartCamera();

	// Start web server and surveillance recorder thread
	app.StartWebServer();
	// Start event logger and risk alert thread
	app.StartSentinel();
	// Start disk cleaner for periodic disk space release
	app.StartDiskCleaner();

	SurvOptions::sys_start_timestamp =
		std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
			.count();
	auto start_time = std::chrono::high_resolution_clock::now();

	for (unsigned int count = 0;; count++)
	{
		RPiCamSurvApp::Msg msg = app.Wait();
		if (msg.type == RPiCamSurvApp::MsgType::Timeout)
		{
			LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
			app.StopCamera();
			app.StartCamera();
			continue;
		}
		if (msg.type == RPiCamSurvApp::MsgType::Quit)
			return;
		else if (msg.type != RPiCamSurvApp::MsgType::RequestComplete)
			throw std::runtime_error("unrecognised message!");

		LOG(2, "Viewfinder frame " << count);
		auto now = std::chrono::high_resolution_clock::now();
		bool timeout = !options->frames && options->timeout && ((now - start_time) > options->timeout.value);
		bool frameout = options->frames && count >= options->frames;
		if (timeout || frameout)
		{
			if (timeout)
				LOG(1, "Halting: reached timeout of " << options->timeout.get<std::chrono::milliseconds>()
													  << " milliseconds.");
			app.StopDiskCleaner();
			app.StopSentinel();
			app.StopWebServer();

			app.StopCamera(); // stop complains if encoder very slow to close
			app.StopEncoder();
			return;
		}

		CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);

		app.InvokeSentinel(completed_request, app.LoresStream());
		app.EncodeBuffer(completed_request, app.VideoStream());
	}
}

int main(int argc, char *argv[])
{
	// Exclude app in case try catch not working
	RPiCamSurvApp app;
	try
	{
		SurvOptions *options = app.GetOptions();
		if (options->Parse(argc, argv))
		{
			if (options->verbose >= 2)
				options->Print();

			event_loop(app);
		}
	}
	catch (std::exception const &e)
	{
		LOG_ERROR("ERROR: *** " << e.what() << " ***");
		return -1;
	}
	return 0;
}
