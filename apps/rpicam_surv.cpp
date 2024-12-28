#include <chrono>
#include <filesystem>
#include <poll.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <sys/stat.h>

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
		: RPiCamEncoder<SurvOptions>(), web_server_(nullptr), sentinel_service_(nullptr), cleaner_running_(false)
	{
	}

	void StartWebServer()
	{
		if (!web_server_)
		{
			web_server_ = WebServer::Create(GetOptions());
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
		disk_cleaner_thread_.join();
	}

private:
	std::unique_ptr<WebServer> web_server_;
	std::unique_ptr<SentinelService> sentinel_service_;
	std::thread disk_cleaner_thread_;
	bool cleaner_running_;

	void cleanupCycle()
	{
		SurvOptions const *options = static_cast<SurvOptions *>(options_.get());
		unsigned int days = options->retention_cycle;
		std::string event_directory = options->event_directory;
		std::string footage_directory = options->footage_directory;

		while (cleaner_running_)
		{
			try
			{
				auto now = std::chrono::system_clock::now();
				auto now_time_t = std::chrono::system_clock::to_time_t(now);
				std::tm tm_now = *std::localtime(&now_time_t);

				tm_now.tm_hour = 0;
				tm_now.tm_min = 0;
				tm_now.tm_sec = 0;
				tm_now.tm_mday += 1;
				auto midnight_timestamp = std::mktime(&tm_now);

				tm_now.tm_mday -= (days);
				std::time_t retention_timestamp = std::mktime(&tm_now);

				auto midnight_time_point = std::chrono::system_clock::from_time_t(midnight_timestamp);
				auto diff_seconds = std::chrono::duration_cast<std::chrono::seconds>(midnight_time_point - now).count();

				if (diff_seconds > 0)
					std::this_thread::sleep_for(std::chrono::seconds(diff_seconds));

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
							LOG(1, "Periodically cleanup folders " << entry);
							std::filesystem::remove_all(entry.path());
						}
					}
				}
			}
			catch (const std::exception &e)
			{
				LOG_ERROR("Disk Cleaner Error: *** " << e.what() << " ***");
			}
		}
	}
};

static int signal_received;
static void default_signal_handler(int signal_number)
{
	signal_received = signal_number;
	LOG(1, "Received signal " << signal_number);
}

static int get_key_or_signal(SurvOptions const *options, pollfd p[1])
{
	int key = 0;
	if (signal_received == SIGINT)
		return 'x';
	if (options->keypress)
	{
		poll(p, 1, 0);
		if (p[0].revents & POLLIN)
		{
			char *user_string = nullptr;
			size_t len;
			[[maybe_unused]] size_t r = getline(&user_string, &len, stdin);
			key = user_string[0];
		}
	}
	if (options->signal)
	{
		if (signal_received == SIGUSR1)
			key = '\n';
		else if ((signal_received == SIGUSR2) || (signal_received == SIGPIPE))
			key = 'x';
		signal_received = 0;
	}
	return key;
}

static int get_colourspace_flags(std::string const &codec)
{
	if (codec == "mjpeg" || codec == "yuv420")
		return RPiCamSurvApp::FLAG_VIDEO_JPEG_COLOURSPACE;
	else
		return RPiCamSurvApp::FLAG_VIDEO_NONE;
}

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
	app.ConfigureVideo(get_colourspace_flags(options->codec));
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

	// Monitoring for keypresses and signals.
	signal(SIGUSR1, default_signal_handler);
	signal(SIGUSR2, default_signal_handler);
	signal(SIGINT, default_signal_handler);
	// SIGPIPE gets raised when trying to write to an already closed socket. This can happen, when
	// you're using TCP to stream to VLC and the user presses the stop button in VLC. Catching the
	// signal to be able to react on it, otherwise the app terminates.
	signal(SIGPIPE, default_signal_handler);
	pollfd p[1] = { { STDIN_FILENO, POLLIN, 0 } };

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
		int key = get_key_or_signal(options, p);
		if (key == '\n')
			output->Signal();

		LOG(2, "Viewfinder frame " << count);
		auto now = std::chrono::high_resolution_clock::now();
		bool timeout = !options->frames && options->timeout && ((now - start_time) > options->timeout.value);
		bool frameout = options->frames && count >= options->frames;
		if (timeout || frameout || key == 'x' || key == 'X')
		{
			if (timeout)
				LOG(1, "Halting: reached timeout of " << options->timeout.get<std::chrono::milliseconds>()
													  << " milliseconds.");
			app.StopCamera(); // stop complains if encoder very slow to close
			app.StopEncoder();
			return;
		}

		CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);

		app.InvokeSentinel(completed_request, app.LoresStream());
		app.EncodeBuffer(completed_request, app.VideoStream());
		app.ShowPreview(completed_request, app.VideoStream());
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
