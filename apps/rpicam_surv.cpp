#include <chrono>
#include <poll.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <sys/stat.h>

#include "core/rpicam_encoder.hpp"
#include "output/surv_output.hpp"

#include "core/surv_options.hpp"
#include "web/web_server.hpp"

using namespace std::placeholders;

// Some keypress/signal handling.

class RPiCamSurvApp : public RPiCamEncoder<SurvOptions>
{
public:
	RPiCamSurvApp() : RPiCamEncoder<SurvOptions>(), web_server_(nullptr) {}

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

private:
	std::unique_ptr<WebServer> web_server_;
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

	// Start web server and surveillance recorder thread first
	app.StartWebServer();

	app.OpenCamera();
	app.ConfigureVideo(get_colourspace_flags(options->codec));
	app.StartEncoder();
	app.StartCamera();
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
		app.EncodeBuffer(completed_request, app.VideoStream());
		app.ShowPreview(completed_request, app.VideoStream());
	}
}

int main(int argc, char *argv[])
{
	try
	{
		RPiCamSurvApp app;
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
