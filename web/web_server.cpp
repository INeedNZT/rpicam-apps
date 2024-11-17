#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "local_handler.hpp"
#include "mongoose.h"
#include "web_server.hpp"

struct Log
{
	std::string const log_dir;
	std::string log_fname;
	std::ofstream log_ofs;
	bool new_line;
	std::mutex log_mutex;

	Log(const std::string &log_dir) : log_dir(log_dir), log_fname(""), new_line(true), log_mutex() {}

	~Log()
	{
		if (log_ofs.is_open())
			log_ofs.close();
	}
};

static void log_fn(char c, void *param)
{
	struct Log *log = static_cast<Log *>(param);
	std::string current_date = SurvOptions::ToTimeStr(std::time(nullptr), "%Y-%m-%d");

	if (log->log_fname != current_date)
	{
		if (log->log_ofs.is_open())
		{
			log->log_ofs.close();
		}

		log->log_fname = current_date;

		if (!std::filesystem::exists(log->log_dir))
			std::filesystem::create_directories(log->log_dir);

		std::string log_file_path = log->log_dir + "/" + log->log_fname + ".log";
		log->log_ofs.open(log_file_path, std::ios::out | std::ios::app);
	}

	if (log->log_ofs.is_open())
	{
		std::lock_guard<std::mutex> guard(log->log_mutex);

		if (log->new_line)
		{
			std::string current_time = SurvOptions::ToTimeStr(std::time(nullptr), "%H:%M:%S");
			log->log_ofs << "[" << current_time << "] ";
			log->new_line = false;
		}

		log->log_ofs << c;

		if (c == '\n')
		{
			log->new_line = true;
			log->log_ofs.flush();
		}
	}
	else
	{
		std::cerr << "Error opening log file" << std::endl;
	}
}

class MongooseServer : public WebServer
{
public:
	MongooseServer(SurvOptions const *options) : WebServer(options), event_loop_thread_(nullptr)
	{
		footage_dir_ = options->footage_directory;
		page404_ = options->web_root_directory + "/404.html";
		root_dir_ = options->web_root_directory + "," + FOOTAGE_PREFIX + "=" + options->footage_directory;
		http_server_options_ = {};
		http_server_options_.page404 = page404_.c_str();
		http_server_options_.root_dir = root_dir_.c_str();

		date_format_ = options->footage_date_format;
		time_format_ = options->playlist_time_format;

		Log *log = new Log { options->web_log_directory };

		mg_log_set_fn(log_fn, static_cast<void *>(log));
		mg_mgr_init(&mgr_);
	}

	~MongooseServer() { Stop(); }

	void Start() override
	{
		if (running_)
		{
			return;
		}

		std::string url = host_ + ":" + std::to_string(port_);
		mg_http_listen(&mgr_, url.c_str(), eventHandler, this);

		running_ = true;

		event_loop_thread_ = new std::thread(&MongooseServer::run, this);
	}

	void Stop() override
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

		mg_mgr_free(&mgr_);
	}

private:
	std::string page404_;
	std::string root_dir_;
	std::string footage_dir_;

	std::string date_format_;
	std::string time_format_;

	struct mg_mgr mgr_;
	std::thread *event_loop_thread_;
	struct mg_http_serve_opts http_server_options_;

	static inline int numconns(struct mg_mgr *mgr_)
	{
		int count = 0;
		for (struct mg_connection *c = mgr_->conns; c != nullptr; c = c->next)
		{
			count++;
		}
		return count;
	}

	template <typename Func, typename... Args>
	static void handler_wrapper(Func &&fn, struct mg_connection *c, struct mg_http_message *hm, Args &&...args)
	{
		try
		{
			std::forward<Func>(fn)(c, hm, std::forward<Args>(args)...);
		}
		catch (const std::exception &e)
		{
			mg_http_reply(c, 500, "Content-Type: text/plain\r\n", "Internal Server Error: %s\n", e.what());
		}
	}

	static void surv_footage_handler(struct mg_connection *c, struct mg_http_message *hm, std::string footage_dir, std::string date_format)
	{
		std::vector<day_surv_footage> footage = getDaySurvFootage(footage_dir);
		mg_http_reply(c, 200, "Content-Type: application/json\r\n", toJSON(footage, date_format).c_str());
	}

	static void footage_playlist_handler(struct mg_connection *c, struct mg_http_message *hm, std::string footage_dir, std::string time_format)
	{
		long timestamp_long = mg_json_get_long(hm->body, "$.st", -1);
		if (timestamp_long == -1)
			throw std::invalid_argument("Invalid timestamp value");
		std::vector<hour_playlist> playlist = getHourPlaylistByDate(footage_dir, static_cast<time_t>(timestamp_long));
		mg_http_reply(c, 200, "Content-Type: application/json\r\n", toJSON(playlist, time_format).c_str());
	}

	static void eventHandler(struct mg_connection *c, int ev, void *ev_data)
	{
		MongooseServer *server = static_cast<MongooseServer *>(c->fn_data);

		if (ev == MG_EV_ACCEPT)
		{
			if (numconns(&server->mgr_) > server->max_connections_)
			{
				MG_ERROR(("Too many connections"));
				c->is_closing = 1;
			}
		}

		if (ev == MG_EV_HTTP_MSG)
		{
			struct mg_http_message *hm = (struct mg_http_message *)ev_data;

			if (mg_match(hm->uri, mg_str("/api/survfootage"), NULL))
			{
				handler_wrapper(surv_footage_handler, c, hm, server->footage_dir_, server->date_format_);
			}
			else if (mg_match(hm->uri, mg_str("/api/playlist"), NULL))
			{
				handler_wrapper(footage_playlist_handler, c, hm, server->footage_dir_, server->time_format_);
			}
			else
			{
				mg_http_serve_dir(c, hm, &server->http_server_options_);
			}
		}

		if (ev == MG_EV_CLOSE)
		{
		}
	}

	void run()
	{
		while (running_)
		{
			mg_mgr_poll(&mgr_, 1000);
		}
	}
};

std::unique_ptr<WebServer> WebServer::Create(SurvOptions const *options)
{
	return std::make_unique<MongooseServer>(options);
}
