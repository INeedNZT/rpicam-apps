#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "local_handler.hpp"
#include "mongoose.h"
#include "web_server.hpp"

struct FrameBuffer
{
	FrameBuffer(void *m, size_t s) : size(s)
	{
		mem = malloc(size);
		memcpy(mem, m, size);
	}

	~FrameBuffer()
	{
		if (mem)
		{
			free(mem);
			size = 0;
		}
	}

	void *mem;
	size_t size;
};

using FrameBufferPtr = std::shared_ptr<FrameBuffer>;

struct MsgWrapper
{
	FrameBufferPtr ptr;
};

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
	MongooseServer(SurvOptions const *options)
		: WebServer(options), streaming_(false), max_queue_size_(60), video_width_(options->width),
		  video_height_(options->height)
	{
		event_dir_ = options->event_directory;
		footage_dir_ = options->footage_directory;
		page404_ = options->web_root_directory + "/404.html";
		root_dir_ = options->web_root_directory + "," + FOOTAGE_PREFIX + "=" + options->footage_directory + "," +
					EVENT_PREFIX + "=" + options->event_directory;
		http_server_options_ = {};
		http_server_options_.page404 = page404_.c_str();
		http_server_options_.root_dir = root_dir_.c_str();

		date_format_ = options->footage_date_format;
		time_format_ = options->playlist_time_format;

		Log *log = new Log { options->web_log_directory };
		mg_log_set_fn(log_fn, static_cast<void *>(log));

		// Print more debug information if needed
		// mg_log_set(MG_LL_VERBOSE);

		mg_mgr_init(&mgr_);
		mg_wakeup_init(&mgr_);
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
		event_loop_thread_ = std::thread(&MongooseServer::run, this);

		streaming_ = true;
		streaming_thread_ = std::thread(&MongooseServer::broadcast, this);
	}

	void Stop() override
	{
		if (!running_)
		{
			return;
		}

		running_ = false;

		event_loop_thread_.join();

		streaming_ = false;

		streaming_thread_.join();

		mg_mgr_free(&mgr_);
	}

	void RecvFrameData(void *mem, size_t size) override
	{
		FrameBufferPtr frame_ptr = std::make_shared<FrameBuffer>(mem, size);

		if (frame_queue_.size() >= max_queue_size_)
		{
			std::lock_guard<std::mutex> lock(frame_queue_mutex_);
			std::queue<FrameBufferPtr> empty_queue;
			frame_queue_.swap(empty_queue);
		}

		frame_queue_.push(frame_ptr);
		frame_queue_cv_.notify_one();
	}

private:
	std::string page404_;
	std::string root_dir_;
	std::string footage_dir_;
	std::string event_dir_;

	std::string date_format_;
	std::string time_format_;

	struct mg_mgr mgr_;
	std::thread event_loop_thread_;
	struct mg_http_serve_opts http_server_options_;

	bool streaming_;
	std::queue<FrameBufferPtr> frame_queue_;
	unsigned int max_queue_size_;
	std::mutex frame_queue_mutex_;
	std::condition_variable frame_queue_cv_;
	std::thread streaming_thread_;

	unsigned int video_width_;
	unsigned int video_height_;

	// Connection pool for broadcast clients
	std::set<mg_connection *> ws_connections_;
	std::mutex ws_connections_mutex_;
	std::mutex broadcast_mutex_;
	std::condition_variable broadcast_cv_;

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

	static void surv_footage_handler(struct mg_connection *c, struct mg_http_message *hm, std::string footage_dir,
									 std::string date_format)
	{
		std::vector<day_surv_footage> footage = getDaySurvFootage(footage_dir);
		mg_http_reply(c, 200, "Content-Type: application/json\r\n", toJSON(footage, date_format).c_str());
	}

	static void surv_events_handler(struct mg_connection *c, struct mg_http_message *hm, std::string event_dir,
									std::string date_format, std::string time_format)
	{
		long timestamp_long = mg_json_get_long(hm->body, "$.et", 0);
		if (timestamp_long < 0)
			throw std::invalid_argument("Invalid timestamp value");
		std::vector<event> events = getEventListByDate(event_dir, static_cast<time_t>(timestamp_long));
		mg_http_reply(c, 200, "Content-Type: application/json\r\n", toJSON(events, date_format, time_format).c_str());
	}

	static void footage_playlist_handler(struct mg_connection *c, struct mg_http_message *hm, std::string footage_dir,
										 std::string time_format)
	{
		long timestamp_long = mg_json_get_long(hm->body, "$.st", -1);
		if (timestamp_long == -1)
			throw std::invalid_argument("Invalid timestamp value");
		std::vector<hour_playlist> playlist = getHourPlaylistByDate(footage_dir, static_cast<time_t>(timestamp_long));
		mg_http_reply(c, 200, "Content-Type: application/json\r\n", toJSON(playlist, time_format).c_str());
	}

	static void event_logs_handler(struct mg_connection *c, struct mg_http_message *hm, std::string event_dir,
								   std::string time_format)
	{
		std::string event_id = std::string(mg_json_get_str(hm->body, "$.ei"));
		if (event_id.empty())
			throw std::invalid_argument("Invalid event id");
		std::vector<event_log> logs = getEventLogsById(event_dir, event_id);
		mg_http_reply(c, 200, "Content-Type: application/json\r\n", toJSON(logs, time_format).c_str());
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
			else if (mg_match(hm->uri, mg_str("/api/survevents"), NULL))
			{
				handler_wrapper(surv_events_handler, c, hm, server->event_dir_, server->date_format_,
								server->time_format_);
			}
			else if (mg_match(hm->uri, mg_str("/api/playlist"), NULL))
			{
				handler_wrapper(footage_playlist_handler, c, hm, server->footage_dir_, server->time_format_);
			}
			else if (mg_match(hm->uri, mg_str("/api/eventlogs"), NULL))
			{
				handler_wrapper(event_logs_handler, c, hm, server->event_dir_, server->time_format_);
			}
			else if (mg_match(hm->uri, mg_str("/live"), NULL))
			{
				mg_ws_upgrade(c, hm, NULL);
			}
			else
			{
				mg_http_serve_dir(c, hm, &server->http_server_options_);
			}
		}
		else if (ev == MG_EV_WS_MSG)
		{
			struct mg_ws_message *wm = (struct mg_ws_message *)ev_data;

			if (std::string(wm->data.buf) == "REQUESTSTREAM ")
			{
				// Add to WebSocket connection pool
				{
					std::lock_guard<std::mutex> lock(server->ws_connections_mutex_);
					server->ws_connections_.insert(c);
					server->broadcast_cv_.notify_one();
				}
				MG_INFO(("User added to broadcast list"));
			}
			else if (std::string(wm->data.buf) == "STOPSTREAM")
			{
				{
					std::lock_guard<std::mutex> lock(server->ws_connections_mutex_);
					server->ws_connections_.erase(c);
				}
				MG_INFO(("User removed from broadcast list"));
			}
		}
		else if (ev == MG_EV_WS_OPEN)
		{
			char json[50];
			std::sprintf(json, R"({"action": "init", "width": %d, "height": %d})", server->video_width_, server->video_height_);
			mg_ws_send(c, json, std::strlen(json), WEBSOCKET_OP_TEXT);
			MG_INFO(("WS connection opened"));
		}
		else if (ev == MG_EV_WAKEUP)
		{
			struct mg_str *data = (struct mg_str *)ev_data;
			MsgWrapper *msg_wrapper = *(MsgWrapper **)data->buf;
			FrameBufferPtr frame_ptr = msg_wrapper->ptr;
			mg_ws_send(c, frame_ptr.get()->mem, frame_ptr.get()->size, WEBSOCKET_OP_BINARY);
			delete msg_wrapper;
			msg_wrapper = nullptr;
		}

		if (ev == MG_EV_CLOSE)
		{
			if (c->is_websocket)
			{
				std::lock_guard<std::mutex> lock(server->ws_connections_mutex_);
				server->ws_connections_.erase(c);
			}
		}
	}

	void run()
	{
		while (running_)
		{
			mg_mgr_poll(&mgr_, 1000);
		}
	}

	void broadcast()
	{
		while (streaming_)
		{
			FrameBufferPtr frame_ptr = nullptr;

			std::unique_lock<std::mutex> lock(broadcast_mutex_);
			broadcast_cv_.wait(lock, [this] { return !ws_connections_.empty(); });

			{
				std::unique_lock<std::mutex> frame_lock(frame_queue_mutex_);
				frame_queue_cv_.wait(lock, [this] { return !frame_queue_.empty(); });

				frame_ptr = frame_queue_.front();
				frame_queue_.pop();
			}

			{
				std::lock_guard<std::mutex> lock(ws_connections_mutex_);
				for (auto *conn : ws_connections_)
				{
					if (conn->is_closing || !conn->is_websocket)
					{
						MG_INFO(("WS is closing, skipping broadcast"));
						continue;
					}

					if (frame_ptr)
					{
						auto *msg_wrapper = new MsgWrapper { frame_ptr };
						mg_wakeup(&mgr_, conn->id, &msg_wrapper, sizeof(*msg_wrapper));
					}
				}
			}
		}
	}
};

std::unique_ptr<WebServer> WebServer::Create(SurvOptions const *options)
{
	return std::make_unique<MongooseServer>(options);
}
