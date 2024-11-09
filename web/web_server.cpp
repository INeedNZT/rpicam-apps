#include <thread>

#include "mongoose.h"
#include "web_server.hpp"

class MongooseServer : public WebServer
{
public:
	MongooseServer(const std::string &host, int port, int max_connections)
		: WebServer(host, port, max_connections), event_loop_thread(nullptr)
	{
		mg_mgr_init(&mgr);
	}

	~MongooseServer() { Stop(); }

	void Start() override
	{
		if (running_)
		{
			return;
		}

		std::string url = host_ + ":" + std::to_string(port_);
		mg_http_listen(&mgr, url.c_str(), eventHandler, this);

		running_ = true;

		event_loop_thread = new std::thread(&MongooseServer::run, this);
	}

	void Stop() override
	{
		if (!running_)
		{
			return;
		}

		running_ = false;

		if (event_loop_thread && event_loop_thread->joinable())
		{
			event_loop_thread->join();
		}

		delete event_loop_thread;
		event_loop_thread = nullptr;

		mg_mgr_free(&mgr);
	}

private:
	struct mg_mgr mgr;
	std::thread *event_loop_thread;

	static inline int numconns(struct mg_mgr *mgr)
	{
		int count = 0;
		for (struct mg_connection *c = mgr->conns; c != nullptr; c = c->next)
		{
			count++;
		}
		return count;
	}

	static void eventHandler(struct mg_connection *c, int ev, void *ev_data)
	{
		MongooseServer *server = static_cast<MongooseServer *>(c->fn_data);

		if (ev == MG_EV_ACCEPT)
		{
			if (numconns(&server->mgr) >= server->max_connections_)
			{
				MG_ERROR(("Too many connections"));
				c->is_closing = 1;
			}
		}

		if (ev == MG_EV_HTTP_MSG)
		{
			static struct mg_http_serve_opts opts;
			static bool initialized = false;
			if (!initialized)
			{
				opts.root_dir = "/var/www/static";
				initialized = true;
			}

			struct mg_http_message *hm = (struct mg_http_message *)ev_data;

			if (mg_match(hm->uri, mg_str("/api/hello"), NULL))
			{
				// Return JSON response
				mg_http_reply(c, 200, "Content-Type: application/json\r\n", "{%m:%d}\n", MG_ESC("status"), 1);
			}
			else
			{
				mg_http_serve_dir(c, hm, &opts);
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
			mg_mgr_poll(&mgr, 1000);
		}
	}
};

std::unique_ptr<WebServer> WebServer::Create(const std::string &host, int port, int max_connections)
{
	return std::make_unique<MongooseServer>(host, port, max_connections);
}
