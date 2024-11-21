#pragma once

#include <string>

#include "core/surv_options.hpp"

class WebServer
{
public:
	WebServer(SurvOptions const *options)
		: host_(options->web_host), port_(options->web_port), max_connections_(options->max_connections),
		  running_(false)
	{
	}

	virtual ~WebServer() = default;

	virtual void Start() = 0;
	virtual void Stop() = 0;
	virtual void RecvFrameData(void *mem, size_t size) = 0;

	static std::unique_ptr<WebServer> Create(SurvOptions const *options);

protected:
	std::string host_;
	int port_;
	int max_connections_;
	bool running_;
};
