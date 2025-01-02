#pragma once

#include <string>

#include "core/rpicam_encoder.hpp"
#include "core/surv_options.hpp"

class WebServer
{
public:
	WebServer(RPiCamEncoder<SurvOptions> *app)
		: host_(app->GetOptions()->web_host), port_(app->GetOptions()->web_port),
		  max_connections_(app->GetOptions()->max_connections), running_(false), app_(app)
	{
	}

	virtual ~WebServer() = default;

	virtual void Start() = 0;
	virtual void Stop() = 0;
	virtual void RecvFrameData(void *mem, size_t size) = 0;

	static std::unique_ptr<WebServer> Create(RPiCamEncoder<SurvOptions> *app);

protected:
	std::string host_;
	int port_;
	int max_connections_;
	bool running_;

	RPiCamEncoder<SurvOptions> *app_;
};
