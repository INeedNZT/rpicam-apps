#pragma once

#include <string>

class WebServer {
public:
    WebServer(const std::string &host, int port, int max_connections)
        : host_(host), port_(port), max_connections_(max_connections), running_(false) {}
    
    virtual ~WebServer() = default;

    virtual void Start() = 0;
    virtual void Stop() = 0;

    static std::unique_ptr<WebServer> Create(const std::string &host, int port, int max_connections);

protected:
    std::string host_;
    int port_;
    int max_connections_;
    bool running_;
};
