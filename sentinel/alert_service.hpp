#include <curl/curl.h>
#include <memory>
#include <string>

#include <boost/property_tree/ptree.hpp>

enum class alert_type
{
	None,
	Motion,
	FaceRecognition
};

struct alert
{
	alert_type type;
	std::string time_str;
	std::shared_ptr<uint8_t[]> jpeg_buffer_ptr;
	size_t jpeg_buffer_size;
};

#if LIBCURL_PRESENT
struct EmailConfig
{
	std::string smtp_server_url;
	std::string sender_address;
	std::string password;
	std::string sender_name;
	std::string receiver_address;
	std::string receiver_name;
	std::string template_file;
};

class EmailService
{
public:
	EmailService() {};
	~EmailService() = default;

	void LoadConfig(boost::property_tree::ptree const &params);
	void SendAlert(alert al);

private:
	EmailConfig config_;
	std::string html_template_str_;

	void loadTemplate();
};
#endif