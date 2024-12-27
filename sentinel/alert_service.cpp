#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdio.h>

#include "alert_service.hpp"

#define INFO_COLOR "#3399ff"
#define WARNING_COLOR "#f9b115"
#define DANGER_COLOR "#e55353"

struct upload_status
{
	std::string *payload;
	size_t bytes_read;
};

static std::string base64_encode(const uint8_t *data, size_t length)
{
	static constexpr const char *BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	std::string result;
	int val = 0;
	int valb = -6;

	for (size_t i = 0; i < length; i++)
	{
		val = (val << 8) + data[i];
		valb += 8;

		while (valb >= 0)
		{
			result.push_back(BASE64_ALPHABET[(val >> valb) & 0x3F]);
			valb -= 6;
		}
	}

	if (valb > -6)
	{
		result.push_back(BASE64_ALPHABET[((val << 8) >> (valb + 8)) & 0x3F]);
	}

	while (result.size() % 4)
	{
		result.push_back('=');
	}

	return result;
}

static std::string render_html_template(const std::string &template_content, alert &al)
{
	std::string template_copy = template_content;

	std::map<std::string, std::string> replacements;
	replacements["bg_color"] = INFO_COLOR;
	replacements["alert_level"] = "Info";
	replacements["alert_type"] = "Nothing";
	replacements["risk_level"] = "no-risk";

	if (al.type == alert_type::Motion)
	{
		replacements["bg_color"] = WARNING_COLOR;
		replacements["alert_level"] = "Warning";
		replacements["alert_type"] = "Motion";
		replacements["risk_level"] = "low-risk";
	}

	if (al.type == alert_type::FaceRecognition)
	{
		replacements["bg_color"] = DANGER_COLOR;
		replacements["alert_level"] = "Alert";
		replacements["alert_type"] = "Face";
		replacements["risk_level"] = "high-risk";
	}

	replacements["time_str"] = al.time_str;
	replacements["base64_snapshot"] = base64_encode(al.jpeg_buffer_ptr.get(), al.jpeg_buffer_size);

	for (const auto &pair : replacements)
	{
		std::string placeholder = "{{" + pair.first + "}}";
		size_t pos = 0;

		while ((pos = template_copy.find(placeholder, pos)) != std::string::npos)
		{
			template_copy.replace(pos, placeholder.length(), pair.second);
			pos += pair.second.length();
		}
	}

	return template_copy;
}

static std::string name_address(const std::string &name, const std::string &address)
{
	std::string result = name;
	if (!name.empty())
		result += " ";
	result += address;
	return result;
}

static std::string build_payload(const std::string &html_content, const std::string &sender_name,
								 const std::string &sender_address, const std::string &receiver_name,
								 const std::string &receiver_address)
{
	std::stringstream payload;

	payload << "To: " << name_address(receiver_name, receiver_address) << "\r\n"
			<< "From: " << name_address(sender_name, sender_address) << "\r\n"
			<< "Subject: Security Alert Notification\r\n"
			<< "Content-Type: text/html; charset=UTF-8\r\n"
			<< "\r\n";

	payload << html_content;

	return payload.str();
}

static size_t mail_source(char *ptr, size_t size, size_t nmemb, void *userdata)
{
	upload_status *ctx = (upload_status *)userdata;

	const std::string *payload = ctx->payload;

	size_t content_size = payload->size() - ctx->bytes_read;
	size_t to_copy = size * nmemb < content_size ? size * nmemb : content_size;

	std::memcpy(ptr, payload->c_str() + ctx->bytes_read, to_copy);
	ctx->bytes_read += to_copy;

	return to_copy;
}

void EmailService::LoadConfig(boost::property_tree::ptree const &params)
{
	config_.smtp_server_url = params.get<std::string>("smtp_server_url");
	config_.sender_address = params.get<std::string>("sender_address");
	config_.password = params.get<std::string>("password");
	config_.sender_name = params.get<std::string>("sender_name");
	config_.receiver_address = params.get<std::string>("receiver_address");
	config_.receiver_name = params.get<std::string>("receiver_name");
	config_.template_file = params.get<std::string>("template_file");

	loadTemplate();
}

void EmailService::loadTemplate()
{
	std::ifstream file(config_.template_file);
	if (!file.is_open())
		throw std::runtime_error("Failed to load template file");

	std::stringstream buffer;
	buffer << file.rdbuf();
	html_template_str_ = buffer.str();
}

void EmailService::SendAlert(alert al)
{
	CURL *curl;
	CURLcode res;
	struct curl_slist *recipients = NULL;
	struct upload_status upload_ctx;

	upload_ctx.payload = NULL;
	upload_ctx.bytes_read = 0;

	std::string html_content = render_html_template(html_template_str_, al);
	std::string payload = build_payload(html_content, config_.sender_name, config_.sender_address,
										config_.receiver_name, config_.receiver_address);
	upload_ctx.payload = &payload;

	curl = curl_easy_init();
	if (curl)
	{
		curl_easy_setopt(curl, CURLOPT_URL, config_.smtp_server_url.c_str());

		curl_easy_setopt(curl, CURLOPT_USERNAME, config_.sender_address.c_str());
		curl_easy_setopt(curl, CURLOPT_PASSWORD, config_.password.c_str());
		curl_easy_setopt(curl, CURLOPT_MAIL_FROM, config_.sender_address.c_str());
		curl_easy_setopt(curl, CURLOPT_USE_SSL, (long)CURLUSESSL_ALL);

		recipients = curl_slist_append(recipients, config_.receiver_address.c_str());
		curl_easy_setopt(curl, CURLOPT_MAIL_RCPT, recipients);

		curl_easy_setopt(curl, CURLOPT_READFUNCTION, mail_source);
		curl_easy_setopt(curl, CURLOPT_READDATA, &upload_ctx);
		curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);

		// Set 1L if need to print more debug information
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

		res = curl_easy_perform(curl);
		if (res != CURLE_OK)
			fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));

		curl_slist_free_all(recipients);
		curl_easy_cleanup(curl);
	}
}