#include <filesystem>

#include "surv_output.hpp"

static std::string getDateString(int64_t timestamp_us)
{
	time_t time_in_seconds = static_cast<time_t>(timestamp_us / 1000000);
	std::tm tm = *std::localtime(&time_in_seconds);

	char buffer[16];
	strftime(buffer, sizeof(buffer), "%Y%m%d", &tm);
	return std::string(buffer);
}

static std::string getTimeString(int64_t timestamp)
{
	time_t time_in_seconds = static_cast<time_t>(timestamp / 1000000);

	std::tm tm = *std::localtime(&time_in_seconds);

	char buffer[20];
	strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M", &tm);
	return std::string(buffer);
}

static bool isNewDay(int64_t timestamp_us)
{
	time_t time_in_seconds = static_cast<time_t>(timestamp_us / 1000000);

	// Compare Dates only by day
	std::tm *tm = std::localtime(&time_in_seconds);
	tm->tm_hour = 0;
	tm->tm_min = 0;
	tm->tm_sec = 0;
	time_t day = std::mktime(tm);

	static time_t previous_day = day;

	if (day != previous_day)
	{
		previous_day = day;

		return true;
	}

	return false;
}

SurvOutput::SurvOutput(SurvOptions const *options)
	: Output(options), hls_directory_(options->hls_directory), segment_index_(0), segment_start_time_(0),
	  segment_duration_(options->segment_duration), playlist_start_time_(0), playlist_interval_duration_(60 * 60),
	  sys_start_timestamp_(0)
{
	std::filesystem::create_directories(hls_directory_);

	avformat_alloc_output_context2(&format_ctx_, nullptr, "mpegts", nullptr);
	if (!format_ctx_)
		throw std::runtime_error("Failed to create FFmpeg format context");

	video_stream_ = avformat_new_stream(format_ctx_, nullptr);
	if (!video_stream_)
		throw std::runtime_error("Failed to create video stream");

	video_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
	video_stream_->codecpar->codec_id = AV_CODEC_ID_H264;
	video_stream_->codecpar->width = options->width;
	video_stream_->codecpar->height = options->height;
	video_stream_->time_base = { 1, 90000 };
}

SurvOutput::~SurvOutput()
{
	finalizePlaylist();

	if (format_ctx_)
		avformat_free_context(format_ctx_);
}

void SurvOutput::outputBuffer(void *mem, size_t size, int64_t timestamp_us, uint32_t flags)
{
	int64_t sys_timestamp = getSysTimestamp(timestamp_us);

	if (isNewDay(sys_timestamp) || timestamp_us == 0 ||
		timestamp_us - playlist_start_time_ >= playlist_interval_duration_ * 1000000)
	{
		if (segment_index_ > 0)
            finalizeSegment(timestamp_us);

		if (timestamp_us != 0)
			finalizePlaylist();

		startNewPlaylist(timestamp_us);
		startNewSegment();
		segment_start_time_ = timestamp_us;
	}

	if ((flags & FLAG_KEYFRAME) && (timestamp_us - segment_start_time_ >= segment_duration_ * 1000000))
	{
		if (segment_index_ > 0)
			finalizeSegment(timestamp_us);

		startNewSegment();
		segment_start_time_ = timestamp_us;
	}

	writeSegmentData(mem, size, timestamp_us, flags);
}

void SurvOutput::timestampReady(int64_t timestamp)
{
	//TODO
}

void SurvOutput::startNewPlaylist(int64_t timestamp_us)
{
	playlist_start_time_ = timestamp_us;
	int64_t sys_timestamp = getSysTimestamp(playlist_start_time_);
	std::string date_directory = hls_directory_ + "/" + getDateString(sys_timestamp);
	std::filesystem::create_directories(date_directory);

	std::string time_str = getTimeString(sys_timestamp);
	playlist_directory_ = date_directory + "/" + time_str;
	std::filesystem::create_directories(playlist_directory_);

	std::string playlist_filename = playlist_directory_ + "/" + time_str + ".m3u8";
	playlist_file_.open(playlist_filename, std::ios::out | std::ios::trunc);
	if (!playlist_file_)
		throw std::runtime_error("Failed to create playlist file: " + playlist_filename);

	playlist_file_ << "#EXTM3U\n#EXT-X-VERSION:3\n";
	playlist_file_ << "#EXT-X-TARGETDURATION:" << segment_duration_ + 3 << "\n";

	segment_index_ = 0;
}

void SurvOutput::finalizePlaylist()
{
	if (playlist_file_.is_open())
	{
		playlist_file_ << "#EXT-X-ENDLIST\n";
		playlist_file_.flush();
		playlist_file_.close();
	}
}

void SurvOutput::startNewSegment()
{
	std::string segment_filename = playlist_directory_ + "/" + "segment_" + std::to_string(segment_index_) + ".ts";

	if (!(format_ctx_->oformat->flags & AVFMT_NOFILE))
	{
		if (avio_open(&format_ctx_->pb, segment_filename.c_str(), AVIO_FLAG_WRITE) < 0)
			throw std::runtime_error("Failed to open TS segment file");
	}

	if (avformat_write_header(format_ctx_, nullptr) < 0)
		throw std::runtime_error("Failed to write TS header");

	segment_index_++;
}

void SurvOutput::finalizeSegment(int64_t timestamp_us)
{
	if (format_ctx_->pb)
	{
		av_write_trailer(format_ctx_);
		avio_close(format_ctx_->pb);
	}

	double actual_duration = (timestamp_us - segment_start_time_) / 1000000.0;
	playlist_file_ << "#EXTINF:" << actual_duration << ",\n";
	playlist_file_ << "segment_" << (segment_index_ - 1) << ".ts\n";
	playlist_file_.flush();
}

void SurvOutput::writeSegmentData(void *mem, size_t size, int64_t timestamp_us, uint32_t flags)
{
	AVPacket *pkt = av_packet_alloc();
	if (!pkt)
		throw std::runtime_error("Failed to allocate AVPacket");

	pkt->data = reinterpret_cast<uint8_t *>(mem);
	pkt->size = size;
	pkt->pts = pkt->dts = av_rescale_q(timestamp_us, { 1, 1000000 }, video_stream_->time_base);
	pkt->stream_index = video_stream_->index;
	pkt->flags |= (flags & FLAG_KEYFRAME) ? AV_PKT_FLAG_KEY : 0;

	if (av_interleaved_write_frame(format_ctx_, pkt) < 0)
	{
		av_packet_free(&pkt);
		throw std::runtime_error("Failed to write frame to TS file");
	}

	av_packet_free(&pkt);
}

int64_t SurvOutput::getSysTimestamp(int64_t timestamp_us)
{
	if (sys_start_timestamp_ == 0)
	{
		sys_start_timestamp_ =
			std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
				.count();
	}

	return sys_start_timestamp_ + timestamp_us;
}

Output *SurvOutput::Create(SurvOptions const *options)
{
	if (options->record)
	{
		return new SurvOutput(options);
	}
	return new Output(static_cast<const VideoOptions *>(options));
}
