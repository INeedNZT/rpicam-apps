#include <filesystem>
#include <jpeglib.h>

#include "surv_output.hpp"

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

static std::string getDatePath(const std::string &footage_directory, time_t timestamp_sec)
{
	std::string date_str = SurvOptions::ToTimeStr(timestamp_sec, "%Y-%m-%d");

	for (const auto &entry : std::filesystem::directory_iterator(footage_directory))
	{
		time_t directory_date = static_cast<time_t>(std::stoll(entry.path().filename()));
		if (entry.is_directory() && SurvOptions::ToTimeStr(directory_date, "%Y-%m-%d") == date_str)
			return footage_directory + "/" + std::to_string(directory_date);
	}

	return footage_directory + "/" + std::to_string(timestamp_sec);
}

SurvOutput::SurvOutput(SurvOptions const *options)
	: Output(options), footage_directory_(options->footage_directory), segment_index_(0), segment_start_time_(0),
	  segment_duration_(options->segment_duration), playlist_start_time_(0), playlist_interval_duration_(60 * 60)
{
	std::filesystem::create_directories(footage_directory_);

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
	int64_t sys_timestamp = SurvOptions::GetSysTimestamp(timestamp_us);

	if (flags & FLAG_KEYFRAME)
	{
		if (isNewDay(sys_timestamp) || timestamp_us == 0 ||
			timestamp_us - playlist_start_time_ >= playlist_interval_duration_ * 1000000)
		{
			if (segment_index_ > 0)
				finalizeSegment(timestamp_us);

			if (timestamp_us != 0)
				finalizePlaylist();

			startNewPlaylist(mem, size, timestamp_us);
			startNewSegment();
			segment_start_time_ = timestamp_us;
		}

		if (timestamp_us - segment_start_time_ >= segment_duration_ * 1000000)
		{
			if (segment_index_ > 0)
				finalizeSegment(timestamp_us);

			startNewSegment();
			segment_start_time_ = timestamp_us;
		}
	}

	writeSegmentData(mem, size, timestamp_us, flags);
}

void SurvOutput::timestampReady(int64_t timestamp)
{
	//TODO
}

void SurvOutput::startNewPlaylist(void *mem, size_t size, int64_t timestamp_us)
{
	playlist_start_time_ = timestamp_us;
	int64_t sys_timestamp = SurvOptions::GetSysTimestamp(playlist_start_time_);
	time_t sys_time_sec = static_cast<time_t>(sys_timestamp / 1000000);
	std::string date_directory = getDatePath(footage_directory_, sys_time_sec);
	std::filesystem::create_directories(date_directory);
	std::string date_thumb_path = date_directory + "/" + THUMB_NAME;
	if (!std::filesystem::exists(date_thumb_path))
	{
		// Save first frame as thumbnail
		saveThumbnail(mem, size, timestamp_us, date_thumb_path);
	}

	playlist_directory_ = date_directory + "/" + std::to_string(sys_time_sec);
	std::filesystem::create_directories(playlist_directory_);
	std::string playlist_thumb_path = playlist_directory_ + "/" + THUMB_NAME;
	if (!std::filesystem::exists(playlist_thumb_path))
	{
		saveThumbnail(mem, size, timestamp_us, playlist_thumb_path);
	}

	std::string playlist_filename = playlist_directory_ + "/" + PLAYLIST_NAME;
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

void SurvOutput::saveThumbnail(void *mem, size_t size, int64_t timestamp_us, const std::string &save_path)
{
	AVCodecContext *codec_ctx = nullptr;
	AVPacket *pkt = av_packet_alloc();
	pkt->data = reinterpret_cast<uint8_t *>(mem);
	pkt->size = static_cast<int>(size);

	const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	codec_ctx = avcodec_alloc_context3(codec);
	avcodec_open2(codec_ctx, codec, nullptr);

	AVFrame *frame = av_frame_alloc();
	avcodec_send_packet(codec_ctx, pkt);
	avcodec_receive_frame(codec_ctx, frame);

	AVFrame *rgb_frame = av_frame_alloc();
	int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame->width, frame->height, 1);
	uint8_t *buffer = (uint8_t *)av_malloc(num_bytes);
	av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, buffer, AV_PIX_FMT_RGB24, frame->width, frame->height,
						 1);

	struct SwsContext *sws_ctx = sws_getContext(frame->width, frame->height, codec_ctx->pix_fmt, frame->width,
												frame->height, AV_PIX_FMT_RGB24, 0, nullptr, nullptr, nullptr);
	sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, rgb_frame->data, rgb_frame->linesize);

	FILE *jpeg_file = fopen(save_path.c_str(), "wb");
	if (!jpeg_file)
		throw std::runtime_error("Error opening JPEG file for writing");

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, jpeg_file);

	cinfo.image_width = frame->width;
	cinfo.image_height = frame->height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 90, TRUE);

	jpeg_start_compress(&cinfo, TRUE);

	JSAMPROW row_pointer[1];

	for (int y = 0; y < frame->height; y++)
	{
		row_pointer[0] = &rgb_frame->data[0][y * rgb_frame->linesize[0]];
		jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}

	jpeg_finish_compress(&cinfo);
	fclose(jpeg_file);

	jpeg_destroy_compress(&cinfo);

	av_packet_free(&pkt);
	av_frame_free(&rgb_frame);
	av_frame_free(&frame);
	avcodec_free_context(&codec_ctx);
	sws_freeContext(sws_ctx);
	av_free(buffer);
}

Output *SurvOutput::Create(SurvOptions const *options)
{
	if (options->record)
	{
		return new SurvOutput(options);
	}
	return new Output(static_cast<const VideoOptions *>(options));
}
