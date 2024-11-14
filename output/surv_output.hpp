#pragma once

extern "C" 
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

#include "output.hpp"
#include "core/surv_options.hpp"

class SurvOutput : public Output {
public:
    static Output *Create(SurvOptions const *options);

    SurvOutput(SurvOptions const *options);
    ~SurvOutput();

protected:
    void outputBuffer(void *mem, size_t size, int64_t timestamp_us, uint32_t flags) override;
    void timestampReady(int64_t timestamp) override;

private:
    void startNewPlaylist(int64_t timestamp_us);
    void finalizePlaylist();
    void startNewSegment();
    void finalizeSegment(int64_t timestamp_us);
    void writeSegmentData(void *mem, size_t size, int64_t timestamp_us, uint32_t flags);
    
    int64_t getSysTimestamp(int64_t timestamp);

    std::string footage_directory_;
    std::string playlist_directory_;
    std::ofstream playlist_file_;
    std::ofstream segment_file_;
    unsigned int segment_index_;
    int64_t segment_start_time_;
    const unsigned int segment_duration_;
    int64_t playlist_start_time_;
    const int64_t playlist_interval_duration_;
    int64_t sys_start_timestamp_;
    AVFormatContext *format_ctx_;
    AVStream *video_stream_;
};
