#include <chrono>

#include <dlib/clustering.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/string.h>

#include <filesystem>
#include <libcamera/stream.h>

#include "core/rpicam_app.hpp"
#include "post_processing_stages/post_processing_stage.hpp"

namespace fs = std::filesystem;
using namespace dlib;
using namespace std;
using Stream = libcamera::Stream;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN,
		  typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN,
		  typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<
	128, avg_pool_everything<alevel0<alevel1<alevel2<
			 alevel3<alevel4<max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

class FaceRecognitionStage : public PostProcessingStage
{
public:
	FaceRecognitionStage(RPiCamApp *app) : PostProcessingStage(app) {}

	char const *Name() const override;

	void Read(boost::property_tree::ptree const &params) override;

	void Configure() override;

	bool Process(CompletedRequestPtr &completed_request) override;

	void Stop() override;

private:
	Stream *stream_;
	StreamInfo low_res_info_;
	Stream *full_stream_;
	StreamInfo full_stream_info_;
	unique_ptr<std::future<void>> future_ptr_;
	std::mutex future_ptr_mutex_;
	int refresh_rate_;

	matrix<rgb_pixel> image_;
	frontal_face_detector detector;
	shape_predictor sp;
	anet_type net;

	std::vector<matrix<float, 0, 1>> recorded_face_descriptors;
	void loadAllImages(const fs::path &image_dir);
	void getFaceDescriptors(matrix<rgb_pixel> &img, std::vector<matrix<float, 0, 1>> &face_descriptors);
};

#define NAME "face_recognition_dlib"

char const *FaceRecognitionStage::Name() const
{
	return NAME;
}

void FaceRecognitionStage::Read(boost::property_tree::ptree const &params)
{
	refresh_rate_ = 5;
}

void FaceRecognitionStage::Configure()
{
	stream_ = app_->LoresStream();
	if (!stream_)
		throw std::runtime_error("FaceRecognitionStage: no low resolution stream");
	low_res_info_ = app_->GetStreamInfo(stream_);
	image_.set_size(low_res_info_.height, low_res_info_.width);

	detector = get_frontal_face_detector();
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	loadAllImages("./faces");
}

bool FaceRecognitionStage::Process(CompletedRequestPtr &completed_request)
{
	if (!stream_)
		return false;

	{
		std::unique_lock<std::mutex> lck(future_ptr_mutex_);
		if (completed_request->sequence % refresh_rate_ == 0 &&
			(!future_ptr_ || future_ptr_->wait_for(std::chrono::seconds(0)) == std::future_status::ready))
		{
			BufferReadSync r(app_, completed_request->buffers[stream_]);
			libcamera::Span<uint8_t> buffer = r.Get()[0];
			StreamInfo stream_info;
			stream_info.width = low_res_info_.width;
			stream_info.height = low_res_info_.height;
			stream_info.stride = stream_info.width * 3;
			std::vector<uint8_t> rgb_image = Yuv420ToRgb(buffer.data(), low_res_info_, stream_info);

			// TODO:could this be more efficient?
			const uint8_t *data = rgb_image.data();
			for (unsigned int y = 0; y < stream_info.height; ++y)
			{
				for (unsigned int x = 0; x < stream_info.width; ++x)
				{
					image_(y, x) = rgb_pixel(data[0], data[1], data[2]);
					data += 3;
				}
			}

			try
			{
				save_jpeg(image_, "output_image.jpg");
				cout << "Image saved to output_image.jpg" << endl;
			}
			catch (const dlib::error &e)
			{
				cout << "Error saving image: " << e.what() << endl;
			}

			std::vector<matrix<float, 0, 1>> face_descriptors;

			getFaceDescriptors(image_, face_descriptors);

			if (face_descriptors.empty())
			{
				cout << "No face found" << endl;
				return false;
			}

			for (size_t i = 0; i < recorded_face_descriptors.size(); ++i)
			{
				for (size_t j = i; j < face_descriptors.size(); ++j)
				{
					if (length(recorded_face_descriptors[i] - face_descriptors[j]) < 0.6)
					{
						cout << "Found recorded face!" << endl;
						return false;
					}
				}
			}
			cout << "Stranger Alert!" << endl;
		}
	}
	return false;
}

void FaceRecognitionStage::Stop()
{
	// if (future_ptr_)
	// 	future_ptr_->wait();
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new FaceRecognitionStage(app);
}

static RegisterStage reg(NAME, &Create);

bool is_image_file(const fs::path &path)
{
	static const std::vector<string> valid_extensions = { ".jpg", ".jpeg", ".png", ".bmp" };
	string ext = path.extension().string();
	for (const auto &valid_ext : valid_extensions)
	{
		if (ext == valid_ext)
		{
			return true;
		}
	}
	return false;
}

void FaceRecognitionStage::loadAllImages(const fs::path &image_dir)
{
	if (!fs::exists(image_dir) || !fs::is_directory(image_dir))
	{
		cerr << "Invalid directory path!" << endl;
		return;
	}

	for (const auto &entry : fs::directory_iterator(image_dir))
	{
		if (entry.is_regular_file())
		{
			fs::path file_path = entry.path();

			if (is_image_file(file_path))
			{
				cout << "Processing image: " << file_path << endl;

				matrix<rgb_pixel> img;
				load_image(img, file_path.string());

				std::vector<matrix<float, 0, 1>> face_descriptors;

				getFaceDescriptors(img, face_descriptors);

				if (face_descriptors.empty())
					throw std::runtime_error("FaceRecognitionStage: no faces found in image!");

				recorded_face_descriptors.insert(recorded_face_descriptors.end(), face_descriptors.begin(),
												 face_descriptors.end());
			}
		}
	}
}

void FaceRecognitionStage::getFaceDescriptors(matrix<rgb_pixel> &img,
											  std::vector<matrix<float, 0, 1>> &face_descriptors)
{
	std::vector<rectangle> faces = detector(img);
	if (faces.size() == 0)
	{
		face_descriptors.clear();
		return;
	}

	std::vector<matrix<rgb_pixel>> face_chips;
	for (auto &face : faces)
	{
		auto shape = sp(img, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
		face_chips.push_back(move(face_chip));
	}

	face_descriptors = net(face_chips);
}