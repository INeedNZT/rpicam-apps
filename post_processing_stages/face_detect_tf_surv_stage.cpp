// This file is almost identical to face_detect_tf_stage.cpp, with the addition of motion detection logic.
// Face detection will only be performed if motion is detected, which helps reduce computational resource consumption.

#include "tf_stage.hpp"

constexpr int INPUT_WIDTH = 320;
constexpr int INPUT_HEIGHT = 240;

constexpr float CLIP_MIN = 0.0f;
constexpr float CLIP_MAX = 1.0f;

struct FaceDetectTfSurvConfig : public TfConfig
{
	float conf_threshold;
	float nms_iou_threshold;
	float center_variance;
	float size_variance;
};

#define NAME "face_detect_tf_surv"

class FaceDetectTfSurvStage : public TfStage
{
public:
	FaceDetectTfSurvStage(RPiCamApp *app) : TfStage(app, INPUT_WIDTH, INPUT_HEIGHT)
	{
		config_ = std::make_unique<FaceDetectTfSurvConfig>();
	}
	char const *Name() const override { return NAME; }

	bool Process(CompletedRequestPtr &completed_request) override;

protected:
	FaceDetectTfSurvConfig *config() const { return static_cast<FaceDetectTfSurvConfig *>(config_.get()); }

	void readExtras(boost::property_tree::ptree const &params) override;

	void checkConfiguration() override;

	void interpretOutputs() override;

	void applyResults(CompletedRequestPtr &completed_request) override;

private:
	std::vector<float> anchors_xy_;
	std::vector<float> anchors_wh_;
	std::vector<std::vector<float>> detected_boxes_;
	std::vector<float> detected_scores_;

	void generateAnchors();
	void decodeRegression(const float *reg, std::vector<std::vector<float>> &decoded_boxes);
	void applyNMS(const std::vector<std::vector<float>> &boxes, const std::vector<float> &scores,
				  std::vector<int> &selected_indices);
};

static float computeIOU(const std::vector<float> &box1, const std::vector<float> &box2)
{
	float x1 = std::max(box1[0], box2[0]);
	float y1 = std::max(box1[1], box2[1]);
	float x2 = std::min(box1[2], box2[2]);
	float y2 = std::min(box1[3], box2[3]);

	float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
	float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
	float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
	float union_area = area1 + area2 - intersection;

	return union_area > 0 ? intersection / union_area : 0.0f;
}

bool FaceDetectTfSurvStage::Process(CompletedRequestPtr &completed_request)
{
	bool motion_detected;
	completed_request->post_process_metadata.Get("motion_detect.result", motion_detected);

	if (!motion_detected)
		return false;
	else
		return TfStage::Process(completed_request);
}

void FaceDetectTfSurvStage::readExtras(boost::property_tree::ptree const &params)
{
	config()->conf_threshold = params.get<float>("confidence_threshold", 0.6f);
	config()->nms_iou_threshold = params.get<float>("nms_iou_threshold", 0.3f);
	config()->center_variance = params.get<float>("center_variance", 0.1f);
	config()->size_variance = params.get<float>("size_variance", 0.2f);

	if (config()->verbose)
	{
		LOG(1, "Confidence Threshold: " << config()->conf_threshold
										<< ", NMS IoU Threshold: " << config()->nms_iou_threshold
										<< ", Center Variance: " << config()->center_variance
										<< ", Size Variance: " << config()->size_variance);
	}

	generateAnchors();
}

void FaceDetectTfSurvStage::checkConfiguration()
{
	if (!lores_stream_)
		throw std::runtime_error("FaceDetectTfSurvStage: Low resolution stream is required");
}

void FaceDetectTfSurvStage::generateAnchors()
{
	std::array<std::pair<int, int>, 4> feature_maps = { { { 40, 30 }, { 20, 15 }, { 10, 8 }, { 5, 4 } } };
	std::vector<std::vector<float>> min_boxes = { { 10, 16, 24 }, { 32, 48 }, { 64, 96 }, { 128, 192, 256 } };
	anchors_xy_.clear();
	anchors_wh_.clear();

	for (size_t k = 0; k < feature_maps.size(); ++k)
	{
		int f_w = feature_maps[k].first;
		int f_h = feature_maps[k].second;

		for (int y = 0; y < f_h; ++y)
		{
			for (int x = 0; x < f_w; ++x)
			{
				float cx = (x + 0.5f) / f_w;
				float cy = (y + 0.5f) / f_h;

				for (float box : min_boxes[k])
				{
					float w = box / INPUT_WIDTH;
					float h = box / INPUT_HEIGHT;
					anchors_xy_.push_back(std::clamp(cx, CLIP_MIN, CLIP_MAX));
					anchors_xy_.push_back(std::clamp(cy, CLIP_MIN, CLIP_MAX));
					anchors_wh_.push_back(std::clamp(w, CLIP_MIN, CLIP_MAX));
					anchors_wh_.push_back(std::clamp(h, CLIP_MIN, CLIP_MAX));
				}
			}
		}
	}
}

void FaceDetectTfSurvStage::decodeRegression(const float *reg, std::vector<std::vector<float>> &decoded_boxes)
{
	size_t num_boxes = anchors_xy_.size() / 2;
	decoded_boxes.resize(num_boxes);

	for (size_t i = 0; i < num_boxes; ++i)
	{
		float cx = reg[i * 4] * config()->center_variance * anchors_wh_[i * 2] + anchors_xy_[i * 2];
		float cy = reg[i * 4 + 1] * config()->center_variance * anchors_wh_[i * 2 + 1] + anchors_xy_[i * 2 + 1];
		float w = std::exp(reg[i * 4 + 2] * config()->size_variance) * anchors_wh_[i * 2] / 2;
		float h = std::exp(reg[i * 4 + 3] * config()->size_variance) * anchors_wh_[i * 2 + 1] / 2;

		decoded_boxes[i] = { std::clamp(cx - w, CLIP_MIN, CLIP_MAX), std::clamp(cy - h, CLIP_MIN, CLIP_MAX),
							 std::clamp(cx + w, CLIP_MIN, CLIP_MAX), std::clamp(cy + h, CLIP_MIN, CLIP_MAX) };
	}
}

void FaceDetectTfSurvStage::applyNMS(const std::vector<std::vector<float>> &boxes, const std::vector<float> &scores,
									 std::vector<int> &selected_indices)
{
	std::vector<std::pair<float, int>> score_index;
	for (size_t i = 0; i < scores.size(); ++i)
		score_index.emplace_back(scores[i], i);

	std::sort(score_index.rbegin(), score_index.rend());

	std::vector<bool> suppressed(scores.size(), false);

	for (size_t i = 0; i < score_index.size(); ++i)
	{
		int idx = score_index[i].second;
		if (suppressed[idx])
			continue;

		selected_indices.push_back(idx);
		for (size_t j = i + 1; j < score_index.size(); ++j)
		{
			int next_idx = score_index[j].second;
			if (suppressed[next_idx])
				continue;

			float iou = computeIOU(boxes[idx], boxes[next_idx]);
			if (iou > config()->nms_iou_threshold)
				suppressed[next_idx] = true;
		}
	}
}

void FaceDetectTfSurvStage::interpretOutputs()
{
	float *boxes = interpreter_->tensor(interpreter_->outputs()[0])->data.f;
	float *scores = interpreter_->tensor(interpreter_->outputs()[1])->data.f;

	std::vector<std::vector<float>> decoded_boxes;
	decodeRegression(boxes, decoded_boxes);

	std::vector<float> valid_scores;
	std::vector<std::vector<float>> valid_boxes;
	for (size_t i = 0; i < decoded_boxes.size(); ++i)
	{
		float score = scores[i * 2 + 1];
		if (score > config()->conf_threshold)
		{
			valid_boxes.push_back(decoded_boxes[i]);
			valid_scores.push_back(score);
		}
	}

	std::vector<int> final_indices;
	applyNMS(valid_boxes, valid_scores, final_indices);

	detected_boxes_.clear();
	detected_scores_.clear();

	for (int idx : final_indices)
	{
		detected_boxes_.push_back(valid_boxes[idx]);
		detected_scores_.push_back(valid_scores[idx]);
	}

	if (config()->verbose && final_indices.size())
	{
		LOG(1, "Face detection results:");
		for (size_t i = 0; i < detected_boxes_.size(); ++i)
		{
			LOG(1, "Box " << i << ": [" << detected_boxes_[i][0] << ", " << detected_boxes_[i][1] << ", "
						  << detected_boxes_[i][2] << ", " << detected_boxes_[i][3]
						  << "], Score: " << detected_scores_[i]);
		}
	}
}

void FaceDetectTfSurvStage::applyResults(CompletedRequestPtr &completed_request)
{
	completed_request->post_process_metadata.Set("face_detect.boxes", detected_boxes_);
	completed_request->post_process_metadata.Set("face_detect.scores", detected_scores_);
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new FaceDetectTfSurvStage(app);
}

static RegisterStage reg(NAME, &Create);
