#include "tflitemodelstate.h"
#include "tensorflow/lite/string_util.h"
#include "workspace_status.h"

#ifdef __ANDROID__
#include <android/log.h>
#define  LOG_TAG    "libmozilla_voice_stt"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define  LOGD(...)
#define  LOGE(...)
#endif // __ANDROID__

using namespace tflite;
using std::vector;

int
TFLiteModelState::get_tensor_by_name(const vector<int>& list,
                                     const char* name)
{
  int rv = -1;

  for (int i = 0; i < list.size(); ++i) {
    const string& node_name = interpreter_->tensor(list[i])->name;
    if (node_name.compare(string(name)) == 0) {
      rv = i;
    }
  }

  assert(rv >= 0);
  return rv;
}

int
TFLiteModelState::get_input_tensor_by_name(const char* name)
{
  int idx = get_tensor_by_name(interpreter_->inputs(), name);
  return interpreter_->inputs()[idx];
}

int
TFLiteModelState::get_output_tensor_by_name(const char* name)
{
  int idx = get_tensor_by_name(interpreter_->outputs(), name);
  return interpreter_->outputs()[idx];
}

void
push_back_if_not_present(std::deque<int>& list, int value)
{
  if (std::find(list.begin(), list.end(), value) == list.end()) {
    list.push_back(value);
  }
}

// Backwards BFS on the node DAG. At each iteration we get the next tensor id
// from the frontier list, then for each node which has that tensor id as an
// output, add it to the parent list, and add its input tensors to the frontier
// list. Because we start from the final tensor and work backwards to the inputs,
// the parents list is constructed in reverse, adding elements to its front.
vector<int>
TFLiteModelState::find_parent_node_ids(int tensor_id)
{
  std::deque<int> parents;
  std::deque<int> frontier;
  frontier.push_back(tensor_id);
  while (!frontier.empty()) {
    int next_tensor_id = frontier.front();
    frontier.pop_front();
    // Find all nodes that have next_tensor_id as an output
    for (int node_id = 0; node_id < interpreter_->nodes_size(); ++node_id) {
      TfLiteNode node = interpreter_->node_and_registration(node_id)->first;
      // Search node outputs for the tensor we're looking for
      for (int i = 0; i < node.outputs->size; ++i) {
        if (node.outputs->data[i] == next_tensor_id) {
          // This node is part of the parent tree, add it to the parent list and
          // add its input tensors to the frontier list
          parents.push_front(node_id);
          for (int j = 0; j < node.inputs->size; ++j) {
            push_back_if_not_present(frontier, node.inputs->data[j]);
          }
        }
      }
    }
  }

  return vector<int>(parents.begin(), parents.end());
}

TFLiteModelState::TFLiteModelState()
  : ModelState()
  , interpreter_(nullptr)
  , fbmodel_(nullptr)
{
}

TFLiteModelState::~TFLiteModelState()
{
}

std::map<std::string, tflite::Interpreter::TfLiteDelegatePtr>
getTfliteDelegates()
{
  std::map<std::string, tflite::Interpreter::TfLiteDelegatePtr> delegates;

  const char* env_delegate_c = std::getenv("DS_TFLITE_DELEGATE");
  std::string env_delegate = (env_delegate_c != nullptr) ? env_delegate_c : "";

#ifdef __ANDROID__
  if (env_delegate == std::string("gpu")) {
    LOGD("Trying to get GPU delegate ...");
    // Try to get GPU delegate
    {
      tflite::Interpreter::TfLiteDelegatePtr delegate = evaluation::CreateGPUDelegate();
      if (!delegate) {
        LOGD("GPU delegation not supported");
      } else {
        LOGD("GPU delegation supported");
        delegates.emplace("GPU", std::move(delegate));
      }
    }
  }

  if (env_delegate == std::string("nnapi")) {
    LOGD("Trying to get NNAPI delegate ...");
    // Try to get Android NNAPI delegate
    {
      tflite::Interpreter::TfLiteDelegatePtr delegate = evaluation::CreateNNAPIDelegate();
      if (!delegate) {
        LOGD("NNAPI delegation not supported");
      } else {
        LOGD("NNAPI delegation supported");
        delegates.emplace("NNAPI", std::move(delegate));
      }
    }
  }

  if (env_delegate == std::string("hexagon")) {
    LOGD("Trying to get Hexagon delegate ...");
    // Try to get Android Hexagon delegate
    {
      const std::string libhexagon_path("/data/local/tmp");
      tflite::Interpreter::TfLiteDelegatePtr delegate = evaluation::CreateHexagonDelegate(libhexagon_path, /* profiler */ false);
      if (!delegate) {
        LOGD("Hexagon delegation not supported");
      } else {
        LOGD("Hexagon delegation supported");
        delegates.emplace("Hexagon", std::move(delegate));
      }
    }
  }
#endif // __ANDROID__

  return delegates;
}

int
TFLiteModelState::init(const char* model_path)
{
  int err = ModelState::init(model_path);
  if (err != STT_ERR_OK) {
    return err;
  }

  fbmodel_ = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!fbmodel_) {
    std::cerr << "Error at reading model file " << model_path << std::endl;
    return STT_ERR_FAIL_INIT_MMAP;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*fbmodel_, resolver)(&interpreter_);
  if (!interpreter_) {
    std::cerr << "Error at InterpreterBuilder for model file " << model_path << std::endl;
    return STT_ERR_FAIL_INTERPRETER;
  }

  LOGD("Trying to detect delegates ...");
  std::map<std::string, tflite::Interpreter::TfLiteDelegatePtr> delegates = getTfliteDelegates();
  LOGD("Finished enumerating delegates ...");

  interpreter_->AllocateTensors();
  interpreter_->SetNumThreads(4);

  LOGD("Trying to use delegates ...");
  for (const auto& delegate : delegates) {
    LOGD("Trying to apply delegate %s", delegate.first.c_str());
    if (interpreter_->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk) {
      LOGD("FAILED to apply delegate %s to the graph", delegate.first.c_str());
    }
  }

  // Query all the index once
  input_node_idx_       = get_input_tensor_by_name("input_node");
  previous_state_c_idx_ = get_input_tensor_by_name("previous_state_c");
  previous_state_h_idx_ = get_input_tensor_by_name("previous_state_h");
  input_samples_idx_    = get_input_tensor_by_name("input_samples");
  logits_idx_           = get_output_tensor_by_name("logits");
  new_state_c_idx_      = get_output_tensor_by_name("new_state_c");
  new_state_h_idx_      = get_output_tensor_by_name("new_state_h");
  mfccs_idx_            = get_output_tensor_by_name("mfccs");

  int metadata_version_idx  = get_output_tensor_by_name("metadata_version");
  int metadata_sample_rate_idx      = get_output_tensor_by_name("metadata_sample_rate");
  int metadata_feature_win_len_idx  = get_output_tensor_by_name("metadata_feature_win_len");
  int metadata_feature_win_step_idx = get_output_tensor_by_name("metadata_feature_win_step");
  int metadata_beam_width_idx = get_output_tensor_by_name("metadata_beam_width");
  int metadata_alphabet_idx = get_output_tensor_by_name("metadata_alphabet");

  std::vector<int> metadata_exec_plan;
  metadata_exec_plan.push_back(find_parent_node_ids(metadata_version_idx)[0]);
  metadata_exec_plan.push_back(find_parent_node_ids(metadata_sample_rate_idx)[0]);
  metadata_exec_plan.push_back(find_parent_node_ids(metadata_feature_win_len_idx)[0]);
  metadata_exec_plan.push_back(find_parent_node_ids(metadata_feature_win_step_idx)[0]);
  metadata_exec_plan.push_back(find_parent_node_ids(metadata_beam_width_idx)[0]);
  metadata_exec_plan.push_back(find_parent_node_ids(metadata_alphabet_idx)[0]);

  for (int i = 0; i < metadata_exec_plan.size(); ++i) {
    assert(metadata_exec_plan[i] > -1);
  }

  // When we call Interpreter::Invoke, the whole graph is executed by default,
  // which means every time compute_mfcc is called the entire acoustic model is
  // also executed. To workaround that problem, we walk up the dependency DAG
  // from the mfccs output tensor to find all the relevant nodes required for
  // feature computation, building an execution plan that runs just those nodes.
  auto mfcc_plan = find_parent_node_ids(mfccs_idx_);
  auto orig_plan = interpreter_->execution_plan();

  // Remove MFCC and Metatda nodes from original plan (all nodes) to create the acoustic model plan
  auto erase_begin = std::remove_if(orig_plan.begin(), orig_plan.end(), [&mfcc_plan, &metadata_exec_plan](int elem) {
    return (std::find(mfcc_plan.begin(), mfcc_plan.end(), elem) != mfcc_plan.end()
         || std::find(metadata_exec_plan.begin(), metadata_exec_plan.end(), elem) != metadata_exec_plan.end());
  });
  orig_plan.erase(erase_begin, orig_plan.end());

  acoustic_exec_plan_ = std::move(orig_plan);
  mfcc_exec_plan_ = std::move(mfcc_plan);

  interpreter_->SetExecutionPlan(metadata_exec_plan);
  TfLiteStatus status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Error running session: " << status << "\n";
    return STT_ERR_FAIL_INTERPRETER;
  }

  int* const graph_version = interpreter_->typed_tensor<int>(metadata_version_idx);
  if (graph_version == nullptr) {
    std::cerr << "Unable to read model file version." << std::endl;
    return STT_ERR_MODEL_INCOMPATIBLE;
  }

  if (*graph_version < ds_graph_version()) {
    std::cerr << "Specified model file version (" << *graph_version << ") is "
              << "incompatible with minimum version supported by this client ("
              << ds_graph_version() << "). See "
              << "https://github.com/mozilla/DeepSpeech/blob/"
              << ds_git_version() << "/doc/USING.rst#model-compatibility "
              << "for more information" << std::endl;
    return STT_ERR_MODEL_INCOMPATIBLE;
  }

  int* const model_sample_rate = interpreter_->typed_tensor<int>(metadata_sample_rate_idx);
  if (model_sample_rate == nullptr) {
    std::cerr << "Unable to read model sample rate." << std::endl;
    return STT_ERR_MODEL_INCOMPATIBLE;
  }

  sample_rate_ = *model_sample_rate;

  int* const win_len_ms  = interpreter_->typed_tensor<int>(metadata_feature_win_len_idx);
  int* const win_step_ms = interpreter_->typed_tensor<int>(metadata_feature_win_step_idx);
  if (win_len_ms == nullptr || win_step_ms == nullptr) {
    std::cerr << "Unable to read model feature window informations." << std::endl;
    return STT_ERR_MODEL_INCOMPATIBLE;
  }

  audio_win_len_  = sample_rate_ * (*win_len_ms / 1000.0);
  audio_win_step_ = sample_rate_ * (*win_step_ms / 1000.0);

  int* const beam_width = interpreter_->typed_tensor<int>(metadata_beam_width_idx);
  beam_width_ = (unsigned int)(*beam_width);

  tflite::StringRef serialized_alphabet = tflite::GetString(interpreter_->tensor(metadata_alphabet_idx), 0);
  err = alphabet_.Deserialize(serialized_alphabet.str, serialized_alphabet.len);
  if (err != 0) {
    return STT_ERR_INVALID_ALPHABET;
  }

  assert(sample_rate_ > 0);
  assert(audio_win_len_ > 0);
  assert(audio_win_step_ > 0);
  assert(beam_width_ > 0);
  assert(alphabet_.GetSize() > 0);

  TfLiteIntArray* dims_input_node = interpreter_->tensor(input_node_idx_)->dims;

  n_steps_ = dims_input_node->data[1];
  n_context_ = (dims_input_node->data[2] - 1) / 2;
  n_features_ = dims_input_node->data[3];
  mfcc_feats_per_timestep_ = dims_input_node->data[2] * dims_input_node->data[3];

  TfLiteIntArray* dims_logits = interpreter_->tensor(logits_idx_)->dims;
  const int final_dim_size = dims_logits->data[1] - 1;
  if (final_dim_size != alphabet_.GetSize()) {
    std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
              << "has size " << alphabet_.GetSize()
              << ", but model has " << final_dim_size
              << " classes in its output. Make sure you're passing an alphabet "
              << "file with the same size as the one used for training."
              << std::endl;
    return STT_ERR_INVALID_ALPHABET;
  }

  TfLiteIntArray* dims_c = interpreter_->tensor(previous_state_c_idx_)->dims;
  TfLiteIntArray* dims_h = interpreter_->tensor(previous_state_h_idx_)->dims;
  assert(dims_c->data[1] == dims_h->data[1]);
  assert(state_size_ > 0);
  state_size_ = dims_c->data[1];

  return STT_ERR_OK;
}

// Copy contents of vec into the tensor with index tensor_idx.
// If vec.size() < num_elements, set the remainder of the tensor values to zero.
void
TFLiteModelState::copy_vector_to_tensor(const vector<float>& vec,
                                        int tensor_idx,
                                        int num_elements)
{
  float* tensor = interpreter_->typed_tensor<float>(tensor_idx);
  int i;
  for (i = 0; i < vec.size(); ++i) {
    tensor[i] = vec[i];
  }
  for (; i < num_elements; ++i) {
    tensor[i] = 0.f;
  }
}

// Copy num_elements elements from the tensor with index tensor_idx into vec
void
TFLiteModelState::copy_tensor_to_vector(int tensor_idx,
                                        int num_elements,
                                        vector<float>& vec)
{
  float* tensor = interpreter_->typed_tensor<float>(tensor_idx);
  for (int i = 0; i < num_elements; ++i) {
    vec.push_back(tensor[i]);
  }
}

void
TFLiteModelState::infer(const vector<float>& mfcc,
                        unsigned int n_frames,
                        const vector<float>& previous_state_c,
                        const vector<float>& previous_state_h,
                        vector<float>& logits_output,
                        vector<float>& state_c_output,
                        vector<float>& state_h_output)
{
  const size_t num_classes = alphabet_.GetSize() + 1; // +1 for blank

  // Feeding input_node
  copy_vector_to_tensor(mfcc, input_node_idx_, n_frames*mfcc_feats_per_timestep_);

  // Feeding previous_state_c, previous_state_h
  assert(previous_state_c.size() == state_size_);
  copy_vector_to_tensor(previous_state_c, previous_state_c_idx_, state_size_);
  assert(previous_state_h.size() == state_size_);
  copy_vector_to_tensor(previous_state_h, previous_state_h_idx_, state_size_);

  interpreter_->SetExecutionPlan(acoustic_exec_plan_);
  TfLiteStatus status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  copy_tensor_to_vector(logits_idx_, n_frames * BATCH_SIZE * num_classes, logits_output);

  state_c_output.clear();
  state_c_output.reserve(state_size_);
  copy_tensor_to_vector(new_state_c_idx_, state_size_, state_c_output);

  state_h_output.clear();
  state_h_output.reserve(state_size_);
  copy_tensor_to_vector(new_state_h_idx_, state_size_, state_h_output);
}

void
TFLiteModelState::compute_mfcc(const vector<float>& samples,
                               vector<float>& mfcc_output)
{
  // Feeding input_node
  copy_vector_to_tensor(samples, input_samples_idx_, samples.size());

  TfLiteStatus status = interpreter_->SetExecutionPlan(mfcc_exec_plan_);
  if (status != kTfLiteOk) {
    std::cerr << "Error setting execution plan: " << status << "\n";
    return;
  }

  status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  // The feature computation graph is hardcoded to one audio length for now
  int n_windows = 1;
  TfLiteIntArray* out_dims = interpreter_->tensor(mfccs_idx_)->dims;
  int num_elements = 1;
  for (int i = 0; i < out_dims->size; ++i) {
    num_elements *= out_dims->data[i];
  }
  assert(num_elements / n_features_ == n_windows);

  copy_tensor_to_vector(mfccs_idx_, n_windows * n_features_, mfcc_output);
}
