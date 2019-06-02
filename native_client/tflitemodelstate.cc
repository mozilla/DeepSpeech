#include "tflitemodelstate.h"

using namespace tflite;
using std::vector;

int
tflite_get_tensor_by_name(const Interpreter* interpreter,
                          const vector<int>& list, 
                          const char* name)
{
  int rv = -1;

  for (int i = 0; i < list.size(); ++i) {
    const string& node_name = interpreter->tensor(list[i])->name;
    if (node_name.compare(string(name)) == 0) {
      rv = i;
    }
  }

  assert(rv >= 0);
  return rv;
}

int
tflite_get_input_tensor_by_name(const Interpreter* interpreter, const char* name)
{
  int idx = tflite_get_tensor_by_name(interpreter, interpreter->inputs(), name);
  return interpreter->inputs()[idx];
}

int
tflite_get_output_tensor_by_name(const Interpreter* interpreter, const char* name)
{
  int idx = tflite_get_tensor_by_name(interpreter, interpreter->outputs(), name);
  return interpreter->outputs()[idx];
}

void push_back_if_not_present(std::deque<int>& list, int value)
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
std::vector<int>
tflite_find_parent_node_ids(Interpreter* interpreter, int tensor_id)
{
  std::deque<int> parents;
  std::deque<int> frontier;
  frontier.push_back(tensor_id);
  while (!frontier.empty()) {
    int next_tensor_id = frontier.front();
    frontier.pop_front();
    // Find all nodes that have next_tensor_id as an output
    for (int node_id = 0; node_id < interpreter->nodes_size(); ++node_id) {
      TfLiteNode node = interpreter->node_and_registration(node_id)->first;
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

  return std::vector<int>(parents.begin(), parents.end());
}

TFLiteModelState::TFLiteModelState()
  : ModelState()
  , interpreter_(nullptr)
  , fbmodel_(nullptr)
  , previous_state_size_(0)
  , previous_state_c_(nullptr)
  , previous_state_h_(nullptr)
{
}

int
TFLiteModelState::init(const char* model_path,
                       unsigned int n_features,
                       unsigned int n_context,
                       const char* alphabet_path,
                       unsigned int beam_width)
{
  int err = ModelState::init(model_path, n_features, n_context, alphabet_path, beam_width);
  if (err != DS_ERR_OK) {
    return err;
  }

  fbmodel_ = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!fbmodel_) {
    std::cerr << "Error at reading model file " << model_path << std::endl;
    return DS_ERR_FAIL_INIT_MMAP;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*fbmodel_, resolver)(&interpreter_);
  if (!interpreter_) {
    std::cerr << "Error at InterpreterBuilder for model file " << model_path << std::endl;
    return DS_ERR_FAIL_INTERPRETER;
  }

  interpreter_->AllocateTensors();
  interpreter_->SetNumThreads(4);

  // Query all the index once
  input_node_idx_       = tflite_get_input_tensor_by_name(interpreter_.get(), "input_node");
  previous_state_c_idx_ = tflite_get_input_tensor_by_name(interpreter_.get(), "previous_state_c");
  previous_state_h_idx_ = tflite_get_input_tensor_by_name(interpreter_.get(), "previous_state_h");
  input_samples_idx_    = tflite_get_input_tensor_by_name(interpreter_.get(), "input_samples");
  logits_idx_           = tflite_get_output_tensor_by_name(interpreter_.get(), "logits");
  new_state_c_idx_      = tflite_get_output_tensor_by_name(interpreter_.get(), "new_state_c");
  new_state_h_idx_      = tflite_get_output_tensor_by_name(interpreter_.get(), "new_state_h");
  mfccs_idx_            = tflite_get_output_tensor_by_name(interpreter_.get(), "mfccs");

  // When we call Interpreter::Invoke, the whole graph is executed by default,
  // which means every time compute_mfcc is called the entire acoustic model is
  // also executed. To workaround that problem, we walk up the dependency DAG
  // from the mfccs output tensor to find all the relevant nodes required for
  // feature computation, building an execution plan that runs just those nodes.
  auto mfcc_plan = tflite_find_parent_node_ids(interpreter_.get(), mfccs_idx_);
  auto orig_plan = interpreter_->execution_plan();

  // Remove MFCC nodes from original plan (all nodes) to create the acoustic model plan
  auto erase_begin = std::remove_if(orig_plan.begin(), orig_plan.end(), [&mfcc_plan](int elem) {
    return std::find(mfcc_plan.begin(), mfcc_plan.end(), elem) != mfcc_plan.end();
  });
  orig_plan.erase(erase_begin, orig_plan.end());

  acoustic_exec_plan_ = std::move(orig_plan);
  mfcc_exec_plan_ = std::move(mfcc_plan);

  TfLiteIntArray* dims_input_node = interpreter_->tensor(input_node_idx_)->dims;

  n_steps_ = dims_input_node->data[1];
  n_context_ = (dims_input_node->data[2] - 1) / 2;
  n_features_ = dims_input_node->data[3];
  mfcc_feats_per_timestep_ = dims_input_node->data[2] * dims_input_node->data[3];

  TfLiteIntArray* dims_logits = interpreter_->tensor(logits_idx_)->dims;
  const int final_dim_size = dims_logits->data[1] - 1;
  if (final_dim_size != alphabet_->GetSize()) {
    std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
              << "has size " << alphabet_->GetSize()
              << ", but model has " << final_dim_size
              << " classes in its output. Make sure you're passing an alphabet "
              << "file with the same size as the one used for training."
              << std::endl;
    return DS_ERR_INVALID_ALPHABET;
  }

  TfLiteIntArray* dims_c = interpreter_->tensor(previous_state_c_idx_)->dims;
  TfLiteIntArray* dims_h = interpreter_->tensor(previous_state_h_idx_)->dims;
  assert(dims_c->data[1] == dims_h->data[1]);

  previous_state_size_ = dims_c->data[1];
  previous_state_c_.reset(new float[previous_state_size_]());
  previous_state_h_.reset(new float[previous_state_size_]());

  // Set initial values for previous_state_c and previous_state_h
  memset(previous_state_c_.get(), 0, sizeof(float) * previous_state_size_);
  memset(previous_state_h_.get(), 0, sizeof(float) * previous_state_size_);

  return DS_ERR_OK;
}

int
TFLiteModelState::initialize_state()
{
  /* Ensure previous_state_{c,h} are not holding previous stream value */
  memset(previous_state_c_.get(), 0, sizeof(float) * previous_state_size_);
  memset(previous_state_h_.get(), 0, sizeof(float) * previous_state_size_);

  return DS_ERR_OK;
}

void
TFLiteModelState::infer(const float* aMfcc, unsigned int n_frames, vector<float>& logits_output)
{
  const size_t num_classes = alphabet_->GetSize() + 1; // +1 for blank

  // Feeding input_node
  float* input_node = interpreter_->typed_tensor<float>(input_node_idx_);
  {
    int i;
    for (i = 0; i < n_frames*mfcc_feats_per_timestep_; ++i) {
      input_node[i] = aMfcc[i];
    }
    for (; i < n_steps_*mfcc_feats_per_timestep_; ++i) {
      input_node[i] = 0;
    }
  }

  assert(previous_state_size_ > 0);

  // Feeding previous_state_c, previous_state_h
  memcpy(interpreter_->typed_tensor<float>(previous_state_c_idx_), previous_state_c_.get(), sizeof(float) * previous_state_size_);
  memcpy(interpreter_->typed_tensor<float>(previous_state_h_idx_), previous_state_h_.get(), sizeof(float) * previous_state_size_);

  interpreter_->SetExecutionPlan(acoustic_exec_plan_);
  TfLiteStatus status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "Error running session: " << status << "\n";
    return;
  }

  float* outputs = interpreter_->typed_tensor<float>(logits_idx_);

  // The CTCDecoder works with log-probs.
  for (int t = 0; t < n_frames * BATCH_SIZE * num_classes; ++t) {
    logits_output.push_back(outputs[t]);
  }

  memcpy(previous_state_c_.get(), interpreter_->typed_tensor<float>(new_state_c_idx_), sizeof(float) * previous_state_size_);
  memcpy(previous_state_h_.get(), interpreter_->typed_tensor<float>(new_state_h_idx_), sizeof(float) * previous_state_size_);
}

void
TFLiteModelState::compute_mfcc(const vector<float>& samples, vector<float>& mfcc_output)
{
  // Feeding input_node
  float* input_samples = interpreter_->typed_tensor<float>(input_samples_idx_);
  for (int i = 0; i < samples.size(); ++i) {
    input_samples[i] = samples[i];
  }

  interpreter_->SetExecutionPlan(mfcc_exec_plan_);
  TfLiteStatus status = interpreter_->Invoke();
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

  float* outputs = interpreter_->typed_tensor<float>(mfccs_idx_);
  for (int i = 0; i < n_windows * n_features_; ++i) {
    mfcc_output.push_back(outputs[i]);
  }
}
