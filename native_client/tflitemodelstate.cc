#include "tflitemodelstate.h"

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
  input_node_idx_       = get_input_tensor_by_name("input_node");
  previous_state_c_idx_ = get_input_tensor_by_name("previous_state_c");
  previous_state_h_idx_ = get_input_tensor_by_name("previous_state_h");
  input_samples_idx_    = get_input_tensor_by_name("input_samples");
  logits_idx_           = get_output_tensor_by_name("logits");
  new_state_c_idx_      = get_output_tensor_by_name("new_state_c");
  new_state_h_idx_      = get_output_tensor_by_name("new_state_h");
  mfccs_idx_            = get_output_tensor_by_name("mfccs");

  // When we call Interpreter::Invoke, the whole graph is executed by default,
  // which means every time compute_mfcc is called the entire acoustic model is
  // also executed. To workaround that problem, we walk up the dependency DAG
  // from the mfccs output tensor to find all the relevant nodes required for
  // feature computation, building an execution plan that runs just those nodes.
  auto mfcc_plan = find_parent_node_ids(mfccs_idx_);
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
  assert(state_size_ > 0);
  state_size_ = dims_c->data[1];

  return DS_ERR_OK;
}

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
  const size_t num_classes = alphabet_->GetSize() + 1; // +1 for blank

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
