#include <vector>

#include "ctcdecode/ctc_beam_search_decoder.h"

#include "modelstate.h"

using std::vector;

ModelState::ModelState()
  : alphabet_(nullptr)
  , scorer_(nullptr)
  , beam_width_(-1)
  , n_steps_(-1)
  , n_context_(-1)
  , n_features_(-1)
  , mfcc_feats_per_timestep_(-1)
  , sample_rate_(DEFAULT_SAMPLE_RATE)
  , audio_win_len_(DEFAULT_WINDOW_LENGTH)
  , audio_win_step_(DEFAULT_WINDOW_STEP)
  , state_size_(-1)
{
}

ModelState::~ModelState()
{
  delete scorer_;
  delete alphabet_;
}

int
ModelState::init(const char* model_path,
                 unsigned int n_features,
                 unsigned int n_context,
                 const char* alphabet_path,
                 unsigned int beam_width)
{
  n_features_ = n_features;
  n_context_ = n_context;
  alphabet_ = new Alphabet(alphabet_path);
  beam_width_ = beam_width;
  return DS_ERR_OK;
}

vector<Output>
ModelState::decode_raw(DecoderState* state)
{
  vector<Output> out = decoder_decode(state, *alphabet_, beam_width_, scorer_);
  return out;
}

char*
ModelState::decode(DecoderState* state)
{
  vector<Output> out = decode_raw(state);
  return strdup(alphabet_->LabelsToString(out[0].tokens).c_str());
}

Metadata*
ModelState::decode_metadata(DecoderState* state)
{
  vector<Output> out = decode_raw(state);

  std::unique_ptr<Metadata> metadata(new Metadata());
  metadata->num_items = out[0].tokens.size();
  metadata->probability = out[0].probability;

  std::unique_ptr<MetadataItem[]> items(new MetadataItem[metadata->num_items]());

  // Loop through each character
  for (int i = 0; i < out[0].tokens.size(); ++i) {
    items[i].character = strdup(alphabet_->StringFromLabel(out[0].tokens[i]).c_str());
    items[i].timestep = out[0].timesteps[i];
    items[i].start_time = out[0].timesteps[i] * ((float)audio_win_step_ / sample_rate_);

    if (items[i].start_time < 0) {
      items[i].start_time = 0;
    }
  }

  metadata->items = items.release();
  return metadata.release();
}




// .. we have to merge this back in, somehow..

// size_t arg_max(const float *logits, size_t size) {
//   float max = 0.0;
//   size_t arg_max = 0;
//   for (size_t i = 0; i < size; i++)  {
//     if (max < logits[i]) {
//       max = logits[i];
//       arg_max = i;
//     }
//   }
//   return arg_max;  
// }

// float entropy (const float *logits, size_t size) {
//   float sum = 0.0f;
//   for (size_t i = 0; i < size; i++) {
//     if (logits[i]>0.0f)
//       sum += logits[i] * log2(logits[i]);   
//   }
//   return -sum;
// }

// // what we did before
// Metadata*
// ModelState::decode_metadata(const vector<float>& logits)
// {
//   const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

//   vector<Output> out = decode_raw(logits);
//   Output best = out[0]; 

//   std::unique_ptr<Metadata> metadata(new Metadata());
//   metadata->num_items = best.tokens.size();
//   metadata->probability = best.probability;

//   // Loop through each character, assign metadata for each
//   std::unique_ptr<MetadataItem[]> items(new MetadataItem[metadata->num_items]());
//   for (int i = 0; i < best.tokens.size(); ++i) {
    
//     // best guess from acoustic model, for the timestep corresponding to chosen character
//     int argmax = arg_max(&logits[best.timesteps[i] * num_classes], num_classes);  
   
//     items[i].character = strdup(alphabet->StringFromLabel(best.tokens[i]).c_str());
//     items[i].timestep = best.timesteps[i];
//     items[i].start_time = best.timesteps[i] * ((float)audio_win_step / sample_rate);
//     items[i].acoustic_char = strdup(alphabet->StringFromLabel(argmax).c_str());
//     items[i].probability = logits[best.timesteps[i] * num_classes + best.tokens[i]];
//     items[i].entropy = entropy(&logits[best.timesteps[i] * num_classes], num_classes);
   
//     if (items[i].start_time < 0) {
//       items[i].start_time = 0;
//     }
//   }

//   metadata->items = items.release();
//   return metadata.release();
// }



// // partially merged
// Metadata*
// ModelState::decode_metadata(DecoderState* state)
// {
//   const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

//   vector<Output> out = decode_raw(state);
//   Output best = out[0]; 

//   std::unique_ptr<Metadata> metadata(new Metadata());
//   metadata->num_items = best.tokens.size();
//   metadata->probability = best.probability;

//   std::unique_ptr<MetadataItem[]> items(new MetadataItem[metadata->num_items]());

//   // Loop through each character
//   for (int i = 0; i < best.tokens.size(); ++i) {

//     // best guess from acoustic model, for the timestep corresponding to chosen character
//     int argmax = arg_max(&logits[best.timesteps[i] * num_classes], num_classes);  
   
//     items[i].character = strdup(alphabet_->StringFromLabel(best.tokens[i]).c_str());
//     items[i].timestep = best.timesteps[i];
//     items[i].start_time = best.timesteps[i] * ((float)audio_win_step_ / sample_rate_);

//     items[i].acoustic_char = strdup(alphabet_->StringFromLabel(argmax).c_str());
//     items[i].probability = logits[best.timesteps[i] * num_classes + best.tokens[i]];
//     items[i].entropy = entropy(&logits[best.timesteps[i] * num_classes], num_classes);

//     if (items[i].start_time < 0) {
//       items[i].start_time = 0;
//     }
//   }

//   metadata->items = items.release();
//   return metadata.release();
// }
