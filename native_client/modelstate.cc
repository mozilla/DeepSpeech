#include <vector>

#include "ctcdecode/ctc_beam_search_decoder.h"

#include "modelstate.h"

using std::vector;

ModelState::ModelState()
  : beam_width_(-1)
  , n_steps_(-1)
  , n_context_(-1)
  , n_features_(-1)
  , mfcc_feats_per_timestep_(-1)
  , sample_rate_(-1)
  , audio_win_len_(-1)
  , audio_win_step_(-1)
  , state_size_(-1)
{
}

ModelState::~ModelState()
{
}

int
ModelState::init(const char* model_path,
                 const char* alphabet_path,
                 unsigned int beam_width)
{
  if (alphabet_.init(alphabet_path)) {
    return DS_ERR_INVALID_ALPHABET;
  }
  beam_width_ = beam_width;
  return DS_ERR_OK;
}

char*
ModelState::decode(const DecoderState& state)
{
  vector<Output> out = state.decode();
  return strdup(alphabet_.LabelsToString(out[0].tokens).c_str());
}

Metadata*
ModelState::decode_metadata(const DecoderState& state)
{
  vector<Output> out = state.decode();

  std::unique_ptr<Metadata> metadata(new Metadata());
  metadata->num_items = out[0].tokens.size();
  metadata->confidence = out[0].confidence;

  std::unique_ptr<MetadataItem[]> items(new MetadataItem[metadata->num_items]());

  // Loop through each character
  for (int i = 0; i < out[0].tokens.size(); ++i) {
    items[i].character = strdup(alphabet_.StringFromLabel(out[0].tokens[i]).c_str());
    items[i].timestep = out[0].timesteps[i];
    items[i].start_time = out[0].timesteps[i] * ((float)audio_win_step_ / sample_rate_);

    if (items[i].start_time < 0) {
      items[i].start_time = 0;
    }
  }

  metadata->items = items.release();
  return metadata.release();
}
