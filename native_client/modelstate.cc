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
                 unsigned int beam_width)
{
  beam_width_ = beam_width;
  return DS_ERR_OK;
}

char*
ModelState::decode(const DecoderState& state)
{
  vector<Output> out = state.decode(1);
  return strdup(alphabet_.LabelsToString(out[0].tokens).c_str());
}

vector<Metadata*>
ModelState::decode_metadata(const DecoderState& state, 
                            size_t top_paths)
{
  vector<Output> out = state.decode(top_paths);

  vector<Metadata*> meta_out;

  size_t max_results = std::min(top_paths, out.size());

  for (int j = 0; j < max_results; ++j) {
    std::unique_ptr<Metadata> metadata(new Metadata());
    metadata->num_items = out[j].tokens.size();
    metadata->confidence = out[j].confidence;

    std::unique_ptr<MetadataItem[]> items(new MetadataItem[metadata->num_items]());

    // Loop through each character
    for (int i = 0; i < out[j].tokens.size(); ++i) {
      items[i].character = strdup(alphabet_.StringFromLabel(out[j].tokens[i]).c_str());
      items[i].timestep = out[j].timesteps[i];
      items[i].start_time = out[j].timesteps[i] * ((float)audio_win_step_ / sample_rate_);

      if (items[i].start_time < 0) {
        items[i].start_time = 0;
      }
    }

    metadata->items = items.release();
    meta_out.push_back(metadata.release());
  }

  return meta_out;
}
