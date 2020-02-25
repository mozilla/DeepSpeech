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
ModelState::init(const char* model_path)
{
  return DS_ERR_OK;
}

char*
ModelState::decode(const DecoderState& state) const
{
  vector<Output> out = state.decode();
  return strdup(alphabet_.LabelsToString(out[0].tokens).c_str());
}

Metadata*
ModelState::decode_metadata(const DecoderState& state, 
                            size_t num_results)
{
  vector<Output> out = state.decode(num_results);
  size_t num_returned = out.size();

  std::unique_ptr<Metadata> metadata(new Metadata);
  metadata->num_transcripts = num_returned;

  std::unique_ptr<CandidateTranscript[]> transcripts(new CandidateTranscript[num_returned]);

  for (int i = 0; i < num_returned; ++i) {
    transcripts[i].num_tokens = out[i].tokens.size();
    transcripts[i].confidence = out[i].confidence;

    std::unique_ptr<TokenMetadata[]> tokens(new TokenMetadata[transcripts[i].num_tokens]);

    // Loop through each token
    for (int j = 0; j < out[i].tokens.size(); ++j) {
      tokens[j].text = strdup(alphabet_.StringFromLabel(out[i].tokens[j]).c_str());
      tokens[j].timestep = out[i].timesteps[j];
      tokens[j].start_time = out[i].timesteps[j] * ((float)audio_win_step_ / sample_rate_);

      if (tokens[j].start_time < 0) {
        tokens[j].start_time = 0;
      }
    }

    transcripts[i].tokens = tokens.release();
  }

  metadata->transcripts = transcripts.release();
  return metadata.release();
}
