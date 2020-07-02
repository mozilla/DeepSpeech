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
  return strdup(alphabet_.Decode(out[0].tokens).c_str());
}

Metadata*
ModelState::decode_metadata(const DecoderState& state, 
                            size_t num_results)
{
  vector<Output> out = state.decode(num_results);
  unsigned int num_returned = out.size();

  CandidateTranscript* transcripts = (CandidateTranscript*)malloc(sizeof(CandidateTranscript)*num_returned);

  for (int i = 0; i < num_returned; ++i) {
    TokenMetadata* tokens = (TokenMetadata*)malloc(sizeof(TokenMetadata)*out[i].tokens.size());

    for (int j = 0; j < out[i].tokens.size(); ++j) {
      TokenMetadata token {
        strdup(alphabet_.DecodeSingle(out[i].tokens[j]).c_str()),   // text
        static_cast<unsigned int>(out[i].timesteps[j]),                // timestep
        out[i].timesteps[j] * ((float)audio_win_step_ / sample_rate_), // start_time
      };
      memcpy(&tokens[j], &token, sizeof(TokenMetadata));
    }

    CandidateTranscript transcript {
      tokens,                                          // tokens
      static_cast<unsigned int>(out[i].tokens.size()), // num_tokens
      out[i].confidence,                               // confidence
    };
    memcpy(&transcripts[i], &transcript, sizeof(CandidateTranscript));
  }

  Metadata* ret = (Metadata*)malloc(sizeof(Metadata));
  Metadata metadata {
    transcripts,  // transcripts
    num_returned, // num_transcripts
  };
  memcpy(ret, &metadata, sizeof(Metadata));
  return ret;
}
