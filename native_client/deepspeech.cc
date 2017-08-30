#include "deepspeech.h"
#include "deepspeech_utils.h"
#include "alphabet.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace tensorflow;

namespace DeepSpeech {

class Private {
  public:
    Session* session;
    GraphDef graph_def;
    int ncep;
    int ncontext;
    Alphabet* alphabet;
};

Model::Model(const char* aModelPath, int aNCep, int aNContext,
             const char* aAlphabetConfigPath)
{
  mPriv = new Private;

  if (!aModelPath) {
    return;
  }

  Status status = NewSession(SessionOptions(), &mPriv->session);
  if (!status.ok()) {
    return;
  }

  status = ReadBinaryProto(Env::Default(), aModelPath, &mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = NULL;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    mPriv->session->Close();
    mPriv->session = NULL;
    return;
  }

  mPriv->ncep = aNCep;
  mPriv->ncontext = aNContext;

  mPriv->alphabet = new Alphabet(aAlphabetConfigPath);
}

Model::~Model()
{
  if (mPriv->session) {
    mPriv->session->Close();
  }

  delete mPriv->alphabet;

  delete mPriv;
}

void
Model::getInputVector(const short* aBuffer, unsigned int aBufferSize,
                           int aSampleRate, float** aMfcc, int* aNFrames,
                           int* aFrameLen)
{
  return audioToInputVector(aBuffer, aBufferSize, aSampleRate, mPriv->ncep,
                            mPriv->ncontext, aMfcc, aNFrames, aFrameLen);
}

char*
Model::infer(float* aMfcc, int aNFrames, int aFrameLen)
{
  if (!mPriv->session) {
    return NULL;
  }

  const int frameSize = mPriv->ncep + (2 * mPriv->ncep * mPriv->ncontext);
  if (aFrameLen == 0) {
    aFrameLen = frameSize;
  } else if (aFrameLen < frameSize) {
    std::cerr << "mfcc features array is too small (expected " <<
      frameSize << ", got " << aFrameLen << ")\n";
    return NULL;
  }

  Tensor input(DT_FLOAT, TensorShape({1, aNFrames, frameSize}));

  auto input_mapped = input.tensor<float, 3>();
  for (int i = 0, idx = 0; i < aNFrames; i++) {
    for (int j = 0; j < frameSize; j++, idx++) {
      input_mapped(0, i, j) = aMfcc[idx];
    }
    idx += (aFrameLen - frameSize);
  }

  Tensor n_frames(DT_INT32, TensorShape({1}));
  n_frames.scalar<int>()() = aNFrames;

  std::vector<Tensor> outputs;
  Status status = mPriv->session->Run(
    {{ "input_node", input }, { "input_lengths", n_frames }},
    {"output_node"}, {}, &outputs);
  if (!status.ok()) {
    std::cerr << "Error running session: " << status.ToString() << "\n";
    return NULL;
  }

  // Output is an array of shape (1, n_results, result_length).
  // In this case, n_results is also equal to 1.
  auto output_mapped = outputs[0].tensor<int64, 3>();
  size_t output_length = output_mapped.dimension(2) + 1;

  size_t decoded_length = 1; // add 1 for the \0
  for (size_t i = 0; i < output_length - 1; i++) {
    int64 character = output_mapped(0, 0, i);
    const std::string& str = mPriv->alphabet->StringFromLabel(character);
    decoded_length += str.size();
  }

  char* output = (char*)malloc(sizeof(char) * decoded_length);
  char* pen = output;
  for (size_t i = 0; i < output_length - 1; i++) {
    int64 character = output_mapped(0, 0, i);
    const std::string& str = mPriv->alphabet->StringFromLabel(character);
    strncpy(pen, str.c_str(), str.size());
    pen += str.size();
  }
  *pen = '\0';

  return output;
}

char*
Model::stt(const short* aBuffer, unsigned int aBufferSize, int aSampleRate)
{
  float* mfcc;
  char* string;
  int n_frames;

  getInputVector(aBuffer, aBufferSize, aSampleRate, &mfcc, &n_frames, NULL);
  string = infer(mfcc, n_frames);
  free(mfcc);
  return string;
}

}
