// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Getters for converting command-line arguments into the appropriate enums
// or bitmasks, with the simplest ones defined as inline.

#ifndef FST_SCRIPT_GETTERS_H_
#define FST_SCRIPT_GETTERS_H_

#include <string>

#include <fst/compose.h>          // For ComposeFilter.
#include <fst/determinize.h>      // For DeterminizeType.
#include <fst/encode.h>           // For kEncodeLabels (etc.).
#include <fst/epsnormalize.h>     // For EpsNormalizeType.
#include <fst/project.h>          // For ProjectType.
#include <fst/push.h>             // For kPushWeights (etc.).
#include <fst/queue.h>            // For QueueType.
#include <fst/rational.h>         // For ClosureType.
#include <fst/script/arcsort.h>       // For ArcSortType.
#include <fst/script/map.h>           // For MapType.
#include <fst/script/script-impl.h>   // For RandArcSelection.

#include <fst/log.h>

namespace fst {
namespace script {

bool GetArcSortType(const string &str, ArcSortType *sort_type);

inline ClosureType GetClosureType(bool closure_plus) {
  return closure_plus ? CLOSURE_PLUS : CLOSURE_STAR;
}

bool GetComposeFilter(const string &str, ComposeFilter *compose_filter);

bool GetDeterminizeType(const string &str, DeterminizeType *det_type);

inline uint32_t GetEncodeFlags(bool encode_labels, bool encode_weights) {
  return (encode_labels ? kEncodeLabels : 0) |
         (encode_weights ? kEncodeWeights : 0);
}

inline EpsNormalizeType GetEpsNormalizeType(bool eps_norm_output) {
  return eps_norm_output ? EPS_NORM_OUTPUT : EPS_NORM_INPUT;
}

bool GetMapType(const string &str, MapType *map_type);

inline ProjectType GetProjectType(bool project_output) {
  return project_output ? PROJECT_OUTPUT : PROJECT_INPUT;
}

inline uint32_t GetPushFlags(bool push_weights, bool push_labels,
                           bool remove_total_weight, bool remove_common_affix) {
  return ((push_weights ? kPushWeights : 0) |
          (push_labels ? kPushLabels : 0) |
          (remove_total_weight ? kPushRemoveTotalWeight : 0) |
          (remove_common_affix ? kPushRemoveCommonAffix : 0));
}

bool GetQueueType(const string &str, QueueType *queue_type);

bool GetRandArcSelection(const string &str, RandArcSelection *ras);

bool GetReplaceLabelType(const string &str, bool epsilon_on_replace,
                         ReplaceLabelType *rlt);

inline ReweightType GetReweightType(bool to_final) {
  return to_final ? REWEIGHT_TO_FINAL : REWEIGHT_TO_INITIAL;
}

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_GETTERS_H_
