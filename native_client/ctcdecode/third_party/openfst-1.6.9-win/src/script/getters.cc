#include <fst/script/getters.h>

namespace fst {
namespace script {

bool GetArcSortType(const string &str, ArcSortType *sort_type) {
  if (str == "ilabel") {
    *sort_type = ILABEL_SORT;
  } else if (str == "olabel") {
    *sort_type = OLABEL_SORT;
  } else {
    return false;
  }
  return true;
}

bool GetComposeFilter(const string &str, ComposeFilter *compose_filter) {
  if (str == "alt_sequence") {
    *compose_filter = ALT_SEQUENCE_FILTER;
  } else if (str == "auto") {
    *compose_filter = AUTO_FILTER;
  } else if (str == "match") {
    *compose_filter = MATCH_FILTER;
  } else if (str == "null") {
    *compose_filter = NULL_FILTER;
  } else if (str == "sequence") {
    *compose_filter = SEQUENCE_FILTER;
  } else if (str == "trivial") {
    *compose_filter = TRIVIAL_FILTER;
  } else {
    return false;
  }
  return true;
}

bool GetDeterminizeType(const string &str, DeterminizeType *det_type) {
  if (str == "functional") {
    *det_type = DETERMINIZE_FUNCTIONAL;
  } else if (str == "nonfunctional") {
    *det_type = DETERMINIZE_NONFUNCTIONAL;
  } else if (str == "disambiguate") {
    *det_type = DETERMINIZE_DISAMBIGUATE;
  } else {
    return false;
  }
  return true;
}

bool GetMapType(const string &str, MapType *map_type) {
  if (str == "arc_sum") {
    *map_type = ARC_SUM_MAPPER;
  } else if (str == "arc_unique") {
    *map_type = ARC_UNIQUE_MAPPER;
  } else if (str == "identity") {
    *map_type = IDENTITY_MAPPER;
  } else if (str == "input_epsilon") {
    *map_type = INPUT_EPSILON_MAPPER;
  } else if (str == "invert") {
    *map_type = INVERT_MAPPER;
  } else if (str == "output_epsilon") {
    *map_type = OUTPUT_EPSILON_MAPPER;
  } else if (str == "plus") {
    *map_type = PLUS_MAPPER;
  } else if (str == "power") {
    *map_type = POWER_MAPPER;
  } else if (str == "quantize") {
    *map_type = QUANTIZE_MAPPER;
  } else if (str == "rmweight") {
    *map_type = RMWEIGHT_MAPPER;
  } else if (str == "superfinal") {
    *map_type = SUPERFINAL_MAPPER;
  } else if (str == "times") {
    *map_type = TIMES_MAPPER;
  } else if (str == "to_log") {
    *map_type = TO_LOG_MAPPER;
  } else if (str == "to_log64") {
    *map_type = TO_LOG64_MAPPER;
  } else if (str == "to_std" || str == "to_standard") {
    *map_type = TO_STD_MAPPER;
  } else {
    return false;
  }
  return true;
}

bool GetRandArcSelection(const string &str, RandArcSelection *ras) {
  if (str == "uniform") {
    *ras = UNIFORM_ARC_SELECTOR;
  } else if (str == "log_prob") {
    *ras = LOG_PROB_ARC_SELECTOR;
  } else if (str == "fast_log_prob") {
    *ras = FAST_LOG_PROB_ARC_SELECTOR;
  } else {
    return false;
  }
  return true;
}

bool GetQueueType(const string &str, QueueType *queue_type) {
  if (str == "auto") {
    *queue_type = AUTO_QUEUE;
  } else if (str == "fifo") {
    *queue_type = FIFO_QUEUE;
  } else if (str == "lifo") {
    *queue_type = LIFO_QUEUE;
  } else if (str == "shortest") {
    *queue_type = SHORTEST_FIRST_QUEUE;
  } else if (str == "state") {
    *queue_type = STATE_ORDER_QUEUE;
  } else if (str == "top") {
    *queue_type = TOP_ORDER_QUEUE;
  } else {
    return false;
  }
  return true;
}

bool GetReplaceLabelType(const string &str, bool epsilon_on_replace,
                         ReplaceLabelType *rlt) {
  if (epsilon_on_replace || str == "neither") {
    *rlt = REPLACE_LABEL_NEITHER;
  } else if (str == "input") {
    *rlt = REPLACE_LABEL_INPUT;
  } else if (str == "output") {
    *rlt = REPLACE_LABEL_OUTPUT;
  } else if (str == "both") {
    *rlt = REPLACE_LABEL_BOTH;
  } else {
    return false;
  }
  return true;
}

}  // namespace script
}  // namespace fst
