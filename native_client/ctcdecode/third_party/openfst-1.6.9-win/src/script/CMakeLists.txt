file(GLOB HEADER_FILES ../include/fst/script/*.h)
message(STATUS "${HEADER_FILES}")

add_library(fstscript
  arciterator-class.cc
  arcsort.cc
  closure.cc
  compile.cc
  compose.cc
  concat.cc
  connect.cc
  convert.cc
  decode.cc
  determinize.cc
  difference.cc
  disambiguate.cc
  draw.cc
  encode.cc
  encodemapper-class.cc
  epsnormalize.cc
  equal.cc
  equivalent.cc
  fst-class.cc
  getters.cc
  info.cc
  info-impl.cc
  intersect.cc
  invert.cc
  isomorphic.cc
  map.cc
  minimize.cc
  print.cc
  project.cc
  prune.cc
  push.cc
  randequivalent.cc
  randgen.cc
  relabel.cc
  replace.cc
  reverse.cc
  reweight.cc
  rmepsilon.cc
  shortest-distance.cc
  shortest-path.cc
  stateiterator-class.cc
  synchronize.cc
  text-io.cc
  topsort.cc
  union.cc
  weight-class.cc
  verify.cc
  ${HEADER_FILES}
)
target_link_libraries(fstscript PRIVATE fst)

set_target_properties(fstscript PROPERTIES
  SOVERSION "${SOVERSION}"
)
install(TARGETS fstscript
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib)

