FILE(GLOB HEADER_FILES ../include/fst/*.h)


add_library(fst
  compat.cc
  flags.cc
  fst-types.cc
  fst.cc
  mapped-file.cc
  properties.cc
  symbol-table.cc
  symbol-table-ops.cc
  util.cc
  weight.cc
  ${HEADER_FILES}
)
set_target_properties(fst PROPERTIES
  SOVERSION "${SOVERSION}"
)



install(TARGETS fst
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib)
