file(GLOB HEADER_FILES ../../include/fst/extensions/linear/*.h)
message(STATUS "${HEADER_FILES}")


if(HAVE_SCRIPT)
  add_library(fstlinearscript
    linearscript.cc
    ${HEADER_FILES}
  )
  target_link_libraries(fstlinearscript
    fstscript
    fst
  )
  set_target_properties(fstlinearscript PROPERTIES 
    SOVERSION "${SOVERSION}"
    FOLDER linear
  )
  
  install(TARGETS fstlinearscript
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION lib
  )
endif(HAVE_SCRIPT)

if(HAVE_BIN)
  add_executable(fstlinear
    fstlinear.cc)
  target_link_libraries(fstlinear
    fstlinearscript
    fstscript 
    fst
    ${CMAKE_DL_LIBS}
 )

  add_executable(fstloglinearapply
    fstloglinearapply.cc)
  target_link_libraries(fstloglinearapply
    fstlinearscript
    fstscript 
    fst
    ${CMAKE_DL_LIBS}
  )
  install(TARGETS fstlinear fstloglinearapply
    RUNTIME DESTINATION bin
  )
  set_target_properties(fstlinear fstloglinearapply PROPERTIES
    FOLDER linear/bin
  )
endif(HAVE_BIN)


function (add_module _name)
    add_library(${ARGV})
    if (TARGET ${_name})
        target_link_libraries(${_name} fst)
        set_target_properties(${_name} PROPERTIES 
            WINDOWS_EXPORT_ALL_SYMBOLS true
            FOLDER linear/modules
        )
    endif()

    install(TARGETS ${_name} LIBRARY DESTINATION lib/fst)
endfunction()

add_module(linear-tagger-fst MODULE
  linear-tagger-fst.cc)

add_module(linear-classifier-fst MODULE
  linear-classifier-fst.cc)

