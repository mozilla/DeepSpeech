add_library(fstcompact
  compact8_acceptor-fst.cc 
  compact8_string-fst.cc 
  compact8_unweighted-fst.cc 
  compact8_unweighted_acceptor-fst.cc 
  compact8_weighted_string-fst.cc 
  compact16_acceptor-fst.cc 
  compact16_string-fst.cc 
  compact16_unweighted-fst.cc 
  compact16_unweighted_acceptor-fst.cc 
  compact16_weighted_string-fst.cc 
  compact64_acceptor-fst.cc 
  compact64_string-fst.cc 
  compact64_unweighted-fst.cc 
  compact64_unweighted_acceptor-fst.cc 
  compact64_weighted_string-fst.cc
)

target_link_libraries(fstcompact fst)
set_target_properties(fstcompact PROPERTIES 
  SOVERSION "${SOVERSION}"
  FOLDER compact
)

install(TARGETS fstcompact 
	        LIBRARY DESTINATION lib
			ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
)

function (add_module _name)
    add_library(${ARGV})
    if (TARGET ${_name})
        target_link_libraries(${_name} fst)
        set_target_properties(${_name} PROPERTIES 
            WINDOWS_EXPORT_ALL_SYMBOLS true
            FOLDER compact/modules
        )
    endif()

    #set_target_properties(${_name} PROPERTIES SOVERSION "1")
    install(TARGETS ${_name} 
	        LIBRARY DESTINATION lib/fst
			ARCHIVE DESTINATION lib/fst
            RUNTIME DESTINATION lib/fst)
endfunction()

add_module(compact8_acceptor-fst MODULE
  compact8_acceptor-fst.cc)
 
add_module(compact8_string-fst MODULE
  compact8_string-fst.cc)
 
add_module(compact8_unweighted-fst MODULE
  compact8_unweighted-fst.cc)
 
add_module(compact8_unweighted_acceptor-fst MODULE
  compact8_unweighted_acceptor-fst.cc)
 
add_module(compact8_weighted_string-fst MODULE
  compact8_weighted_string-fst.cc)
 
add_module(compact16_acceptor-fst MODULE
  compact16_acceptor-fst.cc)
 
add_module(compact16_string-fst MODULE
  compact16_string-fst.cc)
 
add_module(compact16_unweighted-fst MODULE
  compact16_unweighted-fst.cc)
 
add_module(compact16_unweighted_acceptor-fst MODULE
  compact16_unweighted_acceptor-fst.cc)
 
add_module(compact16_weighted_string-fst MODULE
  compact16_weighted_string-fst.cc)
 
add_module(compact64_acceptor-fst MODULE
  compact64_acceptor-fst.cc)
 
add_module(compact64_string-fst MODULE
  compact64_string-fst.cc)
 
add_module(compact64_unweighted-fst MODULE
  compact64_unweighted-fst.cc)
 
add_module(compact64_unweighted_acceptor-fst MODULE
  compact64_unweighted_acceptor-fst.cc)
 
add_module(compact64_weighted_string-fst MODULE
  compact64_weighted_string-fst.cc)


