file(GLOB HEADER_FILES ../../include/fst/extensions/compress/*.h)
message(STATUS "${HEADER_FILES}")

add_library(fstcompressscript
  compress-script.cc
  ${HEADER_FILES}
 )

target_link_libraries(fstcompressscript 
  fstscript
  fst
  ${ZLIBS}
)
set_target_properties(fstcompressscript PROPERTIES
  SOVERSION "10"
)
install(TARGETS fstcompressscript 
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
 )

if(HAVE_BIN)
  add_executable(fstcompress 
    fstcompress.cc)

  target_link_libraries(fstcompress 
    fstcompressscript 
    fstscript 
    fst
    ${ZLIBS}
    ${CMAKE_DL_LIBS}
   )

  add_executable(fstrandmod
    fstrandmod.cc
  )

  target_link_libraries(fstrandmod
    fstcompressscript 
    fstscript 
    fst
    ${ZLIBS}
    ${CMAKE_DL_LIBS}
  )

  install(TARGETS fstcompress fstrandmod 
	        LIBRARY DESTINATION bin
			ARCHIVE DESTINATION bin
            RUNTIME DESTINATION bin
		)
endif(HAVE_BIN)
