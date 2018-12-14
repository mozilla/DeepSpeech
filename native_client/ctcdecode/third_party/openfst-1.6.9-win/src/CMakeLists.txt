
#-DHAVE_CONFIG_H -I./../include -fno-exceptions -funsigned-char -std=c++11 -MT symbol-table.lo -MD -MP -MF .deps/symbol-table.Tpo -c symbol-table.cc  -fno-common -DPIC -o .libs/symbol-table.o

include_directories(./include/)
install(DIRECTORY include/ DESTINATION include/
        FILES_MATCHING PATTERN "*.h")

add_subdirectory(lib)
add_subdirectory(script)

if(HAVE_BIN)
  add_subdirectory(bin)
endif(HAVE_BIN)

add_subdirectory(extensions)

enable_testing()
add_subdirectory(test)
