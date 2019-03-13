function (add_executable2 _name)
    add_executable(${ARGV})
    if (TARGET ${_name})
        target_link_libraries(${_name} fstscript fst ${CMAKE_DL_LIBS})
        set_target_properties(${_name} PROPERTIES FOLDER bin)
    endif()

    install(TARGETS ${_name} RUNTIME DESTINATION bin)
endfunction()

include_directories(../include ../script/)

add_executable2(fstarcsort fstarcsort-main.cc fstarcsort.cc)

add_executable2(fstclosure fstclosure-main.cc fstclosure.cc)

add_executable2(fstcompile  fstcompile-main.cc fstcompile.cc)

add_executable2(fstcompose fstcompose-main.cc fstcompose.cc)

add_executable2(fstconcat fstconcat-main.cc fstconcat.cc)

add_executable2(fstconnect fstconnect-main.cc fstconnect.cc)

add_executable2(fstconvert fstconvert-main.cc fstconvert.cc)

add_executable2(fstdeterminize fstdeterminize-main.cc fstdeterminize.cc)

add_executable2(fstdifference fstdifference-main.cc fstdifference.cc)

add_executable2(fstdisambiguate fstdisambiguate-main.cc fstdisambiguate.cc)

add_executable2(fstdraw fstdraw-main.cc fstdraw.cc)

add_executable2(fstencode fstencode-main.cc fstencode.cc)

add_executable2(fstepsnormalize fstepsnormalize-main.cc fstepsnormalize.cc)

add_executable2(fstequal fstequal-main.cc fstequal.cc)

add_executable2(fstequivalent fstequivalent-main.cc fstequivalent.cc)

add_executable2(fstinfo fstinfo-main.cc fstinfo.cc)

add_executable2(fstintersect fstintersect-main.cc fstintersect.cc)

add_executable2(fstinvert fstinvert-main.cc fstinvert.cc)

add_executable2(fstisomorphic fstisomorphic-main.cc fstisomorphic.cc)

add_executable2(fstmap fstmap-main.cc fstmap.cc)

add_executable2(fstminimize fstminimize-main.cc fstminimize.cc)

add_executable2(fstprint fstprint-main.cc fstprint.cc)

add_executable2(fstproject fstproject-main.cc fstproject.cc)

add_executable2(fstprune fstprune-main.cc fstprune.cc)

add_executable2(fstpush fstpush-main.cc fstpush.cc)

add_executable2(fstrandgen fstrandgen-main.cc fstrandgen.cc)

add_executable2(fstrelabel fstrelabel-main.cc fstrelabel.cc)

add_executable2(fstreplace fstreplace-main.cc fstreplace.cc)

add_executable2(fstreverse fstreverse-main.cc fstreverse.cc)

add_executable2(fstreweight fstreweight-main.cc fstreweight.cc)

add_executable2(fstrmepsilon fstrmepsilon-main.cc fstrmepsilon.cc)

add_executable2(fstshortestdistance fstshortestdistance-main.cc fstshortestdistance.cc)

add_executable2(fstshortestpath fstshortestpath-main.cc fstshortestpath.cc)

add_executable2(fstsymbols fstsymbols-main.cc fstsymbols.cc)

add_executable2(fstsynchronize fstsynchronize-main.cc fstsynchronize.cc)

add_executable2(fsttopsort fsttopsort-main.cc fsttopsort.cc)

add_executable2(fstunion fstunion-main.cc fstunion.cc)

