{
  "targets": [
    {
      "target_name": "deepspeech",
      "sources": [ "deepspeech_wrap.cxx" ],
      "libraries": [
        "$(LIBS)"
      ],
      "include_dirs": [
        "../"
      ],
      "conditions": [
        [ "OS=='mac'", {
            "xcode_settings": {
              "OTHER_CXXFLAGS": [
                "-stdlib=libc++",
                "-mmacosx-version-min=10.10"
              ],
              "OTHER_LDFLAGS": [
                "-stdlib=libc++",
                "-mmacosx-version-min=10.10"
              ]
            }
          }
        ]
      ]
    },
    {
      "target_name": "action_after_build",
      "type": "none",
      "dependencies": [ "<(module_name)" ],
      "copies": [
        {
          "files": [ "<(PRODUCT_DIR)/<(module_name).node" ],
          "destination": "<(module_path)"
        }
      ]
    }
  ],
  "variables": {
    "build_v8_with_gn": 0,
    "v8_enable_pointer_compression": 0,
    "v8_enable_31bit_smis_on_64bit_arch": 0,
    "enable_lto": 1
  },
}
