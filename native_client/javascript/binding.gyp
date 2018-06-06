{
  "targets": [
    {
      "target_name": "deepspeech",
      "sources": [ "deepspeech_wrap.cxx" ],
      "libraries": [
        "${LIBS}"
      ],
      "include_dirs": [
        "../"
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
  },
}
