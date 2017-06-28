{
  "targets": [
    {
      "target_name": "deepspeech",
      "sources": [ "deepspeech_wrap.cxx" ],
      "libraries": [
        "-ltensorflow_cc", "-ldeepspeech", "-ldeepspeech_utils"
      ],
      "include_dirs": [
        "../"
      ]
    }
  ]
}
