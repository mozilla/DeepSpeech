{
  "targets": [
    {
      "target_name": "deepspeech",
      "sources": [ "deepspeech_wrap.cxx" ],
      "libraries": [
        "-ldeepspeech", "-ldeepspeech_utils", "-ltensorflow_cc"
      ],
      "include_dirs": [
        "../"
      ]
    }
  ]
}
