{
  "targets": [
    {
      "target_name": "deepspeech",
      "sources": [ "deepspeech_wrap.cxx" ],
      "libraries": [
        "-ltensorflow", "-ldeepspeech", "-ldeepspeech_utils"
      ],
      "include_dirs": [
        "../"
      ]
    }
  ]
}
