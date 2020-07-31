.. _c-usage:

C API
=====

.. toctree::
   :maxdepth: 2

   Structs

See also the list of error codes including descriptions for each error in :ref:`error-codes`.

.. doxygenfunction:: STT_CreateModel
   :project: deepspeech-c

.. doxygenfunction:: STT_FreeModel
   :project: deepspeech-c

.. doxygenfunction:: STT_EnableExternalScorer
   :project: deepspeech-c

.. doxygenfunction:: STT_DisableExternalScorer
   :project: deepspeech-c

.. doxygenfunction:: STT_SetScorerAlphaBeta
   :project: deepspeech-c

.. doxygenfunction:: STT_GetModelSampleRate
   :project: deepspeech-c

.. doxygenfunction:: STT_SpeechToText
   :project: deepspeech-c

.. doxygenfunction:: STT_SpeechToTextWithMetadata
   :project: deepspeech-c

.. doxygenfunction:: STT_CreateStream
   :project: deepspeech-c

.. doxygenfunction:: STT_FeedAudioContent
   :project: deepspeech-c

.. doxygenfunction:: STT_IntermediateDecode
   :project: deepspeech-c

.. doxygenfunction:: STT_IntermediateDecodeWithMetadata
   :project: deepspeech-c

.. doxygenfunction:: STT_FinishStream
   :project: deepspeech-c

.. doxygenfunction:: STT_FinishStreamWithMetadata
   :project: deepspeech-c

.. doxygenfunction:: STT_FreeStream
   :project: deepspeech-c

.. doxygenfunction:: STT_FreeMetadata
   :project: deepspeech-c

.. doxygenfunction:: STT_FreeString
   :project: deepspeech-c

.. doxygenfunction:: STT_Version
   :project: deepspeech-c

.. doxygenfunction:: STT_ErrorCodeToErrorMessage
   :project: deepspeech-c
