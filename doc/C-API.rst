.. _c-usage:

C API
=====

.. toctree::
   :maxdepth: 2

   Structs

See also the list of error codes including descriptions for each error in :ref:`error-codes`.

.. doxygenfunction:: DS_CreateModel
   :project: deepspeech-c

.. doxygenfunction:: DS_FreeModel
   :project: deepspeech-c

.. doxygenfunction:: DS_EnableExternalScorer
   :project: deepspeech-c

.. doxygenfunction:: DS_DisableExternalScorer
   :project: deepspeech-c

.. doxygenfunction:: DS_AddHotWord
   :project: deepspeech-c

.. doxygenfunction:: DS_EraseHotWord
   :project: deepspeech-c

.. doxygenfunction:: DS_ClearHotWord
   :project: deepspeech-c

.. doxygenfunction:: DS_SetScorerAlphaBeta
   :project: deepspeech-c

.. doxygenfunction:: DS_GetModelSampleRate
   :project: deepspeech-c

.. doxygenfunction:: DS_SpeechToText
   :project: deepspeech-c

.. doxygenfunction:: DS_SpeechToTextWithMetadata
   :project: deepspeech-c

.. doxygenfunction:: DS_CreateStream
   :project: deepspeech-c

.. doxygenfunction:: DS_FeedAudioContent
   :project: deepspeech-c

.. doxygenfunction:: DS_IntermediateDecode
   :project: deepspeech-c

.. doxygenfunction:: DS_IntermediateDecodeWithMetadata
   :project: deepspeech-c

.. doxygenfunction:: DS_FinishStream
   :project: deepspeech-c

.. doxygenfunction:: DS_FinishStreamWithMetadata
   :project: deepspeech-c

.. doxygenfunction:: DS_FreeStream
   :project: deepspeech-c

.. doxygenfunction:: DS_FreeMetadata
   :project: deepspeech-c

.. doxygenfunction:: DS_FreeString
   :project: deepspeech-c

.. doxygenfunction:: DS_Version
   :project: deepspeech-c
