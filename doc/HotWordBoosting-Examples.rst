Hot-word boosting API Usage example
===================================

With DeepSpeech 0.9 release a new API feature was introduced that allows boosting probability from the scorer of given words. It is exposed in all bindings (C, Python, JS, Java and .Net). 

Currently, it provides three methods for the Model class:

- ``AddHotWord(word, boost)``
- ``EraseHotWord(word)`` 
- ``ClearHotWords()``

Exact API binding for the language you are using can be found in API Reference.

General usage
-------------

It is worth noting that boosting non-existent words in scorer (mostly proper nouns) or a word that share no phonetic prefix with other word in the input audio don't change the final transcription. Additionally, hot-word that has a space will not be taken into consideration, meaning that combination of words can not be boosted and each word must be added as hot-word separately. 

Adjusting the boosting value
----------------------------

For hot-word boosting it is hard to determine what the optimal value that one might be searching for is. Additionally, this is dependant on the input audio file. In practice, as it was reported by DeepSpeech users, the value should be not bigger than 20.0 for positive value boosting. Nevertheless, each usecase is different and you might need to adjust values on your own.

There is a user contributed script available on ``DeepSpeech-examples`` repository for adjusting boost values:

`https://github.com/mozilla/DeepSpeech-examples/tree/master/hotword_adjusting <https://github.com/mozilla/DeepSpeech-examples/tree/master/hotword_adjusting>`_.


Positive value boosting
-----------------------

By adding a positive boost value to one of the words it is possible to increase the probability of the word occurence. This is particularly useful for detecting speech that is expected by the system. 

In the output, overextensive positive boost value (e.g. 250.0 but it does vary) may cause a word following the boosted hot-word to be split into separate letters. This problem is related to the scorer structure and currently only way to avoid it is to tune boost to a lower value.  

Negative value boosting
-----------------------

Respectively, applying negative boost value might cause the selected word to occur less frequently. Keep in mind that words forming similar sound of a boosted word might be used instead (e.g. homophones "accept" as "except") or it will be split into separate parts (e.g. "another" into "an other").

Previously mentioned problem where extensive boost value caused letter splitting doesn't arise for negative boost values.

Example 
-------

To use hot-word boosting just add hot-words of your choice before performing an inference to a ``Model``. You can also erase boosting of a chosen word or clear it for all hot-words.

.. code-block:: python

	ds = Model(args.model)
	...
	ds.addHotWord(word, boosting)
	...
	print(ds.stt(audio))
	
Adding boost value to a word repeatedly or erasing hot-word without previously boosting it results in an error.
