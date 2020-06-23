DEEPSPEECH_REPO ?= https://github.com/mozilla/DeepSpeech.git
DEEPSPEECH_SHA  ?= origin/master

Dockerfile%: Dockerfile%.tmpl
	sed \
		-e "s|#DEEPSPEECH_REPO#|$(DEEPSPEECH_REPO)|g" \
		-e "s|#DEEPSPEECH_SHA#|$(DEEPSPEECH_SHA)|g" \
		< $< > $@
