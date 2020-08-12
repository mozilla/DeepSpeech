MOZILLA_VOICE_STT_REPO ?= https://github.com/mozilla/STT.git
MOZILLA_VOICE_STT_SHA  ?= origin/master

Dockerfile%: Dockerfile%.tmpl
	sed \
		-e "s|#MOZILLA_VOICE_STT_REPO#|$(MOZILLA_VOICE_STT_REPO)|g" \
		-e "s|#MOZILLA_VOICE_STT_SHA#|$(MOZILLA_VOICE_STT_SHA)|g" \
		< $< > $@
