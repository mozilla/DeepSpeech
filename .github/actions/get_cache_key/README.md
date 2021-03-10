GitHub Action to compute cache key
==================================

It is intended to work in harmony with `check_artifact_exists`:
 - compute a stable cache key
 - as simple to use as possible (less parameters)

It will expect to be ran in a GitHub Action job that follows
`SUBMODULE_FLAVOR-PLATFORM`:
 - it will use the `SUBMODULE` part to check what is the current SHA1 of this git submodule.
 - the `FLAVOR` allows to distringuish e.g., opt/dbg builds
 - the PLATFORM permits defining an os/arch couple

It allows for an `extras` field for extensive customization, like forcing a
re-build.
