This file contains some notes on coding style within the C++ portion of the
DeepSpeech project. It is very much a work in progress and incomplete.

General
=======

The code has been imported from various places. Please try and blend in your
code with the surrounding code.

If you are coding from scratch, please follow [Google coding guidelines for C++](https://google.github.io/styleguide/cppguide.html).

Variable naming
===============

* class/struct member variables which are private should follow lowercase with
  a final underscore, e.g. `unsigned int beam_width_;` in `modelstate.h`.


Doubts
======

If in doubt, please ask on our IRC channel, `#machinelearning` on `irc.mozilla.org`.
