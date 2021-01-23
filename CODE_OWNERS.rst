DeepSpeech code owners / governance system
==========================================

DeepSpeech is run under a governance system inspired (and partially copied from) by the `Mozilla module ownership system <https://www.mozilla.org/about/governance/policies/module-ownership/>`_. The project is roughly divided into modules, and each module has its own owners, which are responsible for reviewing pull requests and deciding on technical direction for their modules. Module ownership authority is given to people who have worked extensively on areas of the project.

Module owners also have the authority of naming other module owners or appointing module peers, which are people with authority to review pull requests in that module. They can also sub-divide their module into sub-modules with their own owners.

Module owners are not tyrants. They are chartered to make decisions with input from the community and in the best interests of the community. Module owners are not required to make code changes or additions solely because the community wants them to do so. (Like anyone else, the module owners may write code because they want to, because their employers want them to, because the community wants them to, or for some other reason.) Module owners do need to pay attention to patches submitted to that module. However “pay attention” does not mean agreeing to every patch. Some patches may not make sense for the WebThings project; some may be poorly implemented. Module owners have the authority to decline a patch; this is a necessary part of the role. We ask the module owners to describe in the relevant issue their reasons for wanting changes to a patch, for declining it altogether, or for postponing review for some period. We don’t ask or expect them to rewrite patches to make them acceptable. Similarly, module owners may need to delay review of a promising patch due to an upcoming deadline. For example, a patch may be of interest, but not for the next milestone. In such a case it may make sense for the module owner to postpone review of a patch until after matters needed for a milestone have been finalized. Again, we expect this to be described in the relevant issue. And of course, it shouldn’t go on very often or for very long or escalation and review is likely.

The work of the various module owners and peers is overseen by the global owners, which are responsible for making final decisions in case there's conflict between owners as well as set the direction for the project as a whole.

This file describes module owners who are active on the project and which parts of the code they have expertise on (and interest in). If you're making changes to the code and are wondering who's an appropriate person to talk to, this list will tell you who to ping.

There's overlap in the areas of expertise of each owner, and in particular when looking at which files are covered by each area, there is a lot of overlap. Don't worry about getting it exactly right when requesting review, any code owner will be happy to redirect the request to a more appropriate person.

Global owners
----------------

These are people who have worked on the project extensively and are familiar with all or most parts of it. Their expertise and review guidance is trusted by other code owners to cover their own areas of expertise. In case of conflicting opinions from other owners, global owners will make a final decision.

- Alexandre Lissy (@lissyx)
- Reuben Morais (@reuben)

Training, feeding
-----------------

- Reuben Morais (@reuben)

Model exporting
---------------

- Alexandre Lissy (@lissyx)

Transfer learning
-----------------

- Josh Meyer (@JRMeyer)
- Reuben Morais (@reuben)

Testing & CI
------------

- Alexandre Lissy (@lissyx)
- Reuben Morais (@reuben)

Native inference client
-----------------------

Everything that goes into libdeepspeech.so and is not specifically covered in another area fits here.

- Alexandre Lissy (@lissyx)
- Reuben Morais (@reuben)

Streaming decoder
-----------------

- Reuben Morais (@reuben)
- @dabinat

Python bindings
---------------

- Alexandre Lissy (@lissyx)
- Reuben Morais (@reuben)

Java Bindings
-------------

- Alexandre Lissy (@lissyx)

JavaScript/NodeJS/ElectronJS bindings
-------------------------------------

- Alexandre Lissy (@lissyx)
- Reuben Morais (@reuben)

.NET bindings
-------------

- Carlos Fonseca (@carlfm01)

Swift bindings
--------------

- Reuben Morais (@reuben)

Android support
---------------

- Alexandre Lissy (@lissyx)

Raspberry Pi support
--------------------

- Alexandre Lissy (@lissyx)

Windows support
---------------

- Carlos Fonseca (@carlfm01)

iOS support
-----------

- Reuben Morais (@reuben)

Documentation
-------------

- Alexandre Lissy (@lissyx)
- Reuben Morais (@reuben)

Third party bindings
--------------------

Hosted externally and owned by the individual authors. See the `list of third-party bindings <https://deepspeech.readthedocs.io/en/master/USING.html#third-party-bindings>`_ for more info.
