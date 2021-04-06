GitHub Action to set NumPy versions
===================================

This actions aims at computing correct values for NumPy dependencies:
 - `NUMPY_BUILD_VERSION`: range of accepted versions at Python binding build time
 - `NUMPY_DEP_VERSION`: range of accepted versions for execution time

Versions are set considering several factors:
 - API and ABI compatibility ; otherwise we can have the binding wrapper
   throwing errors like "Illegal instruction", or computing wrong values
   because of changed memory layout
 - Wheels availability: for CI and end users, we want to avoid having to
   rebuild numpy so we stick to versions where there is an existing upstream
   `wheel` file
