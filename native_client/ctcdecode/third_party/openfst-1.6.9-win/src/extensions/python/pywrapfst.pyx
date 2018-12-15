#cython: nonecheck=True, c_string_type=unicode, c_string_encoding=utf8
# See www.openfst.org for extensive documentation on this weighted
# finite-state transducer library.


"""Python interface to the FST scripting API.

Operations which construct new FSTs are implemented as traditional functions, as
are two-argument boolean functions like `equal` and `equivalent`. Destructive
operations---those that mutate an FST, in place---are instance methods, as is
`write`. Operator overloading is not used. The following example, based on
Mohri et al. 2002, shows the construction of an ASR system given a pronunciation
lexicon L, grammar G, a transducer from context-dependent phones to
context-independent phones C, and an HMM set H:

  L = fst.Fst.read("L.fst")
  G = fst.Fst.read("G.fst")
  C = fst.Fst.read("C.fst")
  H = fst.Fst.read("H.fst")
  LG = fst.determinize(fst.compose(L, G))
  CLG = fst.determinize(fst.compose(C, LG))
  HCLG = fst.determinize(fst.compose(H, CLG))
  HCLG.minimize()                                      # NB: works in-place.

Python variables here use snake_case and constants are in all caps, minus the
normal `k` prefix.
"""

# Overview of the file:
#
# * Imports
# * Custom exceptions
# * General helpers
# * Weight and helpers
# * _SymbolTable, _EncodeMapperSymbolTable, _FstSymbolTable,
#   _MutableFstSymbolTable, SymbolTable, and helpers
# * SymbolTableIterator
# * EncodeMapper
# * _Fst, _MutableFst, Fst, and helpers
# * FST properties
# * Arc, ArcIterator, and MutableArcIterator
# * StateIterator
# * FST operations
# * Compiler
# * FarReader and FarWriter
# * Cleanup operations for module entrance and exit.
#
# TODO(kbg): Try breaking this apart into smaller pieces.
#
# A few of the more idiosyncratic choices made here are due to "impedance
# mismatches" between C++ and Python, as follows.
#
# Due to differences in C++ and Python scope rules, most C++ class instances
# have to be heap-allocated. Since all are packed into Python class instances,
# Python destructors are used to semi-automatically free C++ instances.
#
# Cython's type annotations (e.g., `string`) are used when the variables will
# be sent as arguments to C++ functions, but are not used for variables used
# within the module.


## Imports.

# C imports.
from libc.stdint cimport INT32_MAX
from libc.stdint cimport SIZE_MAX
from posix.unistd cimport getpid

# C++ imports.
from libcpp cimport bool
from libcpp.cast cimport const_cast
from libcpp.cast cimport static_cast

# Our C++ imports.
from ios cimport ofstream
from memory cimport static_pointer_cast

# Cython operator workarounds.
from cython.operator cimport address as addr       # &foo
from cython.operator cimport dereference as deref  # *foo
from cython.operator cimport preincrement as inc   # ++foo

# Python imports.
import atexit
import numbers
import subprocess
import logging


# TODO(kbg): Figure out how to access static class variables so I don't have
# to do it this way.
kNoSymbol = -1


## Custom exceptions.


class FstError(Exception):

  pass


class FstArgError(FstError, ValueError):

  pass


class FstBadWeightError(FstError, ValueError):

  pass


class FstDeletedConstructorError(FstError, RuntimeError):

  pass


class FstIndexError(FstError, IndexError):

  pass


class FstIOError(FstError, IOError):

  pass


class FstOpError(FstError, RuntimeError):

  pass


## General helpers.


cdef string tostring(data, encoding="utf8") except *:
  """Converts strings to bytestrings.

  This function converts Python bytestrings and Unicode strings to bytestrings
  encoded in UTF-8. It is used to process most Python string arguments before
  passing them to the lower-level library.

  Args:
    data: A Unicode string or bytestring.
    encoding: The desired encoding, defaulting to UTF-8.

  Returns:
    A bytestring.

  Raises:
    FstArgError: Cannot encode string.
    UnicodeEncodeError.

  This function is not visible to Python users.
  """
  # A Python bytestring can be implicitly cast to a C++ string.
  if isinstance(data, bytes):
    return data
  elif isinstance(data, unicode):
    return data.encode(encoding)
  raise FstArgError("Cannot encode as string: {!r}".format(data))


cdef string weight_tostring(data, encoding="utf8") except *:
  """Converts strings or numerics to bytestrings.

  This function converts Python bytestrings, Unicode strings, and numerics
  which can be cast to floats to bytestrings encoded in UTF-8. It is used to
  process Python string arguments so they can be used to construct Weight
  objects. In most cases, weights are underlyingly floating-point, but since
  not all weights are, they can only be constructed using a string.

  Args:
    data: A Unicode string, bytestring, or type which can be converted to a
      Python float.

  Returns:
    A bytestring.

  Raise:
    FstArgError: Cannot encode string.
    ValueError: Invalid literal for float.
    UnicodeEncodeError.

  This function is not visible to Python users.
  """
  # A Python bytestring can be implicitly cast to a C++ string.
  if isinstance(data, bytes):
    return data
  elif isinstance(data, unicode):
    return data.encode(encoding)
  elif isinstance(data, numbers.Number):
    return str(data).encode(encoding)
  raise FstArgError("Cannot encode as string: {!r}".format(data))


cdef fst.ComposeFilter _get_compose_filter(
    const string &compose_filter) except *:
  """Matches string with the appropriate ComposeFilter enum value.

  This function takes a string argument and returns the matching ComposeFilter
  enum value used to initialize ComposeOptions instances. ComposeOptions is used
  by difference and intersection in addition to composition.

  Args:
    compose_filter: A string matching a known composition filter; one of:
        "alt_sequence", "auto", "match", "null", "sequence", "trivial".

  Returns:
    A ComposeFilter enum value.

  Raises:
    FstArgError: Unknown compose filter type.

  This function is not visible to Python users.
  """
  cdef fst.ComposeFilter compose_filter_enum
  if not fst.GetComposeFilter(compose_filter, addr(compose_filter_enum)):
    raise FstArgError("Unknown compose filter type: {!r}".format(
        compose_filter))
  return compose_filter_enum


cdef fst.DeterminizeType _get_determinize_type(const string &det_type) except *:
  """Matches string with the appropriate DeterminizeType enum value.

  Args:
    det_type: A string matching a known determinization type; one of:
        "functional", "nonfunctional", "disambiguate".

  Returns:
    A DeterminizeType enum value.

  Raises:
    FstArgError: Unknown determinization type.

  This function is not visible to Python users.
  """
  cdef fst.DeterminizeType det_type_enum
  if not fst.GetDeterminizeType(det_type, addr(det_type_enum)):
    raise FstArgError("Unknown determinization type: {!r}".format(det_type))
  return det_type_enum


cdef fst.QueueType _get_queue_type(const string &queue_type) except *:
  """Matches string with the appropriate QueueType enum value.

  This function takes a string argument and returns the matching QueueType enum
  value passed to the RmEpsilonOptions constructor.

  Args:
    queue_type: A string matching a known queue type; one of: "auto", "fifo",
        "lifo", "shortest", "state", "top".

  Returns:
    A QueueType enum value.

  Raises:
    FstArgError: Unknown queue type.

  This function is not visible to Python users.
  """
  cdef fst.QueueType queue_type_enum
  if not fst.GetQueueType(queue_type, addr(queue_type_enum)):
    raise FstArgError("Unknown queue type: {!r}".format(queue_type))
  return queue_type_enum


cdef fst.RandArcSelection _get_rand_arc_selection(
    const string &select) except *:
  """Matches string with the appropriate RandArcSelection enum value.

  This function takes a string argument and returns the matching
  RandArcSelection enum value passed to the RandGenOptions constructor.

  Args:
    select: A string matching a known random arc selection type; one of:
        "uniform", "log_prob", "fast_log_prob".

  Returns:
    A RandArcSelection enum value.

  Raises:
    FstArgError: Unknown random arc selection type.

  This function is not visible to Python users.
  """
  cdef fst.RandArcSelection select_enum
  if not fst.GetRandArcSelection(select, addr(select_enum)):
    raise FstArgError("Unknown random arc selection type: {!r}".format(select))
  return select_enum


cdef fst.ReplaceLabelType _get_replace_label_type(
    const string &replace_label_type, bool epsilon_on_replace) except *:
  """Matches string with the appropriate ReplaceLabelType enum value.

  This function takes a string argument and returns the matching
  ReplaceLabelType enum value passed to the ReplaceOptions constructor.

  Args:
    replace_label_type: A string matching a known replace label type; one of:
        "neither", "input", "output", "both".
    epsilon_on_replace: Should call/return arcs be epsilon arcs?

  Returns:
    A ReplaceLabelType enum value.

  Raises:
    FstArgError: Unknown replace label type.

  This function is not visible to Python users.
  """
  cdef fst.ReplaceLabelType replace_label_type_enum
  if not fst.GetReplaceLabelType(replace_label_type, epsilon_on_replace,
                                 addr(replace_label_type_enum)):
    raise FstArgError("Unknown replace label type: {!r}".format(
                      replace_label_type))
  return replace_label_type_enum


## Weight and helpers.


cdef class Weight(object):

  """
  Weight(weight_type, weight_string)

  FST weight class.

  This class represents an FST weight. When passed as an argument to an FST
  operation, it should have the weight type of the input FST(s) to said
  operation.

  Args:
    weight_type: A string indicating the weight type.
    weight_string: A string indicating the underlying weight.

  Raises:
    FstArgError: Weight type not found.
    FstBadWeightError: Invalid weight.
  """

  def __repr__(self):
    return "<{} Weight {} at 0x{:x}>".format(self.type(), self.to_string(),
                                             id(self))

  def __str__(self):
    return self.to_string()

  # This attempts to convert the string form into a float, raising
  # ValueError when that is not appropriate.

  def __float__(self):
    return float(self.to_string())

  def __init__(self, weight_type, weight):
    self._weight.reset(new fst.WeightClass(tostring(weight_type),
                                           weight_tostring(weight)))
    self._check_weight()

  cdef void _check_weight(self) except *:
    if self.type() == b"none":
      raise FstArgError("Weight type not found")
    if self.to_string() == b"BadNumber":
      raise FstBadWeightError("Invalid weight")

  cpdef Weight copy(self):
    """
    copy(self)

    Returns a copy of the Weight.
    """
    cdef Weight result = Weight.__new__(Weight)
    result._weight.reset(new fst.WeightClass(deref(self._weight)))
    return result

  # To get around the inability to declare cdef class methods, we define the
  # C++ part out-of-class and then call it from within.

  @classmethod
  def Zero(cls, weight_type):
    """
    Weight.Zero(weight_type)

    Constructs semiring zero.
    """
    return _Zero(weight_type)

  @classmethod
  def One(cls, weight_type):
    """
    Weight.One(weight_type)

    Constructs semiring One.
    """
    return _One(weight_type)

  @classmethod
  def NoWeight(cls, weight_type):
    """
    Weight.NoWeight(weight_type)

    Constructs a non-member weight in the semiring.
    """
    return _NoWeight(weight_type)

  def __eq__(Weight w1, Weight w2):
    return fst.Eq(deref(w1._weight), deref(w2._weight))

  def __ne__(Weight w1, Weight w2):
    return not w1 == w2

  cpdef string to_string(self):
    return self._weight.get().ToString()

  cpdef string type(self):
    """type(self)

    Returns a string indicating the weight type.
    """
    return self._weight.get().Type()


cdef Weight _plus(Weight lhs, Weight rhs):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(fst.Plus(deref(lhs._weight),
                                                    deref(rhs._weight))))
  return result


def plus(Weight lhs, Weight rhs):
  """
  plus(lhs, rhs)

  Computes the sum of two Weights in the same semiring.

  This function computes lhs \oplus rhs, raising an exception if lhs and rhs
  are not in the same semiring.

  Args:
     lhs: Left-hand side Weight.
     rhs: Right-hand side Weight.

  Returns:
    A Weight object.

  Raises:
    FstArgError: Weight type not found (or not in same semiring).
    FstBadWeightError: invalid weight.
  """
  cdef Weight result = _plus(lhs, rhs)
  result._check_weight()
  return result


cdef Weight _times(Weight lhs, Weight rhs):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(fst.Times(deref(lhs._weight),
                                                     deref(rhs._weight))))
  return result


def times(Weight lhs, Weight rhs):
  """
  times(lhs, rhs)

  Computes the product of two Weights in the same semiring.

  This function computes lhs \otimes rhs, raising an exception if lhs and rhs
  are not in the same semiring.

  Args:
     lhs: Left-hand side Weight.
     rhs: Right-hand side Weight.

  Returns:
    A Weight object.

  Raises:
    FstArgError: Weight type not found (or not in same semiring).
    FstBadWeightError: Invalid weight.
  """
  cdef Weight result = _times(lhs, rhs)
  result._check_weight()
  return result


cdef Weight _divide(Weight lhs, Weight rhs):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(fst.Divide(deref(lhs._weight),
                                                      deref(rhs._weight))))
  return result


def divide(Weight lhs, Weight rhs):
  """
  divide(lhs, rhs)

  Computes the quotient of two Weights in the same semiring.

  This function computes lhs \oslash rhs, raising an exception if lhs and rhs
  are not in the same semiring. As there is no way to specify whether to use
  left vs. right division, this assumes a commutative semiring in which these
  are equivalent operations.

  Args:
     lhs: Left-hand side Weight.
     rhs: Right-hand side Weight.

  Returns:
    A Weight object.

  Raises:
    FstArgError: Weight type not found (or not in same semiring).
    FstBadWeightError: Invalid weight.
  """
  cdef Weight result = _divide(lhs, rhs)
  result._check_weight()
  return result


cdef Weight _power(Weight w, size_t n):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(fst.Power(deref(w._weight), n)))
  return result


def power(Weight w, size_t n):
  """
  power(lhs, rhs)

  Computes the iterated product of a weight.

  Args:
     w: The weight.
     n: The power.

  Returns:
    A Weight object.

  Raises:
    FstArgError: Weight type not found (or not in same semiring).
    FstBadWeightError: Invalid weight.
  """
  cdef Weight result = _power(w, n)
  result._check_weight()
  return result


cdef fst.WeightClass _get_WeightClass_or_Zero(const string &weight_type,
                                              weight) except *:
  """Converts weight string to a WeightClass.

  This function constructs a WeightClass instance of the desired weight type.
  If the first argument is null, the weight is set to semiring Zero.

  Args:
    weight_type: A string denoting the desired weight type.
    weight: A object indicating the desired weight; if omitted, the weight is
        set to semiring Zero.

  Returns:
    A WeightClass object.

  This function is not visible to Python users.
  """
  cdef fst.WeightClass result
  if weight is None:
    result = fst.WeightClass.Zero(weight_type)
  elif isinstance(weight, Weight):
    result = deref(<fst.WeightClass *> (<Weight> weight)._weight.get())
  else:
    result = fst.WeightClass(weight_type, weight_tostring(weight))
    if result.ToString() == b"BadNumber":
      raise FstBadWeightError(weight_tostring(weight))
  return result


cdef fst.WeightClass _get_WeightClass_or_One(const string &weight_type,
                                             weight) except *:
  """Converts weight string to a WeightClass.

  This function constructs a WeightClass instance of the desired weight type.
  If the first argument is null, the weight is set to semiring One.

  Args:
    weight_type: A string denoting the desired weight type.
    weight: A object indicating the desired weight; if omitted, the weight is
        set to semiring One.

  Returns:
    A WeightClass object.

  This function is not visible to Python users.
  """
  cdef fst.WeightClass result
  if weight is None:
    result = fst.WeightClass.One(weight_type)
  elif isinstance(weight, Weight):
    result = deref(<fst.WeightClass *> (<Weight> weight)._weight.get())
  else:
    result = fst.WeightClass(weight_type, weight_tostring(weight))
    if result.ToString() == b"BadNumber":
      raise FstBadWeightError(weight_tostring(weight))
  return result


cdef Weight _Zero(weight_type):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(fst.WeightClass.Zero(
      tostring(weight_type))))
  if result._weight.get().Type() == b"none":
    raise FstArgError("Weight type not found")
  return result


cdef Weight _One(weight_type):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(
        fst.WeightClass.One(tostring(weight_type))))
  if result._weight.get().Type() == b"none":
    raise FstArgError("Weight type not found")
  return result


cdef Weight _NoWeight(weight_type):
  cdef Weight result = Weight.__new__(Weight)
  result._weight.reset(new fst.WeightClass(
        fst.WeightClass.NoWeight(tostring(weight_type))))
  return result


## _SymbolTable, _MutableSymbolTable, _EncodeMapperSymbolTable, _FstSymbolTable,
##  _MutableFstSymbolTable, SymbolTable, and helpers.
#
# SymbolTable hierarchy:
#
# _SymbolTable: abstract base class; has-a SymbolTable*
# _EncodeMapperSymbolTable(_SymbolTable): constant symbol table returned by
#     EncodeMapper.input_symbols/output_symbols
# _FstSymbolTable(_SymbolTable): constant symbol table returned by
#     _Fst.input_symbols/output_symbols
#
# _MutableSymbolTable(_SymbolTable): abstract base class adding mutation methods
# _MutableFstSymbolTable(_MutableSymbolTable): mutable symbol table returned by
#     _MutableFst.mutable_input_symbols/mutable_output_symbols
# SymbolTable(_MutableSymbolTable): adds constructor


cdef class _SymbolTable(object):

  """
  (No constructor.)

  Base class for the symbol table hierarchy.

  This class is the base class for SymbolTable. It has a "deleted" constructor
  and implementations for the const methods of the wrapped SymbolTable.
  """

  # NB: Do not expose any non-const methods of the wrapped SymbolTable here.
  # Doing so will allow undefined behavior.

  def __init__(self):
    raise FstDeletedConstructorError(
        "Cannot construct {}".format(self.__class__.__name__))

  def __iter__(self):
    return SymbolTableIterator(self)

  cpdef int64 available_key(self):
    """
    available_key(self)

    Returns an integer indicating the next available key index in the table.
    """
    return self._table.AvailableKey()

  cpdef bytes checksum(self):
    """
    checksum(self)

    Returns a bytestring indicating the label-independent MD5 checksum.
    """
    return self._table.CheckSum()

  cpdef SymbolTable copy(self):
    """
    copy(self)

    Returns a mutable copy of the SymbolTable.
    """
    return _init_SymbolTable(self._table.Copy())

  def find(self, key):
    """
    find(self, key)

    Given a symbol or index, finds the other one.

    This method returns the index associated with a symbol key, or the symbol
    associated with a index key.

    Args:
      key: Either a string or an index.

    Returns:
      If the key is a string, the associated index or NO_LABEL if not found; if
          the key is an integer, the associated symbol or an empty string if
          not found.
    """
    try:
      return self._table.FindIndex(tostring(key))
    except FstArgError:
      return self._table.FindSymbol(key)

  cpdef int64 get_nth_key(self, ssize_t pos) except *:
    """
    get_nth_key(self, pos)

    Retrieves the integer index of the n-th key in the table.

    Args:
      pos: The n-th key to retrieve.

    Returns:
      The integer index of the n-th key, or NO_LABEL if not found.
    """
    return self._table.GetNthKey(pos)

  cpdef bytes labeled_checksum(self):
    """
    labeled_checksum(self)

    Returns a bytestring indicating the label-dependent MD5 checksum.
    """
    return self._table.LabeledCheckSum()

  cpdef bool member(self, key):
    """
    member(self, key)

    Given a symbol or index, returns whether it is found in the table.

    This method returns a boolean indicating whether the given symbol or index
    is present in the table. If one intends to perform subsequent lookup, it is
    better to simply call the find method, catching the KeyError.

    Args:
      key: Either a string or an index.

    Returns:
      Whether or not the key is present (as a string or a index) in the table.
    """
    try:
      return self._table.MemberSymbol(tostring(key))
    except FstArgError:
      return self._table.MemberIndex(key)

  def __contains__(self, key):
    return self.member(key)

  cpdef string name(self):
    """
    name(self)

    Returns the symbol table's name.
    """
    return self._table.Name()

  cpdef size_t num_symbols(self):
    """
    num_symbols(self)

    Returns the number of symbols in the symbol table.
    """
    return self._table.NumSymbols()

  cpdef void write(self, filename) except *:
    """
    write(self, filename)

    Serializes symbol table to a file.

    This methods writes the SymbolTable to a file in binary format.

    Args:
      filename: The string location of the output file.

    Raises:
      FstIOError: Write failed.
    """
    if not self._table.Write(tostring(filename)):
      raise FstIOError("Write failed: {!r}".format(filename))

  cpdef void write_text(self, filename) except *:
    """
    write_text(self, filename)

    Writes symbol table to text file.

    This method writes the SymbolTable to a file in human-readable format.

    Args:
      filename: The string location of the output file.

    Raises:
      FstIOError: Write failed.
    """
    if not self._table.WriteText(tostring(filename)):
      raise FstIOError("Write failed: {!r}".format(filename))


cdef class _EncodeMapperSymbolTable(_SymbolTable):

  """
  (No constructor.)

  Immutable SymbolTable class for tables stored in an EncodeMapper.

  This class wraps a library const SymbolTable and exposes const methods of the
  wrapped object. It is only to be returned by method, never constructed
  directly.
  """

  # NB: Do not expose any non-const methods of the wrapped SymbolTable here.
  # Doing so will allow undefined behavior.

  def __repr__(self):
    return "<const EncodeMapper SymbolTable {!r} at 0x{:x}>".format(self.name(),
                                                                    id(self))


cdef class _FstSymbolTable(_SymbolTable):

  """
  (No constructor.)

  Mutable SymbolTable class for tables stored in a mutable FST.

  This class wraps a library SymbolTable and exposes methods of the wrapped
  object. It is only to be returned by method, never constructed directly.
  """

  # NB: Do not expose any non-const methods of the wrapped SymbolTable here.
  # Doing so will allow undefined behavior.

  def __repr__(self):
    return "<const Fst SymbolTable {!r} at 0x{:x}>".format(self.name(),
                                                           id(self))


cdef class _MutableSymbolTable(_SymbolTable):

  """
  (No constructor.)

  Base class for mutable symbol tables.

  This class is the base class for a mutable SymbolTable. It has a "deleted"
  constructor and implementations of all methods of the wrapped SymbolTable.
  """

  cpdef int64 add_symbol(self, symbol, int64 key=kNoSymbol):
    """
    add_symbol(self, symbol, key=NO_SYMBOL)

    Adds a symbol to the table and returns the index.

    This method adds a symbol to the table. The caller can optionally
    specify a non-negative integer index for the key.

    Args:
      symbol: A symbol string.
      key: An index for the symbol; if not specified, the next index will be
          used.

    Returns:
      The integer key of the new symbol.
    """
    cdef string symbol_string = tostring(symbol)
    if key != kNoSymbol:
      return self._table.AddSymbol(symbol_string, key)
    else:
      return self._table.AddSymbol(symbol_string)

  cpdef void add_table(self, _SymbolTable syms):
    """
    add_table(self, syms)

    Adds another SymbolTable to this table.

    This method merges another symbol table into the current table. All key
    values will be offset by the current available key.

    Args:
      syms: A SymbolTable to be merged with the current table.
    """
    self._table.AddTable(deref(syms._table))

  cpdef void set_name(self, new_name) except *:
    self._table.SetName(tostring(new_name))


cdef class _MutableFstSymbolTable(_MutableSymbolTable):
  """
  (No constructor.)

  Mutable SymbolTable assigned to an FST.
  """

  def __repr__(self):
    return "<Fst SymbolTable {!r} at 0x{:x}>".format(self.name(), id(self))


cdef class SymbolTable(_MutableSymbolTable):

  """
  SymbolTable(name="<unspecified>")

  Mutable SymbolTable class.

  This class wraps the library SymbolTable and exposes both const (i.e.,
  access) and non-const (i.e., mutation) methods of wrapped object.

  Unlike other classes in the hierarchy, it has a working constructor and can be
  used to programmatically construct a SymbolTable in memory.

  Args:
    name: An optional string indicating the table's name.
  """

  def __repr__(self):
    return "<SymbolTable {!r} at 0x{:x}>".format(self.name(), id(self))

  def __init__(self, name="<unspecified>"):
    self._table = new fst.SymbolTable(tostring(name))
    self._smart_table.reset(self._table)

  @classmethod
  def read(cls, filename):
    """
    SymbolTable.read(filename)

    Reads symbol table from binary file.

    This class method creates a new SymbolTable from a symbol table binary file.

    Args:
      filename: The string location of the input binary file.

    Returns:
      A new SymbolTable instance.

    See also: `SymbolTable.read_fst`, `SymbolTable.read_text`.
    """
    cdef fst.SymbolTable *tsyms = fst.SymbolTable.Read(tostring(filename))
    if tsyms == NULL:
      raise FstIOError("Read failed: {!r}".format(filename))
    return _init_SymbolTable(tsyms)

  @classmethod
  def read_text(cls, filename, bool allow_negative_labels=False):
    """
    SymbolTable.read_text(filename)

    Reads symbol table from text file.

    This class method creates a new SymbolTable from a symbol table text file.

    Args:
      filename: The string location of the input text file.
      allow_negative_labels: Should negative labels be allowed? (Not
          recommended; may cause conflicts).

    Returns:
      A new SymbolTable instance.

    See also: `SymbolTable.read`, `SymbolTable.read_fst`.
    """
    cdef unique_ptr[fst.SymbolTableTextOptions] opts
    opts.reset(new fst.SymbolTableTextOptions(allow_negative_labels))
    cdef fst.SymbolTable *tsyms = fst.SymbolTable.ReadText(tostring(filename),
                                                           deref(opts))
    if tsyms == NULL:
      raise FstIOError("Read failed: {!r}".format(filename))
    return _init_SymbolTable(tsyms)

  @classmethod
  def read_fst(cls, filename, bool input_table):
    """
    SymbolTable.read_fst(filename, input_table)

    Reads symbol table from an FST file without loading the corresponding FST.

    This class method creates a new SymbolTable by reading either the input or
    output symbol table from an FST file, without loading the corresponding FST.

    Args:
      filename: The string location of the input FST file.
      input_table: Should the input table be read (True) or the output table
          (False)?

    Returns:
      A new SymbolTable instance, or None if none can be read.

    Raises:
      FstIOError: Read failed.

    See also: `SymbolTable.read`, `SymbolTable.read_text`.
    """
    cdef fst.SymbolTable *tsyms = fst.FstReadSymbols(tostring(filename),
                                                     input_table)
    if tsyms == NULL:
      raise FstIOError("Read failed: {!r}".format(filename))
    return _init_SymbolTable(tsyms)


cdef _EncodeMapperSymbolTable _init_EncodeMapperSymbolTable(
    fst.SymbolTable *table, shared_ptr[fst.EncodeMapperClass] encoder):
  cdef _EncodeMapperSymbolTable result = (
      _EncodeMapperSymbolTable.__new__(_EncodeMapperSymbolTable))
  result._table = table
  result._encoder = encoder
  return result


cdef _FstSymbolTable _init_FstSymbolTable(fst.SymbolTable *table,
                                          shared_ptr[fst.FstClass] ifst):
  cdef _FstSymbolTable result = _FstSymbolTable.__new__(_FstSymbolTable)
  result._table = table
  result._fst = ifst
  return result


cdef _MutableFstSymbolTable _init_MutableFstSymbolTable(fst.SymbolTable *table,
    shared_ptr[fst.MutableFstClass] ifst):
  cdef _MutableFstSymbolTable result = (
      _MutableFstSymbolTable.__new__(_MutableFstSymbolTable))
  result._table = table
  result._mfst = ifst
  return result


cdef SymbolTable _init_SymbolTable(fst.SymbolTable *table):
  cdef SymbolTable result = SymbolTable.__new__(SymbolTable)
  result._table = table
  return result


# Constructive SymbolTable operations.


cpdef SymbolTable compact_symbol_table(_SymbolTable syms):
  """
  compact_symbol_table(syms)

  Constructively relabels a SymbolTable to make it a contiguous mapping.

  Args:
    syms: Input SymbolTable.

  Returns:
    A new compacted SymbolTable.
  """
  return _init_SymbolTable(fst.CompactSymbolTable(deref(syms._table)))


cpdef SymbolTable merge_symbol_table(_SymbolTable lhs, _SymbolTable rhs):
  """
  merge_symbol_table(lhs, rhs)

  Merges all symbols from the left table into the right.

  This function creates a new SymbolTable which is the merger of the two input
  symbol Tables. Symbols in the right-hand table that conflict with those in the
  left-hand table will be assigned values from the left-hand table. Thus the
  returned table will never modify symbol assignments from the left-hand side,
  but may do so on the right.

  If the left-hand table is associated with an FST, it may be necessary to
  relabel it using the output table.

  Args:
    lhs: Left-hand side SymbolTable.
    rhs: Left-hand side SymbolTable.

  Returns:
    A new merged SymbolTable.

  See also: `relabel_symbols`.
  """
  return _init_SymbolTable(fst.MergeSymbolTable(deref(lhs._table),
                                                deref(rhs._table), NULL))


## SymbolTableIterator.


cdef class SymbolTableIterator(object):

  """
  SymbolTableIterator(syms)

  This class is used for iterating over a symbol table.
  """

  def __repr__(self):
    return "<SymbolTableIterator at 0x{:x}>".format(id(self))

  def __init__(self, _SymbolTable syms):
    self._siter.reset(new fst.SymbolTableIterator(deref(syms._table)))

  # This just registers this class as a possible iterator.
  def __iter__(self):
    return self

  # Magic method used to get a Pythonic API out of the C++ API.
  def __next__(self):
    if self.done():
      raise StopIteration
    cdef int64 value = self.value()
    cdef string symbol = self.symbol()
    self.next()
    return (value, symbol)

  cpdef bool done(self):
    """
    done(self)

    Indicates whether the iterator is exhausted or not.

    Returns:
      True if the iterator is exhausted, False otherwise.
    """
    return self._siter.get().Done()

  cpdef void next(self):
    """
    next(self)

    Advances the iterator.
    """
    self._siter.get().Next()

  cpdef void reset(self):
    """
    reset(self)

    Resets the iterator to the initial position.
    """
    self._siter.get().Reset()

  cpdef string symbol(self):
    """
    symbol(self)

    Returns the current symbol string.

    This method returns the current symbol string at this point in the table.

    Returns:
      A symbol string.
    """
    return self._siter.get().Symbol()

  cpdef int64 value(self):
    """
    value(self)

    Returns the current integer index of the symbol.

    Returns:
      An integer index.
    """
    return self._siter.get().Value()


## EncodeMapper.


cdef class EncodeMapper(object):

  """
  EncodeMapper(arc_type="standard", encode_labels=False, encode_weights=False)

  Arc encoder class, wrapping EncodeMapperClass.

  This class provides an object which can be used to encode or decode FST arcs.
  This is most useful to convert an FST to an unweighted acceptor, on which
  some FST operations are more efficient, and then decoding the FST afterwards.

  To use an instance of this class to encode or decode a mutable FST, pass it
  as the first argument to the FST instance methods `encode` and `decode`.

  For implementational reasons, it is not currently possible to use an encoder
  on disk to construct this class.

  Args:
    arc_type: A string indicating the arc type.
    encode_labels: Should labels be encoded?
    encode_weights: Should weights be encoded?
  """

  def __repr__(self):
    return "<EncodeMapper at 0x{:x}>".format(id(self))

  def __init__(self,
               arc_type=b"standard",
               bool encode_labels=False,
               bool encode_weights=False):
    cdef uint32 flags = fst.GetEncodeFlags(encode_labels, encode_weights)
    self._encoder.reset(new fst.EncodeMapperClass(tostring(arc_type), flags,
                                                  fst.ENCODE))
    if not self._encoder:
      raise FstOpError("Unknown arc type: {!r}".format(arc_type))

  cpdef string arc_type(self):
    """
    arc_type(self)

    Returns a string indicating the arc type.
    """
    return self._encoder.get().ArcType()

  # Python's equivalent to operator().

  def __call__(self, Arc arc):
    """
    self(state, ilabel, olabel, weight, nextstate)

    Uses the encoder to encode an arc.

    Args:
      ilabel: The integer index of the input label.
      olabel: The integer index of the output label.
      weight: A Weight or weight string indicating the desired final weight; if
        null, it is set to semiring One.
      nextstate: The integer index of the destination state.

    Raises:
      FstOpError: Incompatible or invalid weight.
    """
    return _init_Arc(self._encoder.get().__call__(deref(arc._arc)))

  cpdef uint32 flags(self):
    """
    flags(self)

    Returns the encoder's flags.
    """
    return self._encoder.get().Flags()

  cpdef _EncodeMapperSymbolTable input_symbols(self):
    """
    input_symbols(self)

    Returns the encoder's input symbol table, or None if none is present.
    """
    cdef fst.SymbolTable *syms = const_cast[SymbolTable_ptr](
        self._encoder.get().InputSymbols())
    if syms == NULL:
      return
    return _init_EncodeMapperSymbolTable(syms, self._encoder)

  cpdef _EncodeMapperSymbolTable output_symbols(self):
    """
    output_symbols(self)

    Returns the encoder's output symbol table, or None if none is present.
    """
    cdef fst.SymbolTable *syms = const_cast[SymbolTable_ptr](
        self._encoder.get().OutputSymbols())
    if syms == NULL:
      return
    return _init_EncodeMapperSymbolTable(syms, self._encoder)

  cpdef uint64 properties(self, uint64 mask):
    """
    properties(self, mask)

    Provides property bits.

    This method provides user access to the properties of the encoder.

    Args:
      mask: The property mask to be compared to the encoder's properties.

    Returns:
      A 64-bit bitmask representing the requested properties.
    """
    return self._encoder.get().Properties(mask)

  cpdef void set_input_symbols(self, _SymbolTable syms) except *:
    """
    set_input_symbols(self, syms)

    Sets the encoder's input symbol table.

    Args:
      syms: A SymbolTable.

    See also: `set_output_symbols`.
    """
    self._encoder.get().SetInputSymbols(syms._table)

  cpdef void set_output_symbols(self, _SymbolTable syms) except *:
    """
    set_output_symbols(self, syms)

    Sets the encoder's output symbol table.

    Args:
      syms: A SymbolTable.

    See also: `set_input_symbols`.
    """
    self._encoder.get().SetOutputSymbols(syms._table)

  cpdef string weight_type(self):
    """
    weight_type(self)

    Returns a string indicating the weight type.
    """
    return self._encoder.get().WeightType()


## _Fst, _MutableFst, Fst, and helpers.
#
# Fst hierarchy:
#
# _Fst: base class; has-a FstClass*.
# _MutableFst(_Fst): adds mutable methods.
# Fst(filename): pseudo-constructor.


cdef class _Fst(object):

  """
  (No constructor.)

  Immutable FST class, wrapping FstClass.

  This class is the basic user-facing FST object. It does not itself support any
  mutation operations.
  """

  # IPython notebook magic to produce an SVG of the FST.
  def _repr_svg_(self):
    """IPython notebook magic to produce an SVG of the FST using GraphViz.

    This method produces an SVG of the internal graph. Users wishing to create
    publication-quality graphs should instead use the method `draw`, which
    exposes additional parameters.

    Raises:
      OSError: Cannot locate the `dot` executable.
      subprocess.CalledProcessError: `dot` returned non-zero exit code.

    See also: `draw`, `text`.
    """
    # Quickly throws OSError if the dot executable i s not found.
    proc = subprocess.Popen(["dot", "-Tsvg"],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    cdef stringstream sstrm
    fst.DrawFst(deref(self._fst), self._fst.get().InputSymbols(),
                self._fst.get().OutputSymbols(), NULL,
                self._fst.get().Properties(fst.kAcceptor, True) ==
                fst.kAcceptor,
                b"", 8.5, 11, True, False, 0.4, 0.25, 14, 5, b"g", False,
                addr(sstrm), b"_repr_svg")
    # The stream gets decoded automatically so we have to re-encode it to pass
    # it to the process.
    (sout, serr) = proc.communicate(sstrm.str().encode("utf8"))
    if proc.returncode != 0:  # Just to be explicit.
      raise subprocess.CalledProcessError(proc.returncode, self._DOT_TSVG)
    return sout.decode("utf8")

  def __repr__(self):
    return "<{} Fst at 0x{:x}>".format(self.fst_type(), id(self))

  def __init__(self):
    raise FstDeletedConstructorError(
        "Cannot construct {}".format(self.__class__.__name__))

  def __str__(self):
    return self.text()

  # Registers the class for pickling; must be repeated in any subclass which
  # can't be derived by _init_XFst.

  def __reduce__(self):
    return (_read_from_string, (self.write_to_string(),))

  cpdef string arc_type(self):
    """
    arc_type(self)

    Returns a string indicating the arc type.
    """
    return self._fst.get().ArcType()

  cpdef ArcIterator arcs(self, int64 state):
    """
    arcs(self, state)

    Returns an iterator over arcs leaving the specified state.

    Args:
      state: The source state ID.

    Returns:
      An ArcIterator.

    See also: `mutable_arcs`, `states`.
    """
    return ArcIterator(self, state)

  cpdef _Fst copy(self):
    """
    copy(self)

    Makes a copy of the FST.
    """
    return _init_XFst(new fst.FstClass(deref(self._fst)))

  cpdef void draw(self,
                  filename,
                  _SymbolTable isymbols=None,
                  _SymbolTable osymbols=None,
                  SymbolTable ssymbols=None,
                  bool acceptor=False,
                  title=b"",
                  double width=8.5,
                  double height=11,
                  bool portrait=False,
                  bool vertical=False,
                  double ranksep=0.4,
                  double nodesep=0.25,
                  int32 fontsize=14,
                  int32 precision=5,
                  float_format=b"g",
                  bool show_weight_one=False):
    """
    draw(self, filename, isymbols=None, osymbols=None, ssymbols=None,
         acceptor=False, title="", width=8.5, height=11, portrait=False,
         vertical=False, ranksep=0.4, nodesep=0.25, fontsize=14,
         precision=5, float_format="g", show_weight_one=False):

    Writes out the FST in Graphviz text format.

    This method writes out the FST in the dot graph description language. The
    graph can be rendered using the `dot` executable provided by Graphviz.

    Args:
      filename: The string location of the output dot/Graphviz file.
      isymbols: An optional symbol table used to label input symbols.
      osymbols: An optional symbol table used to label output symbols.
      ssymbols: An optional symbol table used to label states.
      acceptor: Should the figure be rendered in acceptor format if possible?
      title: An optional string indicating the figure title.
      width: The figure width, in inches.
      height: The figure height, in inches.
      portrait: Should the figure be rendered in portrait rather than
          landscape?
      vertical: Should the figure be rendered bottom-to-top rather than
          left-to-right?
      ranksep: The minimum separation separation between ranks, in inches.
      nodesep: The minimum separation between nodes, in inches.
      fontsize: Font size, in points.
      precision: Numeric precision for floats, in number of chars.
      float_format: One of: 'e', 'f' or 'g'.
      show_weight_one: Should weights equivalent to semiring One be printed?

    See also: `text`.
    """
    cdef string filename_string = tostring(filename)
    cdef unique_ptr[ofstream] ostrm
    ostrm.reset(new ofstream(filename_string))
    cdef fst.SymbolTable *ssymbols_ptr = NULL
    if ssymbols is not None:
      ssymbols_ptr = ssymbols._table
    fst.DrawFst(deref(self._fst),
        self._fst.get().InputSymbols() if isymbols is None
        else isymbols._table,
        self._fst.get().OutputSymbols() if osymbols is None
        else osymbols._table,
        ssymbols_ptr, acceptor, tostring(title), width, height, portrait,
        vertical, ranksep, nodesep, fontsize, precision,
        tostring(float_format), show_weight_one, ostrm.get(),
        filename_string)

  cpdef Weight final(self, int64 state):
    """
    final(self, state)

    Returns the final weight of a state.

    Args:
      state: The integer index of a state.

    Returns:
      The final Weight of that state.

    Raises:
      FstIndexError: State index out of range.
    """
    cdef Weight weight = Weight.__new__(Weight)
    weight._weight.reset(new fst.WeightClass(self._fst.get().Final(state)))
    return weight

  cpdef string fst_type(self):
    """
    fst_type(self)

    Returns a string indicating the FST type.
    """
    return self._fst.get().FstType()

  cpdef _FstSymbolTable input_symbols(self):
    """
    input_symbols(self)

    Returns the FST's input symbol table, or None if none is present.

    See also: `input_symbols`.
    """
    cdef fst.SymbolTable *syms = const_cast[SymbolTable_ptr](
      self._fst.get().InputSymbols())
    if syms == NULL:
      return
    return _init_FstSymbolTable(syms, self._fst)

  cpdef size_t num_arcs(self, int64 state) except *:
    """
    num_arcs(self, state)

    Returns the number of arcs leaving a state.

    Args:
      state: The integer index of a state.

    Returns:
      The number of arcs leaving that state.

    Raises:
      FstIndexError: State index out of range.

    See also: `num_states`.
    """
    cdef size_t result = self._fst.get().NumArcs(state)
    if result == SIZE_MAX:
      raise FstIndexError("State index out of range")
    return result

  cpdef size_t num_input_epsilons(self, int64 state) except *:
    """
    num_input_epsilons(self, state)

    Returns the number of arcs with epsilon input labels leaving a state.

    Args:
      state: The integer index of a state.

    Returns:
      The number of epsilon-input-labeled arcs leaving that state.

    Raises:
      FstIndexError: State index out of range.

    See also: `num_output_epsilons`.
    """
    cdef size_t result = self._fst.get().NumInputEpsilons(state)
    if result == SIZE_MAX:
      raise FstIndexError("State index out of range")
    return result

  cpdef size_t num_output_epsilons(self, int64 state) except *:
    """
    num_output_epsilons(self, state)

    Returns the number of arcs with epsilon output labels leaving a state.

    Args:
      state: The integer index of a state.

    Returns:
      The number of epsilon-output-labeled arcs leaving that state.

    Raises:
      FstIndexError: State index out of range.

    See also: `num_input_epsilons`.
    """
    cdef size_t result = self._fst.get().NumOutputEpsilons(state)
    if result == SIZE_MAX:
      raise FstIndexError("State index out of range")
    return result

  cpdef _FstSymbolTable output_symbols(self):
    """
    output_symbols(self)

    Returns the FST's output symbol table, or None if none is present.

    See also: `input_symbols`.
    """
    cdef fst.SymbolTable *syms = const_cast[SymbolTable_ptr](
      self._fst.get().OutputSymbols())
    if syms == NULL:
      return
    return _init_FstSymbolTable(syms, self._fst)

  cpdef uint64 properties(self, uint64 mask, bool test):
    """
    properties(self, mask, test)

    Provides property bits.

    This method provides user access to the properties attributes for the FST.
    The resulting value is a long integer, but when it is cast to a boolean,
    it represents whether or not the FST has the `mask` property.

    Args:
      mask: The property mask to be compared to the FST's properties.
      test: Should any unknown values be computed before comparing against
          the mask?

    Returns:
      A 64-bit bitmask representing the requested properties.
    """
    return self._fst.get().Properties(mask, test)

  cpdef int64 start(self):
    """
    start(self)

    Returns the start state.
    """
    return self._fst.get().Start()

  cpdef StateIterator states(self):
    """
    states(self)

    Returns an iterator over all states in the FST.

    Returns:
      A StateIterator object for the FST.

    See also: `arcs`, `mutable_arcs`.
    """
    return StateIterator(self)

  cpdef string text(self, _SymbolTable isymbols=None,
      _SymbolTable osymbols=None, _SymbolTable ssymbols=None,
      bool acceptor=False, bool show_weight_one=False, missing_sym=b""):
    """
    text(self, isymbols=None, osymbols=None, ssymbols=None, acceptor=False,
         show_weight_one=False, missing_sym="")

    Produces a human-readable string representation of the FST.

    This method generates a human-readable string representation of the FST.
    The caller may optionally specify SymbolTables used to label input labels,
    output labels, or state labels, respectively.

    Args:
      isymbols: An optional symbol table used to label input symbols.
      osymbols: An optional symbol table used to label output symbols.
      ssymbols: An optional symbol table used to label states.
      acceptor: Should the FST be rendered in acceptor format if possible?
      show_weight_one: Should weights equivalent to semiring One be printed?
      missing_symbol: The string to be printed when symbol table lookup fails.

    Returns:
      A formatted string representing the machine.
    """
    # Prints FST to stringstream, then returns resulting string.
    cdef fst.SymbolTable *ssymbols_ptr = NULL
    if ssymbols is not None:
      ssymbols_ptr = ssymbols._table
    cdef stringstream sstrm
    fst.PrintFst(deref(self._fst), sstrm, "<pywrapfst>",
        self._fst.get().InputSymbols() if isymbols is None
        else isymbols._table,
        self._fst.get().OutputSymbols() if osymbols is None
        else osymbols._table,
        ssymbols_ptr, acceptor, show_weight_one, tostring(missing_sym))
    return sstrm.str()

  cpdef bool verify(self):
    """
    verify(self)

    Verifies that an FST's contents are sane.

    Returns:
      True if the contents are sane, False otherwise.
    """
    return fst.Verify(deref(self._fst))

  cpdef string weight_type(self):
    """
    weight_type(self)

    Provides the FST's weight type.

    Returns:
      A string representing the weight type.
    """
    return self._fst.get().WeightType()

  cpdef void write(self, filename) except *:
    """
    write(self, filename)

    Serializes FST to a file.

    This method writes the FST to a file in a binary format.

    Args:
      filename: The string location of the output file.

    Raises:
      FstIOError: Write failed.
    """
    if not self._fst.get().Write(tostring(filename)):
      raise FstIOError("Write failed: {!r}".format(filename))

  cpdef bytes write_to_string(self):
    """
    write_to_string(self)

    Serializes FST to a string.

    Returns:
      A string.

    Raises:
      FstIOError: Write to string failed.

    See also: `read_from_string`.
    """
    cdef stringstream sstrm
    if not self._fst.get().Write(sstrm, "write_to_string"):
      raise FstIOError("Write to string failed")
    return sstrm.str()


cdef class _MutableFst(_Fst):

  """
  (No constructor.)

  Mutable FST class, wrapping MutableFstClass.

  This class extends _Fst by adding mutation operations.
  """

  cdef void _check_mutating_imethod(self) except *:
    """Checks whether an operation mutating the FST has produced an error.

    This function is not visible to Python users.
    """
    if self._fst.get().Properties(fst.kError, True) == fst.kError:
      raise FstOpError("Operation failed")

  cdef void _add_arc(self, int64 state, Arc arc) except *:
    if not self._fst.get().ValidStateId(state):
      raise FstIndexError("State index out of range")
    if not self._mfst.get().AddArc(state, deref(arc._arc)):
      raise FstOpError("Incompatible or invalid weight type")
    self._check_mutating_imethod()

  def add_arc(self, int64 state, Arc arc):
    """
    add_arc(self, state, arc)

    Adds a new arc to the FST and return self.

    Args:
      state: The integer index of the source state.
      arc: The arc to add.

    Returns:
      self.

    Raises:
      FstIndexError: State index out of range.
      FstOpdexError: Incompatible or invalid weight type.

    See also: `add_state`.
    """
    self._add_arc(state, arc)
    return self

  cpdef int64 add_state(self) except *:
    """
    add_state(self)

    Adds a new state to the FST and returns the state ID.

    Returns:
      The integer index of the new state.

    See also: `add_arc`, `set_start`, `set_final`.
    """
    cdef int64 result = self._mfst.get().AddState()
    self._check_mutating_imethod()
    return result

  cdef void _arcsort(self, sort_type=b"ilabel") except *:
    cdef fst.ArcSortType sort_type_enum
    if not fst.GetArcSortType(tostring(sort_type), addr(sort_type_enum)):
      raise FstArgError("Unknown sort type {!r}".format(sort_type))
    fst.ArcSort(self._mfst.get(), sort_type_enum)
    self._check_mutating_imethod()

  def arcsort(self, sort_type=b"ilabel"):
    """
    arcsort(self, sort_type="ilabel")

    Sorts arcs leaving each state of the FST.

    This operation destructively sorts arcs leaving each state using either
    input or output labels.

    Args:
      sort_type: Either "ilabel" (sort arcs according to input labels) or
          "olabel" (sort arcs according to output labels).

    Returns:
      self.

    Raises:
      FstArgError: Unknown sort type.

    See also: `topsort`.
    """
    self._arcsort(sort_type)
    return self

  cdef void _closure(self, bool closure_plus=False) except *:
    fst.Closure(self._mfst.get(), fst.GetClosureType(closure_plus))
    self._check_mutating_imethod()

  def closure(self, bool closure_plus=False):
    """
    closure(self, closure_plus=False)

    Computes concatenative closure.

    This operation destructively converts the FST to its concatenative closure.
    If A transduces string x to y with weight a, then the closure transduces x
    to y with weight a, xx to yy with weight a \otimes a, xxx to yyy with weight
    a \otimes a \otimes a, and so on. The empty string is also transduced to
    itself with semiring One if `closure_plus` is False.

    Args:
      closure_plus: If False, do not accept the empty string.

    Returns:
      self.
    """
    self._closure(closure_plus)
    return self

  cdef void _concat(self, _Fst ifst) except *:
    fst.Concat(self._mfst.get(), deref(ifst._fst))
    self._check_mutating_imethod()

  def concat(self, _Fst ifst):
    """
    concat(self, ifst)

    Computes the concatenation (product) of two FSTs.

    This operation destructively concatenates the FST with a second FST. If A
    transduces string x to y with weight a and B transduces string w to v with
    weight b, then their concatenation transduces string xw to yv with weight a
    \otimes b.

    Args:
      ifst: The second input FST.

    Returns:
      self.
    """
    self._concat(ifst)
    return self

  cdef void _connect(self) except *:
    fst.Connect(self._mfst.get())
    self._check_mutating_imethod()

  def connect(self):
    """
    connect(self)

    Removes unsuccessful paths.

    This operation destructively trims the FST, removing states and arcs that
    are not part of any successful path.

    Returns:
      self.
    """
    self._connect()
    return self

  cdef void _decode(self, EncodeMapper encoder) except *:
    fst.Decode(self._mfst.get(), deref(encoder._encoder))
    self._check_mutating_imethod()

  def decode(self, EncodeMapper encoder):
    """
    decode(self, encoder)

    Decodes encoded labels and/or weights.

    This operation reverses the encoding performed by `encode`.

    Args:
      encoder: An EncodeMapper object used to encode the FST.

    Returns:
      self.

    See also: `encode`.
    """
    self._decode(encoder)
    return self

  cdef void _delete_arcs(self, int64 state, size_t n=0) except *:
    if not (self._mfst.get().DeleteArcs(state, n) if n else
            self._mfst.get().DeleteArcs(state)):
      raise FstIndexError("State index out of range")
    self._check_mutating_imethod()

  def delete_arcs(self, int64 state, size_t n=0):
    """
    delete_arcs(self, state, n=0)

    Deletes arcs leaving a particular state.

    Args:
      state: The integer index of a state.
      n: An optional argument indicating how many arcs to be deleted. If this
          argument is omitted or passed as zero, all arcs from this state are
          deleted.

    Returns:
      self.

    Raises:
      FstIndexError: State index out of range.

    See also: `delete_states`.
    """
    self._delete_arcs(state, n)
    return self

  cdef void _delete_states(self, states=None) except *:
    # Only the former signature has a possible indexing failure.
    if states:
      if not self._mfst.get().DeleteStates(<const vector[int64]> states):
        raise FstIndexError("State index out of range")
    else:
      self._mfst.get().DeleteStates()
    self._check_mutating_imethod()

  def delete_states(self, states=None):
    """
    delete_states(self, states=None)

    Deletes states.

    Args:
      states: An optional iterable of integer indices of the states to be
          deleted. If this argument is omitted, all states are deleted.

    Returns:
      self.

    Raises:
      FstIndexError: State index out of range.

    See also: `delete_arcs`.
    """
    self._delete_states(states)
    return self

  cdef void _encode(self, EncodeMapper encoder) except *:
    fst.Encode(self._mfst.get(), encoder._encoder.get())
    self._check_mutating_imethod()

  def encode(self, EncodeMapper encoder):
    """
    encode(self, encoder)

    Encodes labels and/or weights.

    This operation allows for the representation of a weighted transducer as a
    weighted acceptor, an unweighted transducer, or an unweighted acceptor by
    considering the pair (input label, output label), the pair (input label,
    weight), or the triple (input label, output label, weight) as a single
    label. Applying this operation mutates the EncodeMapper argument, which
    can then be used to decode.

    Args:
      encoder: An EncodeMapper object to be used as the encoder.

    Returns:
      self.

    See also: `decode`.
    """
    self._encode(encoder)
    return self

  cdef void _invert(self) except *:
    fst.Invert(self._mfst.get())
    self._check_mutating_imethod()

  def invert(self):
    """
    invert(self)

    Inverts the FST's transduction.

    This operation destructively inverts the FST's transduction by exchanging
    input and output labels.

    Returns:
      self.
    """
    self._invert()
    return self

  cdef void _minimize(self, float delta=fst.kShortestDelta,
                      bool allow_nondet=False) except *:
    # This runs in-place when the second argument is null.
    fst.Minimize(self._mfst.get(), NULL, delta, allow_nondet)
    self._check_mutating_imethod()

  def minimize(self, float delta=fst.kShortestDelta, bool allow_nondet=False):
    """
    minimize(self, delta=1e-6, allow_nondet=False)

    Minimizes the FST.

    This operation destructively performs the minimization of deterministic
    weighted automata and transducers. If the input FST A is an acceptor, this
    operation produces the minimal acceptor B equivalent to A, i.e. the
    acceptor with a minimal number of states that is equivalent to A. If the
    input FST A is a transducer, this operation internally builds an equivalent
    transducer with a minimal number of states. However, this minimality is
    obtained by allowing transition having strings of symbols as output labels,
    this known in the litterature as a real-time transducer. Such transducers
    are not directly supported by the library. This function will convert such
    transducer by expanding each string-labeled transition into a sequence of
    transitions. This will results in the creation of new states, hence losing
    the minimality property.

    Args:
      delta: Comparison/quantization delta.
      allow_nondet: Attempt minimization of non-deterministic FST?

    Returns:
      self.
    """
    self._minimize(delta, allow_nondet)
    return self

  cpdef MutableArcIterator mutable_arcs(self, int64 state):
    """
    mutable_arcs(self, state)

    Returns a mutable iterator over arcs leaving the specified state.

    Args:
      state: The source state ID.

    Returns:
      A MutableArcIterator.

    See also: `arcs`, `states`.
    """
    return MutableArcIterator(self, state)

  def mutable_input_symbols(self):
    """
    mutable_input_symbols(self)

    Returns the FST's (mutable) input symbol table, or None if none is present.
    """
    cdef fst.SymbolTable *tst = self._mfst.get().MutableInputSymbols()
    if tst == NULL:
      return
    return _init_MutableFstSymbolTable(tst, self._mfst)

  def mutable_output_symbols(self):
    """
    mutable_output_symbols(self)

    Returns the FST's (mutable) output symbol table, or None if none is present.
    """
    cdef fst.SymbolTable *tst = self._mfst.get().MutableOutputSymbols()
    if tst == NULL:
      return
    return _init_MutableFstSymbolTable(tst, self._mfst)

  cpdef int64 num_states(self):
    """
    num_states(self)

    Returns the number of states.
    """
    return self._mfst.get().NumStates()

  cdef void _project(self, bool project_output=False) except *:
    fst.Project(self._mfst.get(), fst.GetProjectType(project_output))
    self._check_mutating_imethod()

  def project(self, bool project_output=False):
    """
    project(self, project_output=False)

    Converts the FST to an acceptor using input or output labels.

    This operation destructively projects an FST onto its domain or range by
    either copying each arc's input label to its output label (the default) or
    vice versa.

    Args:
      project_output: Should the output labels be projected?

    Returns:
      self.

    See also: `decode`, `encode`, `relabel_pairs`, `relabel_symbols`.
    """
    self._project(project_output)
    return self

  cdef void _prune(self, float delta=fst.kDelta, int64 nstate=fst.kNoStateId,
                   weight=None) except *:
    # Threshold is set to semiring Zero (no pruning) if no weight is specified.
    cdef fst.WeightClass wc = _get_WeightClass_or_Zero(self.weight_type(),
                                                       weight)
    fst.Prune(self._mfst.get(), wc, nstate, delta)
    self._check_mutating_imethod()

  def prune(self,
            float delta=fst.kDelta,
            int64 nstate=fst.kNoStateId,
            weight=None):
    """
    prune(self, delta=0.0009765625, nstate=NO_STATE_ID, weight=None)

    Removes paths with weights below a certain threshold.

    This operation deletes states and arcs in the input FST that do not belong
    to a successful path whose weight is no more (w.r.t the natural semiring
    order) than the threshold t \otimes-times the weight of the shortest path in
    the input FST. Weights must be commutative and have the path property.

    Args:
      delta: Comparison/quantization delta.
      nstate: State number threshold.
      weight: A Weight or weight string indicating the desired weight threshold
          below which paths are pruned; if omitted, no paths are pruned.

    Returns:
      self.

    See also: The constructive variant.
    """
    self._prune(delta, nstate, weight)
    return self

  cdef void _push(self,
                  float delta=fst.kDelta,
                  bool remove_total_weight=False,
                  bool to_final=False) except *:
    fst.Push(self._mfst.get(), fst.GetReweightType(to_final), delta,
             remove_total_weight)
    self._check_mutating_imethod()

  def push(self,
           float delta=fst.kDelta,
           bool remove_total_weight=False,
           bool to_final=False):
    """
    push(self, delta=0.0009765625, remove_total_weight=False, to_final=False)

    Pushes weights towards the initial or final states.

    This operation destructively produces an equivalent transducer by pushing
    the weights towards the initial state or toward the final states. When
    pushing weights towards the initial state, the sum of the weight of the
    outgoing transitions and final weight at any non-initial state is equal to
    one in the resulting machine. When pushing weights towards the final states,
    the sum of the weight of the incoming transitions at any state is equal to
    one. Weights need to be left distributive when pushing towards the initial
    state and right distributive when pushing towards the final states.

    Args:
      delta: Comparison/quantization delta.
      remove_total_weight: If pushing weights, should the total weight be
          removed?
      to_final: Push towards final states?

    Returns:
      self.

    See also: The constructive variant, which also supports label pushing.
    """
    self._push(delta, remove_total_weight, to_final)
    return self

  cdef void _relabel_pairs(self, ipairs=None, opairs=None) except *:
    cdef unique_ptr[vector[fst.LabelPair]] _ipairs
    _ipairs.reset(new vector[fst.LabelPair]())
    cdef unique_ptr[vector[fst.LabelPair]] _opairs
    _opairs.reset(new vector[fst.LabelPair]())
    cdef int64 before
    cdef int64 after
    if ipairs:
      for (before, after) in ipairs:
        _ipairs.get().push_back(fst.LabelPair(before, after))
    if opairs:
      for (before, after) in opairs:
        _opairs.get().push_back(fst.LabelPair(before, after))
    if _ipairs.get().empty() and _opairs.get().empty():
      raise FstArgError("No relabeling pairs specified.")
    fst.Relabel(self._mfst.get(), deref(_ipairs), deref(_opairs))
    self._check_mutating_imethod()

  def relabel_pairs(self, ipairs=None, opairs=None):
    """
    relabel_pairs(self, ipairs=None, opairs=None)

    Replaces input and/or output labels using pairs of labels.

    This operation destructively relabels the input and/or output labels of the
    FST using pairs of the form (old_ID, new_ID); omitted indices are
    identity-mapped.

    Args:
      ipairs: An iterable containing (older index, newer index) integer pairs.
      opairs: An iterable containing (older index, newer index) integer pairs.

    Returns:
      self.

    Raises:
      FstArgError: No relabeling pairs specified.

    See also: `decode`, `encode`, `project`, `relabel_tables`.
    """
    self._relabel_pairs(ipairs, opairs)
    return self

  cdef void _relabel_tables(self,
                            _SymbolTable old_isymbols=None,
                            _SymbolTable new_isymbols=None,
                            unknown_isymbol=b"",
                            bool attach_new_isymbols=True,
                            _SymbolTable old_osymbols=None,
                            _SymbolTable new_osymbols=None,
                            unknown_osymbol=b"",
                            bool attach_new_osymbols=True) except *:
    if new_isymbols is None and new_osymbols is None:
      raise FstArgError("No new SymbolTables specified")
    cdef fst.SymbolTable *new_isymbols_ptr = NULL
    if new_isymbols is not None:
      new_isymbols_ptr = new_isymbols._table
    cdef fst.SymbolTable *new_osymbols_ptr = NULL
    if new_osymbols is not None:
      new_osymbols_ptr = new_osymbols._table
    fst.Relabel(self._mfst.get(),
        self._fst.get().InputSymbols() if old_isymbols is None else
        old_isymbols._table, new_isymbols_ptr, tostring(unknown_isymbol),
        attach_new_isymbols,
        self._fst.get().OutputSymbols() if old_osymbols is None else
        old_osymbols._table, new_osymbols_ptr, tostring(unknown_osymbol),
        attach_new_osymbols)
    self._check_mutating_imethod()

  def relabel_tables(self,
                     _SymbolTable old_isymbols=None,
                     _SymbolTable new_isymbols=None,
                     unknown_isymbol=b"",
                     bool attach_new_isymbols=True,
                     _SymbolTable old_osymbols=None,
                     _SymbolTable new_osymbols=None,
                     unknown_osymbol=b"",
                     bool attach_new_osymbols=True):
    """
    relabel_tables(self, old_isymbols=None, new_isymbols=None,
                   unknown_isymbol="", attach_new_isymbols=True,
                   old_osymbols=None, new_osymbols=None,
                   unknown_osymbol="", attach_new_osymbols=True)

    Replaces input and/or output labels using SymbolTables.

    This operation destructively relabels the input and/or output labels of the
    FST using user-specified symbol tables; omitted symbols are identity-mapped.

    Args:
       old_isymbols: The old SymbolTable for input labels, defaulting to the
          FST's input symbol table.
       new_isymbols: A SymbolTable used to relabel the input labels
       unknown_isymbol: Input symbol to use to relabel OOVs (if empty,
          OOVs raise an exception)
       attach_new_isymbols: Should new_isymbols be made the FST's input symbol
          table?
       old_osymbols: The old SymbolTable for output labels, defaulting to the
          FST's output symbol table.
       new_osymbols: A SymbolTable used to relabel the output labels.
       unknown_osymbol: Outnput symbol to use to relabel OOVs (if empty,
          OOVs raise an exception)
       attach_new_isymbols: Should new_osymbols be made the FST's output symbol
          table?

    Returns:
      self.

    Raises:
      FstArgError: No SymbolTable specified.

    See also: `decode`, `encode`, `project`, `relabel_pairs`.
    """
    self._relabel_tables(old_isymbols, new_isymbols,
                         unknown_isymbol, attach_new_isymbols,
                         old_osymbols, new_osymbols,
                         unknown_osymbol, attach_new_osymbols)
    return self

  cdef void _reserve_arcs(self, int64 state, size_t n) except *:
    if not self._mfst.get().ReserveArcs(state, n):
      raise FstIndexError("State index out of range")
    self._check_mutating_imethod()

  def reserve_arcs(self, int64 state, size_t n):
    """
    reserve_arcs(self, state, n)

    Reserve n arcs at a particular state (best effort).

    Args:
      state: The integer index of a state.
      n: The number of arcs to reserve.

    Returns:
      self.

    Raises:
      FstIndexError: State index out of range.

    See also: `reserve_states`.
    """
    self._reserve_arcs(state, n)
    return self

  cdef void _reserve_states(self, int64 n) except *:
    self._mfst.get().ReserveStates(n)
    self._check_mutating_imethod()

  def reserve_states(self, int64 n):
    """
    reserve_states(self, n)

    Reserve n states (best effort).

    Args:
      n: The number of states to reserve.

    Returns:
      self.

    See also: `reserve_arcs`.
    """
    self._reserve_states(n)
    return self

  cdef void _reweight(self, potentials, bool to_final=False) except *:
    cdef unique_ptr[vector[fst.WeightClass]] _potentials
    _potentials.reset(new vector[fst.WeightClass]())
    cdef string weight_type = self.weight_type()
    for weight in potentials:
        _potentials.get().push_back(_get_WeightClass_or_One(self.weight_type(),
                                                            weight))
    fst.Reweight(self._mfst.get(), deref(_potentials),
                 fst.GetReweightType(to_final))
    self._check_mutating_imethod()

  def reweight(self, potentials, bool to_final=False):
    """
    reweight(self, potentials, to_final=False)

    Reweights an FST using an iterable of potentials.

    This operation destructively reweights an FST according to the potentials
    and in the direction specified by the user. An arc of weight w, with an
    origin state of potential p and destination state of potential q, is
    reweighted by p^{-1} \otimes (w \otimes q) when reweighting towards the
    initial state, and by (p \otimes w) \otimes q^{-1} when reweighting towards
    the final states. The weights must be left distributive when reweighting
    towards the initial state and right distributive when reweighting towards
    the final states (e.g., TropicalWeight and LogWeight).

    Args:
      potentials: An iterable of Weight or weight strings.
      to_final: Push towards final states?

    Returns:
      self.
    """
    self._reweight(potentials, to_final)
    return self

  cdef void _rmepsilon(self,
                       queue_type=b"auto",
                       bool connect=True,
                       weight=None,
                       int64 nstate=fst.kNoStateId,
                       float delta=fst.kShortestDelta) except *:
    cdef fst.WeightClass wc = _get_WeightClass_or_Zero(self.weight_type(),
                                                       weight)
    cdef unique_ptr[fst.RmEpsilonOptions] opts
    opts.reset(new fst.RmEpsilonOptions(_get_queue_type(tostring(queue_type)),
                                        connect, wc, nstate, delta))
    fst.RmEpsilon(self._mfst.get(), deref(opts))
    self._check_mutating_imethod()

  def rmepsilon(self,
                queue_type=b"auto",
                bool connect=True,
                weight=None,
                int64 nstate=fst.kNoStateId,
                float delta=fst.kShortestDelta):
    """
    rmepsilon(self, queue_type="auto", connect=True, weight=None,
              nstate=NO_STATE_ID, delta=1e-6):

    Removes epsilon transitions.

    This operation destructively removes epsilon transitions, i.e., those where
    both input and output labels are epsilon) from an FST.

    Args:
      queue_type: A string matching a known queue type; one of: "auto", "fifo",
          "lifo", "shortest", "state", "top".
      connect: Should output be trimmed?
      weight: A Weight or weight string indicating the desired weight threshold
          below which paths are pruned; if omitted, no paths are pruned.
      nstate: State number threshold.
      delta: Comparison/quantization delta.

    Returns:
      self.
    """
    self._rmepsilon(queue_type, connect, weight, nstate, delta)
    return self

  cdef void _set_final(self, int64 state, weight=None) except *:
    if not self._mfst.get().ValidStateId(state):
      raise FstIndexError("State index out of range")
    cdef fst.WeightClass wc = _get_WeightClass_or_One(self.weight_type(),
                                                      weight)
    if not self._mfst.get().SetFinal(state, wc):
      raise FstOpError("Incompatible or invalid weight")
    self._check_mutating_imethod()

  def set_final(self, int64 state, weight=None):
    """
    set_final(self, state, weight)

    Sets the final weight for a state.

    Args:
      state: The integer index of a state.
      weight: A Weight or weight string indicating the desired final weight; if
          omitted, it is set to semiring One.

    Returns:
      self.

    Raises:
      FstIndexError: State index out of range.
      FstOpError: Incompatible or invalid weight.

    See also: `set_start`.
    """
    self._set_final(state, weight)
    return self

  cdef void _set_input_symbols(self, _SymbolTable syms) except *:
    if syms is None:
      self._mfst.get().SetInputSymbols(NULL)
      return
    self._mfst.get().SetInputSymbols(syms._table)
    self._check_mutating_imethod()

  def set_input_symbols(self, _SymbolTable syms):
    """
    set_input_symbols(self, syms)

    Sets the input symbol table.

    Passing None as a value will delete the input symbol table.

    Args:
      syms: A SymbolTable.

    Returns:
      self.

    See also: `set_output_symbols`.
    """
    self._set_input_symbols(syms)
    return self

  cdef void _set_output_symbols(self, _SymbolTable syms) except *:
    if syms is None:
      self._mfst.get().SetOutputSymbols(NULL)
      return
    self._mfst.get().SetOutputSymbols(syms._table)
    self._check_mutating_imethod()

  def set_output_symbols(self, _SymbolTable syms):
    """
    set_output_symbols(self, syms)

    Sets the output symbol table.

    Passing None as a value will delete the output symbol table.

    Args:
      syms: A SymbolTable.

    Returns:
      self.

    See also: `set_input_symbols`.
    """
    self._set_output_symbols(syms)
    return self

  cdef void _set_properties(self, uint64 props, uint64 mask):
    self._mfst.get().SetProperties(props, mask)

  def set_properties(self, uint64 props, uint64 mask):
    """
    set_properties(self, props, mask)

    Sets the properties bits.

    Args:
      props: The properties to be set.
      mask: A mask to be applied to the `props` argument before setting the
          FST's properties.

    Returns:
      self.
    """
    self._set_properties(props, mask)
    return self

  cdef void _set_start(self, int64 state) except *:
    if not self._mfst.get().SetStart(state):
      raise FstIndexError("State index out of range")
    self._check_mutating_imethod()

  def set_start(self, int64 state):
    """
    set_start(self, state)

    Sets a state to be the initial state state.

    Args:
      state: The integer index of a state.

    Returns:
      self.

    Raises:
      FstIndexError: State index out of range.

    See also: `set_final`.
    """
    self._set_start(state)
    return self

  cdef void _topsort(self) except *:
    # TopSort returns False if the FST is cyclic, and thus can't be TopSorted.
    if not fst.TopSort(self._mfst.get()):
      logging.warning("Cannot topsort cyclic FST.")
    self._check_mutating_imethod()

  def topsort(self):
    """
    topsort(self)

    Sorts transitions by state IDs.

    This operation destructively topologically sorts the FST, if it is acyclic;
    otherwise it remains unchanged. Once sorted, all transitions are from lower
    state IDs to higher state IDs

    Returns:
       self.

    See also: `arcsort`.
    """
    self._topsort()
    return self

  cdef void _union(self, _Fst ifst) except *:
    fst.Union(self._mfst.get(), deref(ifst._fst))
    self._check_mutating_imethod()

  def union(self, _Fst ifst):
    """
    union(self, ifst)

    Computes the union (sum) of two FSTs.

    This operation computes the union (sum) of two FSTs. If A transduces string
    x to y with weight a and B transduces string w to v with weight b, then
    their union transduces x to y with weight a and w to v with weight b.

    Args:
      ifst: The second input FST.

    Returns:
      self.
    """
    self._union(ifst)
    return self


# Pseudo-constructors for _Fst and _MutableFst.
#
# _init_Fst and _init_MutableFst use an FstClass pointer to instantiate _Fst
# and _MutableFst objects, respectively. The latter function is only safe to
# call when the FST being wrapped is known to be kMutable. The caller can
# safely use it when they have either checked this bit (e.g., by using
# _init_XFst) or have themselves constructed a mutable container for the
# FstClass pointer they're passing (e.g., most of the constructive operations,
# storing their results in a VectorFstClass, a derivative of MutableFstClass).
#
# _create_Fst constructs an empty VectorFstClass of a user-specified arc type,
# and passes this pointer to _init_MutableFst.
#
# _read_Fst reads an FST from disk, performing FST conversion if requested, and
# then passes this pointer to _init_XFst.
#
# The Python class Fst provides a wrapper for these two operations. The former
# can be accessed by calling Fst(...), which acts like a class method, and the
# latter via Fst.read(...), which acts like a static method. This is a bit
# nasty, but totally hidden from the Python user.


cdef _Fst _init_Fst(FstClass_ptr tfst):
  if tfst.Properties(fst.kError, True):
    raise FstOpError("Operation failed")
  cdef _Fst ofst = _Fst.__new__(_Fst)
  ofst._fst.reset(tfst)
  return ofst


cdef _MutableFst _init_MutableFst(MutableFstClass_ptr tfst):
  if tfst.Properties(fst.kError, True):
    raise FstOpError("Operation failed")
  cdef _MutableFst ofst = _MutableFst.__new__(_MutableFst)
  ofst._fst.reset(tfst)
  # Makes a copy of it as the derived type! Cool.
  ofst._mfst = static_pointer_cast[fst.MutableFstClass, fst.FstClass](ofst._fst)
  return ofst


cdef _Fst _init_XFst(FstClass_ptr tfst):
  if tfst.Properties(fst.kMutable, True):
    return _init_MutableFst(static_cast[MutableFstClass_ptr](tfst))
  else:
    return _init_Fst(tfst)


cdef _MutableFst _create_Fst(arc_type=b"standard"):
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(tostring(arc_type)))
  if tfst.get() == NULL:
    raise FstOpError("Unknown arc type: {!r}".format(arc_type))
  return _init_MutableFst(tfst.release())


cpdef _Fst _read(filename):
  cdef unique_ptr[fst.FstClass] tfst
  tfst.reset(fst.FstClass.Read(tostring(filename)))
  if tfst.get() == NULL:
    raise FstIOError("Read failed: {!r}".format(filename))
  return _init_XFst(tfst.release())


cpdef _Fst _read_from_string(state):
  cdef stringstream sstrm
  sstrm << tostring(state)
  cdef unique_ptr[fst.FstClass] tfst
  tfst.reset(fst.FstClass.ReadFromStream(sstrm, "<pywrapfst>"))
  if tfst.get() == NULL:
    raise FstIOError("Read failed: <string>")
  return _init_XFst(tfst.release())


class Fst(object):

   """
   Fst(arc_type="standard")

   Constructs an empty FST.

   Args:
     arc_type: A string indicating the arc type.

   Raises:
     FstError: Unknown arc type.

   Raises:
     FstOpError: operation failed.
   """

   def __new__(cls, arc_type=b"standard"):
    return _create_Fst(arc_type)

   @staticmethod
   def read(filename):
     """
     read(filename):

     Reads an FST from a file.

     Args:
       filename: The string location of the input file.

     Returns:
       An FST object.

     Raises:
       FstIOError: Read failed.
     """
     return _read(filename)

   @staticmethod
   def read_from_string(state):
     """
     read_from_string(string, fst_type=None)

     Reads an FST from a serialized string.

     Args:
       state: A string containing the serialized FST.

     Returns:
       An FST object.

     Raises:
       FstIOError: Read failed.
       FstOpError: Read-time conversion failed.

     See also: `write_to_string`.
     """
     return _read_from_string(state)


## FST constants.


NO_LABEL = fst.kNoLabel
NO_STATE_ID = fst.kNoStateId
# TODO(kbg): Figure out how to access static class variables so I don't have
# to do it this way.
NO_SYMBOL = kNoSymbol


## FST properties.


EXPANDED = fst.kExpanded
MUTABLE = fst.kMutable
ERROR = fst.kError
ACCEPTOR = fst.kAcceptor
NOT_ACCEPTOR = fst.kNotAcceptor
I_DETERMINISTIC = fst.kIDeterministic
NON_I_DETERMINISTIC = fst.kNonIDeterministic
O_DETERMINISTIC = fst.kODeterministic
NON_O_DETERMINISTIC = fst.kNonODeterministic
EPSILONS = fst.kEpsilons
NO_EPSILONS = fst.kNoEpsilons
I_EPSILONS = fst.kIEpsilons
NO_I_EPSILONS = fst.kNoIEpsilons
O_EPSILONS = fst.kOEpsilons
NO_O_EPSILONS = fst.kNoOEpsilons
I_LABEL_SORTED = fst.kILabelSorted
NOT_I_LABEL_SORTED = fst.kNotILabelSorted
O_LABEL_SORTED = fst.kOLabelSorted
NOT_O_LABEL_SORTED = fst.kNotOLabelSorted
WEIGHTED = fst.kWeighted
UNWEIGHTED = fst.kUnweighted
CYCLIC = fst.kCyclic
ACYCLIC = fst.kAcyclic
INITIAL_CYCLIC = fst.kInitialCyclic
INITIAL_ACYCLIC = fst.kInitialAcyclic
TOP_SORTED = fst.kTopSorted
NOT_TOP_SORTED = fst.kNotTopSorted
ACCESSIBLE = fst.kAccessible
NOT_ACCESSIBLE = fst.kNotAccessible
COACCESSIBLE = fst.kCoAccessible
NOT_COACCESSIBLE = fst.kNotCoAccessible
STRING = fst.kString
NOT_STRING = fst.kNotString
WEIGHTED_CYCLES = fst.kWeightedCycles
UNWEIGHTED_CYCLES = fst.kUnweightedCycles
NULL_PROPERTIES = fst.kNullProperties
COPY_PROPERTIES = fst.kCopyProperties
INTRINSIC_PROPERTIES = fst.kIntrinsicProperties
EXTRINSIC_PROPERTIES = fst.kExtrinsicProperties
SET_START_PROPERTIES = fst.kSetStartProperties
SET_FINAL_PROPERTIES = fst.kSetFinalProperties
ADD_STATE_PROPERTIES = fst.kAddStateProperties
ADD_ARC_PROPERTIES = fst.kAddArcProperties
SET_ARC_PROPERTIES = fst.kSetArcProperties
DELETE_STATE_PROPERTIES = fst.kDeleteStatesProperties
DELETE_ARC_PROPERTIES = fst.kDeleteArcsProperties
STATE_SORT_PROPERTIES = fst.kStateSortProperties
ARC_SORT_PROPERTIES = fst.kArcSortProperties
I_LABEL_INVARIANT_PROPERTIES = fst.kILabelInvariantProperties
O_LABEL_INVARIANT_PROPERTIES = fst.kOLabelInvariantProperties
WEIGHT_INVARIANT_PROPERTIES = fst.kWeightInvariantProperties
ADD_SUPERFINAL_PROPERTIES = fst.kAddSuperFinalProperties
RM_SUPERFINAL_PROPERTIES = fst.kRmSuperFinalProperties
BINARY_PROPERTIES = fst.kBinaryProperties
TRINARY_PROPERTIES = fst.kTrinaryProperties
POS_TRINARY_PROPERTIES = fst.kPosTrinaryProperties
NEG_TRINARY_PROPERTIES = fst.kNegTrinaryProperties
FST_PROPERTIES = fst.kFstProperties


## Arc iterator properties.


ARC_I_LABEL_VALUE = fst.kArcILabelValue
ARC_O_LABEL_VALUE = fst.kArcOLabelValue
ARC_WEIGHT_VALUE = fst.kArcWeightValue
ARC_NEXT_STATE_VALUE = fst.kArcNextStateValue
ARC_NO_CACHE = fst.kArcNoCache
ARC_VALUE_FLAGS = fst.kArcValueFlags
ARC_FLAGS = fst.kArcFlags


## EncodeMapper properties.


ENCODE_LABELS = fst.kEncodeLabels
ENCODE_WEIGHTS = fst.kEncodeWeights
ENCODE_FLAGS = fst.kEncodeFlags


## Arc, ArcIterator, and MutableArcIterator.


cdef class Arc(object):

  """
  Arc(ilabel, olabel, weight, nextstate)

  This class represents an arc while remaining agnostic about the underlying arc
  type.  Attributes of the arc can be accessed or mutated, and the arc can be
  copied.

  Attributes:
    ilabel: The input label.
    olabel: The output label.
    weight: The arc weight.
    nextstate: The destination state for the arc.
  """

  def __repr__(self):
    return "<Arc at 0x{:x}>".format(id(self))

  def __init__(self, int64 ilabel, int64 olabel, weight, int64 nextstate):
    cdef fst.WeightClass wc = _get_WeightClass_or_One(b"tropical", weight)
    self._arc.reset(new fst.ArcClass(ilabel, olabel, wc, nextstate))

  cpdef Arc copy(self):
    return Arc(self.ilabel, self.olabel, self.weight, self.nextstate)

  property ilabel:

    def __get__(self):
      return deref(self._arc).ilabel

    def __set__(self, int64 value):
      deref(self._arc).ilabel = value

  property olabel:

    def __get__(self):
      return deref(self._arc).olabel

    def __set__(self, int64 value):
      deref(self._arc).olabel = value

  property weight:

    def __get__(self):
      cdef Weight weight = Weight.__new__(Weight)
      weight._weight.reset(new fst.WeightClass(deref(self._arc).weight))
      return weight

    def __set__(self, weight):
      deref(self._arc).weight = _get_WeightClass_or_One(b"tropical", weight)

  property nextstate:

    def __get__(self):
      return deref(self._arc).nextstate

    def __set__(self, int64 value):
      deref(self._arc).nextstate = value


cdef Arc _init_Arc(const fst.ArcClass &arc):
  cdef Weight weight = Weight.__new__(Weight)
  weight._weight.reset(new fst.WeightClass(arc.weight))
  return Arc(arc.ilabel, arc.olabel, weight, arc.nextstate)


cdef class ArcIterator(object):

  """
  ArcIterator(ifst, state)

  This class is used for iterating over the arcs leaving some state of an FST.
  """

  def __repr__(self):
    return "<ArcIterator at 0x{:x}>".format(id(self))

  def __init__(self, _Fst ifst, int64 state):
    if not ifst._fst.get().ValidStateId(state):
      raise FstIndexError("State index out of range")
    # Makes copy of the shared_ptr, potentially extending the FST's lifetime.
    self._fst = ifst._fst
    self._aiter.reset(new fst.ArcIteratorClass(deref(self._fst), state))

  # This just registers this class as a possible iterator.
  def __iter__(self):
    return self

  # Magic method used to get a Pythonic API out of the C++ API.
  def __next__(self):
    if self.done():
      raise StopIteration
    result = self.value()
    self.next()
    return result

  cpdef bool done(self):
    """
    done(self)

    Indicates whether the iterator is exhausted or not.

    Returns:
      True if the iterator is exhausted, False otherwise.
    """
    return self._aiter.get().Done()

  cpdef uint32 flags(self):
    """
    flags(self)

    Returns the current iterator behavioral flags.

    Returns:
      The current iterator behavioral flags as an integer.
    """
    return self._aiter.get().Flags()

  cpdef void next(self):
    """
    next(self)

    Advances the iterator.
    """
    self._aiter.get().Next()

  cpdef size_t position(self):
    """
    position(self)

    Returns the position of the iterator.

    Returns:
      The iterator's position, expressed as an integer.
    """
    return self._aiter.get().Position()

  cpdef void reset(self):
    """
    reset(self)

    Resets the iterator to the initial position.
    """
    self._aiter.get().Reset()

  cpdef void seek(self, size_t a):
    """
    seek(self, a)

    Advance the iterator to a new position.

    Args:
      a: The position to seek to.
    """
    self._aiter.get().Seek(a)

  cpdef void set_flags(self, uint32 flags, uint32 mask):
    """
    set_flags(self, flags, mask)

    Sets the current iterator behavioral flags.

    Args:
      flags: The properties to be set.
      mask: A mask to be applied to the `flags` argument before setting them.
    """
    self._aiter.get().SetFlags(flags, mask)

  cpdef object value(self):
    """
    value(self)

    Returns the current arc.
    """
    return _init_Arc(self._aiter.get().Value())


cdef class MutableArcIterator(object):

  """
  MutableArcIterator(ifst, state)

  This class is used for iterating over the arcs leaving some state of an FST,
  also permitting mutation of the current arc.
  """

  def __repr__(self):
    return "<MutableArcIterator at 0x{:x}>".format(id(self))

  def __init__(self, _MutableFst ifst, int64 state):
    if not ifst._fst.get().ValidStateId(state):
      raise FstIndexError("State index out of range")
    # Makes copy of the shared_ptr, potentially extending the FST's lifetime.
    self._mfst = ifst._mfst
    self._aiter.reset(new fst.MutableArcIteratorClass(ifst._mfst.get(), state))

  cpdef bool done(self):
    """
    done(self)

    Indicates whether the iterator is exhausted or not.

    Returns:
      True if the iterator is exhausted, False otherwise.
    """
    return self._aiter.get().Done()

  cpdef uint32 flags(self):
    """
    flags(self)

    Returns the current iterator behavioral flags.

    Returns:
      The current iterator behavioral flags as an integer.
    """
    return self._aiter.get().Flags()

  cpdef void next(self):
    """
    next(self)

    Advances the iterator.
    """
    self._aiter.get().Next()

  cpdef size_t position(self):
    """
    position(self)

    Returns the position of the iterator.

    Returns:
      The iterator's position, expressed as an integer.
    """
    return self._aiter.get().Position()

  cpdef void reset(self):
    """
    reset(self)

    Resets the iterator to the initial position.
    """
    self._aiter.get().Reset()

  cpdef void seek(self, size_t a):
    """
    seek(self, a)

    Advance the iterator to a new position.

    Args:
      a: The position to seek to.
    """
    self._aiter.get().Seek(a)

  cpdef void set_flags(self, uint32 flags, uint32 mask):
    """
    set_flags(self, flags, mask)

    Sets the current iterator behavioral flags.

    Args:
      flags: The properties to be set.
      mask: A mask to be applied to the `flags` argument before setting them.
    """
    self._aiter.get().SetFlags(flags, mask)

  cpdef void set_value(self, Arc arc):
    """
    set_value(self, arc)

    Replace the current arc with a new arc.

    Args:
      arc: The arc to replace the current arc with.
    """
    self._aiter.get().SetValue(deref(arc._arc))

  cpdef object value(self):
    """
    value(self)

    Returns the current arc.
    """
    return _init_Arc(self._aiter.get().Value())


## StateIterator.


cdef class StateIterator(object):

  """
  StateIterator(ifst)

  This class is used for iterating over the states in an FST.
  """

  def __repr__(self):
    return "<StateIterator at 0x{:x}>".format(id(self))

  def __init__(self, _Fst ifst):
    # Makes copy of the shared_ptr, potentially extending the FST's lifetime.
    self._fst = ifst._fst
    self._siter.reset(new fst.StateIteratorClass(deref(self._fst)))

  # This just registers this class as a possible iterator.
  def __iter__(self):
    return self

  # Magic method used to get a Pythonic API out of the C++ API.
  def __next__(self):
    if self.done():
      raise StopIteration
    cdef int64 result = self.value()
    self.next()
    return result

  cpdef bool done(self):
    """
    done(self)

    Indicates whether the iterator is exhausted or not.

    Returns:
      True if the iterator is exhausted, False otherwise.
    """
    return self._siter.get().Done()

  cpdef void next(self):
    """
    next(self)

    Advances the iterator.
    """
    self._siter.get().Next()

  cpdef void reset(self):
    """
    reset(self)

    Resets the iterator to the initial position.
    """
    self._siter.get().Reset()

  cpdef int64 value(self):
    """
    value(self)

    Returns the current state index.
    """
    return self._siter.get().Value()


## FST operations.


cdef _Fst _map(_Fst ifst,
               float delta=fst.kDelta,
               map_type=b"identity",
               double power=1.,
               weight=None):
  cdef fst.MapType map_type_enum
  if not fst.GetMapType(tostring(map_type), addr(map_type_enum)):
    raise FstArgError("Unknown map type: {!r}".format(map_type))
  cdef fst.WeightClass wc = (_get_WeightClass_or_One(ifst.weight_type(),
      weight) if map_type_enum == fst.TIMES_MAPPER else
      _get_WeightClass_or_Zero(ifst.weight_type(), weight))
  return _init_XFst(fst.Map(deref(ifst._fst), map_type_enum, delta, power, wc))


cpdef _Fst arcmap(_Fst ifst,
                  float delta=fst.kDelta,
                  map_type=b"identity",
                  double power=1.,
                  weight=None):
  """
  arcmap(ifst, delta=0.0009765625, map_type="identity", weight=None)

  Constructively applies a transform to all arcs and final states.

  This operation transforms each arc and final state in the input FST using
  one of the following:

    * identity: maps to self.
    * input_epsilon: replaces all input labels with epsilon.
    * invert: reciprocates all non-Zero weights.
    * float_power: raises all weights to a floating-point power.
    * output_epsilon: replaces all output labels with epsilon.
    * quantize: quantizes weights.
    * plus: adds a constant to all weights.
    * power: raises all weights to an integral power.
    * rmweight: replaces all non-Zero weights with 1.
    * superfinal: redirects final states to a new superfinal state.
    * times: right-multiplies a constant to all weights.
    * to_log: converts weights to the log semiring.
    * to_log64: converts weights to the log64 semiring.
    * to_standard: converts weights to the tropical ("standard") semiring.

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta (ignored unless `map_type` is
        `quantize`).
    map_type: A string matching a known mapping operation (see above).
    power: A positive scalar or integer power; ignored unless `map_type` is
        `float_power` or `power` (in which case it defaults to 1).
    weight: A Weight or weight string passed to the arc-mapper; ignored unless
        `map_type` is `plus` (in which case it defaults to semiring Zero) or
        `times` (in which case it defaults to semiring One).

  Returns:
    An FST with arcs and final states remapped.

  Raises:
    FstArgError: Unknown map type.

  See also: `statemap`.
  """
  return _map(ifst, delta, map_type, power, weight)


cpdef _MutableFst compose(_Fst ifst1,
                          _Fst ifst2,
                          compose_filter=b"auto",
                          bool connect=True):
  """
  compose(ifst1, ifst2, compose_filter="auto", connect=True)

  Constructively composes two FSTs.

  This operation computes the composition of two FSTs. If A transduces string
  x to y with weight a and B transduces y to z with weight b, then their
  composition transduces string x to z with weight a \otimes b. The output
  labels of the first transducer or the input labels of the second transducer
  must be sorted (or otherwise support appropriate matchers).

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    compose_filter: A string matching a known composition filter; one of:
        "alt_sequence", "auto", "match", "null", "sequence", "trivial".
    connect: Should output be trimmed?

  Returns:
    An FST.

  See also: `arcsort`.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst1.arc_type()))
  cdef unique_ptr[fst.ComposeOptions] opts
  opts.reset(new fst.ComposeOptions(connect,
      _get_compose_filter(tostring(compose_filter))))
  fst.Compose(deref(ifst1._fst), deref(ifst2._fst), tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _Fst convert(_Fst ifst, fst_type=b""):
  """
  convert(ifst, fst_type="")

  Constructively converts an FST to a new internal representation.

  Args:
    ifst: The input FST.
    fst_type: A string indicating the FST type to convert to, or an empty string
        if no conversion is desired.

  Returns:
    The input FST converted to the desired FST type.

  Raises:
    FstOpError: Conversion failed.
  """
  cdef string fst_type_string = tostring(fst_type)
  cdef unique_ptr[fst.FstClass] tfst
  tfst.reset(fst.Convert(deref(ifst._fst), fst_type_string))
  # Script-land Convert returns a null pointer to signal failure.
  if tfst.get() == NULL:
    raise FstOpError("Conversion to {!r} failed".format(fst_type))
  return _init_XFst(tfst.release())


cpdef _MutableFst determinize(_Fst ifst,
                              float delta=fst.kShortestDelta,
                              det_type=b"functional",
                              int64 nstate=fst.kNoStateId,
                              int64 subsequential_label=0,
                              weight=None,
                              bool increment_subsequential_label=False):
  """
  determinize(ifst, delta=1e-6, det_type="functional",
              nstate=NO_STATE_ID, subsequential_label=0, weight=None,
              incremental_subsequential_label=False)

  Constructively determinizes a weighted FST.

  This operations creates an equivalent FST that has the property that no
  state has two transitions with the same input label. For this algorithm,
  epsilon transitions are treated as regular symbols (cf. `rmepsilon`).

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta.
    det_type: Type of determinization; one of: "functional" (input transducer is
        functional), "nonfunctional" (input transducer is not functional) and
        disambiguate" (input transducer is not functional but only keep the min
        of ambiguous outputs).
    nstate: State number threshold.
    subsequential_label: Input label of arc corresponding to residual final
        output when producing a subsequential transducer.
    weight: A Weight or weight string indicating the desired weight threshold
        below which paths are pruned; if omitted, no paths are pruned.
    increment_subsequential_label: Increment subsequential when creating
        several arcs for the residual final output at a given state.

  Returns:
    An equivalent deterministic FST.

  Raises:
    FstArgError: Unknown determinization type.

  See also: `disambiguate`, `rmepsilon`.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  # Threshold is set to semiring Zero (no pruning) if weight unspecified.
  cdef fst.WeightClass wc = _get_WeightClass_or_Zero(ifst.weight_type(),
                                                     weight)
  cdef fst.DeterminizeType determinize_type_enum
  if not fst.GetDeterminizeType(tostring(det_type),
                                addr(determinize_type_enum)):
    raise FstArgError("Unknown determinization type: {!r}".format(det_type))
  cdef unique_ptr[fst.DeterminizeOptions] opts
  opts.reset(new fst.DeterminizeOptions(delta, wc, nstate, subsequential_label,
                                        determinize_type_enum,
                                        increment_subsequential_label))
  fst.Determinize(deref(ifst._fst), tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _MutableFst difference(_Fst ifst1,
                             _Fst ifst2,
                             compose_filter=b"auto",
                             bool connect=True):
  """
  difference(ifst1, ifst2, compose_filter="auto", connect=True)

  Constructively computes the difference of two FSTs.

  This operation computes the difference between two FSAs. Only strings that are
  in the first automaton but not in second are retained in the result. The first
  argument must be an acceptor; the second argument must be an unweighted,
  epsilon-free, deterministic acceptor. The output labels of the first
  transducer or the input labels of the second transducer must be sorted (or
  otherwise support appropriate matchers).

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    compose_filter: A string matching a known composition filter; one of:
        "alt_sequence", "auto", "match", "null", "sequence", "trivial".
    connect: Should the output FST be trimmed?

  Returns:
    An FST representing the difference of the FSTs.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst1.arc_type()))
  cdef unique_ptr[fst.ComposeOptions] opts
  opts.reset(new fst.ComposeOptions(connect, _get_compose_filter(
      tostring(compose_filter))))
  fst.Difference(deref(ifst1._fst), deref(ifst2._fst), tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _MutableFst disambiguate(_Fst ifst,
                               float delta=fst.kDelta,
                               int64 nstate=fst.kNoStateId,
                               int64 subsequential_label=0,
                               weight=None):
  """
  disambiguate(ifst, delta=0.0009765625, nstate=NO_STATE_ID,
               subsequential_label=0, weight=None):

  Constructively disambiguates a weighted transducer.

  This operation disambiguates a weighted transducer. The result will be an
  equivalent FST that has the property that no two successful paths have the
  same input labeling. For this algorithm, epsilon transitions are treated as
  regular symbols (cf. `rmepsilon`).

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta.
    nstate: State number threshold.
    subsequential_label: Input label of arc corresponding to residual final
        output when producing a subsequential transducer.
    weight: A Weight or weight string indicating the desired weight threshold
        below which paths are pruned; if omitted, no paths are pruned.

  Returns:
    An equivalent disambiguated FST.

  See also: `determinize`, `rmepsilon`.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  # Threshold is set to semiring Zero (no pruning) if no weight is specified.
  cdef fst.WeightClass wc = _get_WeightClass_or_Zero(ifst.weight_type(),
                                                     weight)
  cdef unique_ptr[fst.DisambiguateOptions] opts
  opts.reset(new fst.DisambiguateOptions(delta, wc, nstate,
                                         subsequential_label))
  fst.Disambiguate(deref(ifst._fst), tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _MutableFst epsnormalize(_Fst ifst, bool eps_norm_output=False):
  """
  epsnormalize(ifst, eps_norm_output=False)

  Constructively epsilon-normalizes an FST.

  This operation creates an equivalent FST that is epsilon-normalized. An
  acceptor is epsilon-normalized if it it is epsilon-removed (cf. `rmepsilon`).
  A transducer is input epsilon-normalized if, in addition, along any path, all
  arcs with epsilon input labels follow all arcs with non-epsilon input labels.
  Output epsilon-normalized is defined similarly. The input FST must be
  functional.

  Args:
    ifst: The input FST.
    eps_norm_output: Should the FST be output epsilon-normalized?

  Returns:
    An equivalent epsilon-normalized FST.

  See also: `rmepsilon`.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  fst.EpsNormalize(deref(ifst._fst), tfst.get(), fst.EPS_NORM_OUTPUT if
                                                 eps_norm_output else
                                                 fst.EPS_NORM_INPUT)
  return _init_MutableFst(tfst.release())


cpdef bool equal(_Fst ifst1, _Fst ifst2, float delta=fst.kDelta):
  """
  equal(ifst1, ifst2, delta=0.0009765625)

  Are two FSTs equal?

  This function tests whether two FSTs have the same states with the same
  numbering and the same transitions with the same labels and weights in the
  same order.

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    delta: Comparison/quantization delta.

  Returns:
    True if the FSTs satisfy the above condition, else False.

  See also: `equivalent`, `isomorphic`, `randequivalent`.
  """
  return fst.Equal(deref(ifst1._fst), deref(ifst2._fst), delta)


cpdef bool equivalent(_Fst ifst1, _Fst ifst2, float delta=fst.kDelta) except *:
  """
  equivalent(ifst1, ifst2, delta=0.0009765625)

  Are the two acceptors equivalent?

  This operation tests whether two epsilon-free deterministic weighted
  acceptors are equivalent, that is if they accept the same strings with the
  same weights.

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    delta: Comparison/quantization delta.

  Returns:
    True if the FSTs satisfy the above condition, else False.

  See also: `equal`, `isomorphic`, `randequivalent`.
  """
  return fst.Equivalent(deref(ifst1._fst), deref(ifst2._fst), delta)


cpdef _MutableFst intersect(_Fst ifst1,
                            _Fst ifst2,
                            compose_filter=b"auto",
                            bool connect=True):
  """
  intersect(ifst1, ifst2, compose_filter="auto", connect=True)

  Constructively intersects two FSTs.

  This operation computes the intersection (Hadamard product) of two FSTs.
  Only strings that are in both automata are retained in the result. The two
  arguments must be acceptors. One of the arguments must be label-sorted (or
  otherwise support appropriate matchers).

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    compose_filter: A string matching a known composition filter; one of:
        "alt_sequence", "auto", "match", "null", "sequence", "trivial".
    connect: Should output be trimmed?

  Returns:
    An intersected FST.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst1.arc_type()))
  cdef unique_ptr[fst.ComposeOptions] opts
  opts.reset(new fst.ComposeOptions(connect,
        _get_compose_filter(tostring(compose_filter))))
  fst.Intersect(deref(ifst1._fst), deref(ifst2._fst), tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef bool isomorphic(_Fst ifst1, _Fst ifst2, float delta=fst.kDelta):
  """
  isomorphic(ifst1, ifst2, delta=0.0009765625)

  Are the two acceptors isomorphic?

  This operation determines if two transducers with a certain required
  determinism have the same states, irrespective of numbering, and the same
  transitions with the same labels and weights, irrespective of ordering. In
  other words, FSTs A, B are isomorphic if and only if the states of A can be
  renumbered and the transitions leaving each state reordered so the two are
  equal (according to the definition given in `equal`).

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    delta: Comparison/quantization delta.

  Returns:
    True if the two transducers satisfy the above condition, else False.

  See also: `equal`, `equivalent`, `randequivalent`.
  """
  return fst.Isomorphic(deref(ifst1._fst), deref(ifst2._fst), delta)


cpdef _MutableFst prune(_Fst ifst,
                        float delta=fst.kDelta,
                        int64 nstate=fst.kNoStateId,
                        weight=None):
  """
  prune(ifst, delta=0.0009765625, nstate=NO_STATE_ID, weight=None)

  Constructively removes paths with weights below a certain threshold.

  This operation deletes states and arcs in the input FST that do not belong
  to a successful path whose weight is no more (w.r.t the natural semiring
  order) than the threshold t \otimes-times the weight of the shortest path in
  the input FST. Weights must be commutative and have the path property.

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta.
    nstate: State number threshold.
    weight: A Weight or weight string indicating the desired weight threshold
        below which paths are pruned; if omitted, no paths are pruned.

  Returns:
    A pruned FST.

  See also: The destructive variant.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  cdef fst.WeightClass wc = _get_WeightClass_or_Zero(ifst.weight_type(), weight)
  fst.Prune(deref(ifst._fst), tfst.get(), wc, nstate, delta)
  return _init_MutableFst(tfst.release())


cpdef _MutableFst push(_Fst ifst,
                       float delta=fst.kDelta,
                       bool push_weights=False,
                       bool push_labels=False,
                       bool remove_common_affix=False,
                       bool remove_total_weight=False,
                       bool to_final=False):
  """
  push(ifst, delta=0.0009765625, push_weights=False, push_labels=False,
       remove_common_affix=False, remove_total_weight=False, to_final=False)

  Constructively pushes weights/labels towards initial or final states.

  This operation produces an equivalent transducer by pushing the weights
  and/or the labels towards the initial state or toward the final states.

  When pushing weights towards the initial state, the sum of the weight of the
  outgoing transitions and final weight at any non-initial state is equal to 1
  in the resulting machine. When pushing weights towards the final states, the
  sum of the weight of the incoming transitions at any state is equal to 1.
  Weights need to be left distributive when pushing towards the initial state
  and right distributive when pushing towards the final states.

  Pushing labels towards the initial state consists in minimizing at every
  state the length of the longest common prefix of the output labels of the
  outgoing paths. Pushing labels towards the final states consists in
  minimizing at every state the length of the longest common suffix of the
  output labels of the incoming paths.

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta.
    push_weights: Should weights be pushed?
    push_labels: Should labels be pushed?
    remove_common_affix: If pushing labels, should common prefix/suffix be
        removed?
    remove_total_weight: If pushing weights, should total weight be removed?
    to_final: Push towards final states?

  Returns:
    An equivalent pushed FST.

  See also: The destructive variant.
  """
  # This is copied, almost verbatim, from ./fstpush.cc.
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  cdef uint32 flags = fst.GetPushFlags(push_weights, push_labels,
                                       remove_common_affix, remove_total_weight)
  fst.Push(deref(ifst._fst), tfst.get(), flags, fst.GetReweightType(to_final),
           delta)
  return _init_MutableFst(tfst.release())


cpdef bool randequivalent(_Fst ifst1,
                          _Fst ifst2,
                          int32 npath=1,
                          float delta=fst.kDelta,
                          time_t seed=0,
                          select=b"uniform",
                          int32 max_length=INT32_MAX) except *:
  """
  randequivalent(ifst1, ifst2, npath=1, delta=0.0009765625, seed=0,
                 select="uniform", max_length=2147483647)

  Are two acceptors stochastically equivalent?

  This operation tests whether two FSTs are equivalent by randomly generating
  paths alternatively in each of the two FSTs. For each randomly generated path,
  the algorithm computes for each of the two FSTs the sum of the weights of all
  the successful paths sharing the same input and output labels as the randomly
  generated path and checks that these two values are within `delta`.

  Args:
    ifst1: The first input FST.
    ifst2: The second input FST.
    npath: The number of random paths to generate.
    delta: Comparison/quantization delta.
    seed: An optional seed value for random path generation; if zero, the
        current time and process ID is used.
    select: A string matching a known random arc selection type; one of:
        "uniform", "log_prob", "fast_log_prob".
    max_length: The maximum length of each random path.

  Returns:
    True if the two transducers satisfy the above condition, else False.

  See also: `equal`, `equivalent`, `isomorphic`, `randgen`.
  """
  cdef fst.RandArcSelection ras = _get_rand_arc_selection(tostring(select))
  cdef unique_ptr[fst.RandGenOptions[fst.RandArcSelection]] opts
  # The three trailing options will be ignored by RandEquivalent.
  opts.reset(new fst.RandGenOptions[fst.RandArcSelection](ras, max_length,
                                                          1, False, False))
  if seed == 0:
    seed = time(NULL) + getpid()
  return fst.RandEquivalent(deref(ifst1._fst), deref(ifst2._fst), npath, delta,
                           seed, deref(opts))


cpdef _MutableFst randgen(_Fst ifst,
                          int32 npath=1,
                          time_t seed=0,
                          select=b"uniform",
                          int32 max_length=INT32_MAX,
                          bool weighted=False,
                          bool remove_total_weight=False):
  """
  randgen(ifst, npath=1, seed=0, select="uniform", max_length=2147483647,
          weight=False, remove_total_weight=False)

  Randomly generate successful paths in an FST.

  This operation randomly generates a set of successful paths in the input FST.
  This relies on a mechanism for selecting arcs, specified using the `select`
  argument. The default selector, "uniform", randomly selects a transition
  using a uniform distribution. The "log_prob" selector randomly selects a
  transition w.r.t. the weights treated as negative log probabilities after
  normalizing for the total weight leaving the state. In all cases, finality is
  treated as a transition to a super-final state.

  Args:
    ifst: The input FST.
    npath: The number of random paths to generate.
    seed: An optional seed value for random path generation; if zero, the
        current time and process ID is used.
    select: A string matching a known random arc selection type; one of:
        "uniform", "log_prob", "fast_log_prob".
    max_length: The maximum length of each random path.
    weighted: Should the output be weighted by path count?
    remove_total_weight: Should the total weight be removed (ignored when
        `weighted` is False)?

  Returns:
    An FST containing one or more random paths.

  See also: `randequivalent`.
  """
  cdef fst.RandArcSelection ras = _get_rand_arc_selection(tostring(select))
  cdef unique_ptr[fst.RandGenOptions[fst.RandArcSelection]] opts
  opts.reset(new fst.RandGenOptions[fst.RandArcSelection](ras, max_length,
                                                          npath, weighted,
                                                          remove_total_weight))
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  if seed == 0:
    seed = time(NULL) + getpid()
  fst.RandGen(deref(ifst._fst), tfst.get(), seed, deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _MutableFst replace(pairs,
                          call_arc_labeling=b"input",
                          return_arc_labeling=b"neither",
                          bool epsilon_on_replace=False,
                          int64 return_label=0):
  """
  replace(pairs, call_arc_labeling="input", return_arc_labeling="neither",
          epsilon_on_replace=False, return_label=0)

  Recursively replaces arcs in the FST with other FST(s).

  This operation performs the dynamic replacement of arcs in one FST with
  another FST, allowing the definition of FSTs analogous to RTNs. It takes as
  input a set of pairs of a set of pairs formed by a non-terminal label and
  its corresponding FST, and a label identifying the root FST in that set.
  The resulting FST is obtained by taking the root FST and recursively replacing
  each arc having a nonterminal as output label by its corresponding FST. More
  precisely, an arc from state s to state d with (nonterminal) output label n in
  this FST is replaced by redirecting this "call" arc to the initial state of a
  copy F of the FST for n, and adding "return" arcs from each final state of F
  to d. Optional arguments control how the call and return arcs are labeled; by
  default, the only non-epsilon label is placed on the call arc.

  Args:

    pairs: An iterable of (nonterminal label, FST) pairs, where the former is an
        unsigned integer and the latter is an Fst instance.
    call_arc_labeling: A string indicating which call arc labels should be
        non-epsilon. One of: "input" (default), "output", "both", "neither".
        This value is set to "neither" if epsilon_on_replace is True.
    return_arc_labeling: A string indicating which return arc labels should be
        non-epsilon. One of: "input", "output", "both", "neither" (default).
        This value is set to "neither" if epsilon_on_replace is True.
    epsilon_on_replace: Should call and return arcs be epsilon arcs? If True,
        this effectively overrides call_arc_labeling and return_arc_labeling,
        setting both to "neither".
    return_label: The integer label for return arcs.

  Returns:
    An FST resulting from expanding the input RTN.
  """
  cdef vector[fst.LabelFstClassPair] _pairs
  cdef int64 root_label
  cdef int64 label
  cdef _Fst ifst
  it = iter(pairs)
  (root_label, ifst) = next(it)
  _pairs.push_back(fst.LabelFstClassPair(root_label, ifst._fst.get()))
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  for (label, ifst) in it:
    _pairs.push_back(fst.LabelFstClassPair(label, ifst._fst.get()))
  cdef fst.ReplaceLabelType cal = _get_replace_label_type(
      tostring(call_arc_labeling), epsilon_on_replace)
  cdef fst.ReplaceLabelType ral = _get_replace_label_type(
      tostring(return_arc_labeling), epsilon_on_replace)
  cdef unique_ptr[fst.ReplaceOptions] opts
  opts.reset(new fst.ReplaceOptions(root_label, cal, ral, return_label))
  fst.Replace(_pairs, tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _MutableFst reverse(_Fst ifst, bool require_superinitial=True):
  """
  reverse(ifst, require_superinitial=True)

  Constructively reverses an FST's transduction.

  This operation reverses an FST. If A transduces string x to y with weight a,
  then the reverse of A transduces the reverse of x to the reverse of y with
  weight a.Reverse(). (Typically, a = a.Reverse() and Arc = RevArc, e.g.,
  TropicalWeight and LogWeight.) In general, e.g., when the weights only form a
  left or right semiring, the output arc type must match the input arc type.

  Args:
    ifst: The input FST.
    require_superinitial: Should a superinitial state be created?

  Returns:
    A reversed FST.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  fst.Reverse(deref(ifst._fst), tfst.get(), require_superinitial)
  return _init_MutableFst(tfst.release())


# Pure C++ helper for shortestdistance.


cdef vector[fst.WeightClass] *_shortestdistance(_Fst ifst,
                                                float delta=fst.kShortestDelta,
                                                int64 nstate=fst.kNoStateId,
                                                queue_type=b"auto",
                                                bool reverse=False) except *:
  cdef unique_ptr[vector[fst.WeightClass]] distance
  distance.reset(new vector[fst.WeightClass]())
  # For scoping reasons, these have to be declared here even though they may
  # not be used in all cases.
  cdef unique_ptr[fst.ShortestDistanceOptions] opts
  if reverse:
    # Only the simpler signature supports shortest distance to final states;
    # `nstate` and `queue_type` arguments are ignored.
    fst.ShortestDistance(deref(ifst._fst), distance.get(), True, delta)
  else:
    opts.reset(new fst.ShortestDistanceOptions(
        _get_queue_type(tostring(queue_type)), fst.ANY_ARC_FILTER, nstate,
        delta))
    fst.ShortestDistance(deref(ifst._fst), distance.get(), deref(opts))
  return distance.release()


def shortestdistance(_Fst ifst,
                     float delta=fst.kShortestDelta,
                     int64 nstate=fst.kNoStateId,
                     queue_type=b"auto",
                     bool reverse=False):
  """
  shortestdistance(ifst, delta=1e-6, nstate=NO_STATE_ID,
                   queue_type="auto", reverse=False)

  Compute the shortest distance from the initial or final state.

  This operation computes the shortest distance from the initial state (when
  `reverse` is False) or from every state to the final state (when `reverse` is
  True). The shortest distance from p to q is the \otimes-sum of the weights of
  all the paths between p and q. The weights must be right (if `reverse` is
  False) or left (if `reverse` is True) distributive, and k-closed (i.e., 1
  \otimes x \otimes x^2 \otimes ... \otimes x^{k + 1} = 1 \otimes x \otimes x^2
  \otimes ... \otimes x^k; e.g., TropicalWeight).

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta.
    nstate: State number threshold (ignored if `reverse` is True).
    queue_type: A string matching a known queue type; one of: "auto", "fifo",
        "lifo", "shortest", "state", "top" (ignored if `reverse` is True).
    reverse: Should the reverse distance (from each state to the final state)
        be computed?

  Returns:
    A list of Weight objects representing the shortest distance for each state.
  """
  cdef unique_ptr[vector[fst.WeightClass]] distance
  distance.reset(_shortestdistance(ifst, delta, nstate, queue_type, reverse))
  cdef string weight_type = ifst.weight_type()
  return [Weight(weight_type, weight.ToString()) for weight in deref(distance)]


cpdef _MutableFst shortestpath(_Fst ifst,
                               float delta=fst.kShortestDelta,
                               int32 nshortest=1,
                               int64 nstate=fst.kNoStateId,
                               queue_type=b"auto",
                               bool unique=False,
                               weight=None):
  """
  shortestpath(ifst, delta=1e-6, nshortest=1, nstate=NO_STATE_ID,
               queue_type="auto", unique=False, weight=None)

  Construct an FST containing the shortest path(s) in the input FST.

  This operation produces an FST containing the n-shortest paths in the input
  FST. The n-shortest paths are the n-lowest weight paths w.r.t. the natural
  semiring order. The single path that can be read from the ith of at most n
  transitions leaving the initial state of the resulting FST is the ith
  shortest path. The weights need to be right distributive and have the path
  property. They also need to be left distributive as well for n-shortest with
  n > 1 (e.g., TropicalWeight).

  Args:
    ifst: The input FST.
    delta: Comparison/quantization delta.
    nshortest: The number of paths to return.
    nstate: State number threshold.
    queue_type: A string matching a known queue type; one of: "auto", "fifo",
        "lifo", "shortest", "state", "top".
    unique: Should the resulting FST only contain distinct paths? (Requires
        the input FST to be an acceptor; epsilons are treated as if they are
        regular symbols.)
    weight: A Weight or weight string indicating the desired weight threshold
        below which paths are pruned; if omitted, no paths are pruned.

  Returns:
    An FST containing the n-shortest paths.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  # Threshold is set to semiring Zero (no pruning) if no weight is specified.
  cdef fst.WeightClass wc = _get_WeightClass_or_Zero(ifst.weight_type(), weight)
  cdef unique_ptr[fst.ShortestPathOptions] opts
  opts.reset(new fst.ShortestPathOptions(_get_queue_type(tostring(queue_type)),
                                         nshortest, unique, delta, wc, nstate))
  fst.ShortestPath(deref(ifst._fst), tfst.get(), deref(opts))
  return _init_MutableFst(tfst.release())


cpdef _Fst statemap(_Fst ifst, map_type):
  """
  state_map(ifst, map_type)

  Constructively applies a transform to all states.

  This operation transforms each state according to the requested map type.
  Note that currently, only one state-mapping operation is supported.

  Args:
    ifst: The input FST.
    map_type: A string matching a known mapping operation; one of: "arc_sum"
        (sum weights of identically-labeled multi-arcs), "arc_unique" (deletes
        non-unique identically-labeled multi-arcs).

  Returns:
    An FST with states remapped.

  Raises:
    FstArgError: Unknown map type.

  See also: `arcmap`.
  """
  return _map(ifst, fst.kDelta, map_type, 1., None)


cpdef _MutableFst synchronize(_Fst ifst):
  """
  synchronize(ifst)

  Constructively synchronizes an FST.

  This operation synchronizes a transducer. The result will be an equivalent
  FST that has the property that during the traversal of a path, the delay is
  either zero or strictly increasing, where the delay is the difference between
  the number of non-epsilon output labels and input labels along the path. For
  the algorithm to terminate, the input transducer must have bounded delay,
  i.e., the delay of every cycle must be zero.

  Args:
    ifst: The input FST.

  Returns:
    An equivalent synchronized FST.
  """
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst.arc_type()))
  fst.Synchronize(deref(ifst._fst), tfst.get())
  return _init_MutableFst(tfst.release())


## Compiler.


cdef class Compiler(object):

  """
  Compiler(fst_type="vector", arc_type="standard", isymbols=None,
           osymbols=None, ssymbols=None, acceptor=False, keep_isymbols=False,
           keep_osymbols=False, keep_state_numbering=False,
           allow_negative_labels=False)

  Class used to compile FSTs from strings.

  This class is used to compile FSTs specified using the AT&T FSM library
  format described here:

  http://web.eecs.umich.edu/~radev/NLP-fall2015/resources/fsm_archive/fsm.5.html

  This is the same format used by the `fstcompile` executable.

  Compiler options (symbol tables, etc.) are set at construction time.

      compiler = fst.Compiler(isymbols=ascii_syms, osymbols=ascii_syms)

  Once constructed, Compiler instances behave like a file handle opened for
  writing:

      # /ba+/
      compiler.write("0 1 50 50")
      compiler.write("1 2 49 49")
      compiler.write("2 2 49 49")
      compiler.write("2")

  The `compile` method returns an actual FST instance:

      sheep_machine = compiler.compile()

  Compilation flushes the internal buffer, so the compiler instance can be
  reused to compile new machines with the same symbol tables (etc.)

  Args:
    fst_type: A string indicating the container type for the compiled FST.
    arc_type: A string indicating the arc type for the compiled FST.
    isymbols: An optional SymbolTable used to label input symbols.
    osymbols: An optional SymbolTable used to label output symbols.
    ssymbols: An optional SymbolTable used to label states.
    acceptor: Should the FST be rendered in acceptor format if possible?
    keep_isymbols: Should the input symbol table be stored in the FST?
    keep_osymbols: Should the output symbol table be stored in the FST?
    keep_state_numbering: Should the state numbering be preserved?
    allow_negative_labels: Should negative labels be allowed? (Not
        recommended; may cause conflicts).
  """

  def __cinit__(self,
                string fst_type=b"vector",
                string arc_type=b"standard",
                SymbolTable isymbols=None,
                SymbolTable osymbols=None,
                SymbolTable ssymbols=None,
                bool acceptor=False,
                bool keep_isymbols=False,
                bool keep_osymbols=False,
                bool keep_state_numbering=False,
                bool allow_negative_labels=False):
    self._sstrm.reset(new stringstream())
    self._fst_type = tostring(fst_type)
    self._arc_type = tostring(arc_type)
    self._isymbols = NULL
    if isymbols is not None:
      self._isymbols = isymbols._table
    self._osymbols = NULL
    if osymbols is not None:
      self._osymbols = osymbols._table
    self._ssymbols = NULL
    if ssymbols is not None:
      self._ssymbols = ssymbols._table
    self._acceptor = acceptor
    self._keep_isymbols = keep_isymbols
    self._keep_osymbols = keep_osymbols
    self._keep_state_numbering = keep_state_numbering
    self._allow_negative_labels = allow_negative_labels

  cpdef _Fst compile(self):
    """
    compile()

    Compiles the FST in the compiler string buffer.

    This method compiles the FST and returns the resulting machine.

    Returns:
      The FST described by the compiler string buffer.

    Raises:
      FstOpError: Compilation failed.
    """
    cdef unique_ptr[fst.FstClass] tfst
    tfst.reset(fst.CompileFstInternal(deref(self._sstrm),
        "<pywrapfst>", self._fst_type, self._arc_type, self._isymbols,
        self._osymbols, self._ssymbols, self._acceptor, self._keep_isymbols,
        self._keep_osymbols, self._keep_state_numbering,
        self._allow_negative_labels))
    self._sstrm.reset(new stringstream())
    if tfst.get() == NULL:
      raise FstOpError("Compilation failed")
    return _init_XFst(tfst.release())

  cpdef void write(self, expression):
    """
    write(expression)

    Writes a string into the compiler string buffer.

    This method adds a line to the compiler string buffer. It is normally
    invoked using the right shift operator, like so:

        compiler = fst.Compiler()
        compiler.write("0 0 49 49")
        compiler.write("0")

    Args:
      expression: A string expression to add to compiler string buffer.
    """
    deref(self._sstrm) << tostring(expression)


## FarReader and FarWriter.


cdef class FarReader(object):

  """
  (No constructor.)

  FAR ("Fst ARchive") reader object.

  This class is used to read a FAR from disk. FARs contain one or more FSTs (of
  the same arc type) indexed by a unique string key. To construct a FarReader
  object, use the `open` class method.

  Attributes:
    arc_type: A string indicating the arc type.
    far_type: A string indicating the FAR type.
  """

  def __init__(self):
    raise FstDeletedConstructorError(
        "Cannot construct {}".format(self.__class__.__name__))

  def __repr__(self):
    return "<{} FarReader at 0x{:x}>".format(self.far_type(), id(self))

  @classmethod
  def open(cls, *filenames):
    """
    FarReader.open(*filenames)

    Creates a FarReader object.

    This class method creates a FarReader given the string location of one or
    more FAR files on disk.

    Args:
      *filenames: The string location of one or more input FAR files.

    Returns:
      A new FarReader instance.

    Raises:
      FstIOError: Read failed.
    """
    cdef vector[string] filename_strings
    for filename in filenames:
      filename_strings.push_back(tostring(filename))
    cdef unique_ptr[fst.FarReaderClass] tfar
    tfar.reset(fst.FarReaderClass.Open(filename_strings))
    if tfar.get() == NULL:
      raise FstIOError("Read failed: {!r}".format(filenames))
    cdef FarReader result = FarReader.__new__(FarReader)
    result._reader.reset(tfar.release())
    return result

  cpdef string arc_type(self):
    """
    arc_type(self)

    Returns a string indicating the arc type.
    """
    return self._reader.get().ArcType()

  cpdef bool done(self):
    """
    done(self)

    Indicates whether the iterator is exhausted or not.

    Returns:
      True if the iterator is exhausted, False otherwise.
    """
    return self._reader.get().Done()

  cpdef bool error(self):
    """
    error(self)

    Indicates whether the FarReader has encountered an error.

    Returns:
      True if the FarReader is in an errorful state, False otherwise.
    """
    return self._reader.get().Error()

  cpdef string far_type(self):
    return fst.GetFarTypeString(self._reader.get().Type())

  cpdef bool find(self, key) except *:
    """
    find(self, key)

    Sets the current position to the first entry greater than or equal to the
    key (a string) and indicates whether or not a match was found.

    Args:
      key: A string key.

    Returns:
      True if the key was found, False otherwise.
    """
    return self._reader.get().Find(tostring(key))

  cpdef _Fst get_fst(self):
    """
    get_fst(self)

    Returns the FST at the current position.

    Returns:
      A copy of the FST at the current position.
    """
    return _init_XFst(new fst.FstClass(
        deref(self._reader.get().GetFstClass())))

  cpdef string get_key(self):
    """
    get_key(self)

    Returns the string key at the current position.

    Returns:
      The string key at the current position.
    """
    return self._reader.get().GetKey()

  cpdef void next(self):
    """
    next(self)

    Advances the iterator.
    """
    self._reader.get().Next()

  cpdef void reset(self):
    """
    reset(self)

    Resets the iterator to the initial position.
    """
    self._reader.get().Reset()

  def __getitem__(self, key):
    if self._reader.get().Find(tostring(key)):
      return self.get_fst()
    else:
      raise KeyError(key)


cdef class FarWriter(object):

  """
  (No constructor.)

  FAR ("Fst ARchive") writer object.

  This class is used to write FSTs (of the same arc type) to a FAR on disk. To
  construct a FarWriter, use the `create` class method.

  Note that the data is not guaranteed to flush to disk until the FarWriter
  is garbage-collected. If a FarWriter has been assigned to only one variable,
  then calling `del` on that variable should decrement the object's reference
  count from 1 to 0, triggering a flush to disk on the next GC cycle.

  Attributes:
    arc_type: A string indicating the arc type.
    far_type: A string indicating the FAR type.
  """

  def __init__(self):
    raise FstDeletedConstructorError(
        "Cannot construct {}".format(self.__class__.__name__))

  def __repr__(self):
    return "<{} FarWriter at 0x{:x}>".format(self.far_type(), id(self))

  @classmethod
  def create(cls, filename, arc_type=b"standard", far_type=b"default"):
    """
    FarWriter.

    Creates a FarWriter object.

    This class method creates a FarWriter given the desired output location,
    arc type, and FAR type.

    Args:
      filename: The string location for the output FAR files.
      arc_type: A string indicating the arc type.
      far_type: A string indicating the FAR type; one of: "fst", "stlist",
          "sttable", "sstable", "default".

    Returns:
      A new FarWriter instance.

    Raises:
      FstIOError: Read failed.
    """
    cdef fst.FarType ft = fst.GetFarType(tostring(far_type))
    cdef fst.FarWriterClass *tfar = fst.FarWriterClass.Create(
        tostring(filename), tostring(arc_type), ft)
    if tfar == NULL:
      raise FstIOError("Open failed: {!r}".format(filename))
    cdef FarWriter result = FarWriter.__new__(FarWriter)
    result._writer.reset(tfar)
    return result

  # NB: Invoking this method may be dangerous: calling any other method on the
  # instance after this is invoked may result in a null dereference.
  cdef void close(self):
    self._writer.reset()

  cpdef void add(self, key, _Fst ifst) except *:
    """
    add(self, key, ifst)

    Adds an FST to the FAR.

    This method adds an FST to the FAR which can be retrieved with the
    specified string key.

    Args:
      key: The string used to key the input FST.
      ifst: The FST to write to the FAR.

    Raises:
      FstArgError: Key out of order.
      FstOpError: Incompatible or invalid arc type.
    """
    # Failure here results from passing an FST with a different arc type than
    # used by the FAR was initialized to use.
    if not self._writer.get().Add(tostring(key), deref(ifst._fst)):
      raise FstOpError("Incompatible or invalid arc type")
    # An error here usually indicates a key out of order.
    if self._writer.get().Error():
      raise FstArgError("Key out of order")

  cpdef string arc_type(self):
    """
    arc_type(self)

    Returns a string indicating the arc type.
    """
    return self._writer.get().ArcType()

  cpdef bool error(self):
    """
    error(self)

    Indicates whether the FarWriter has encountered an error.

    Returns:
      True if the FarWriter is in an errorful state, False otherwise.
    """
    return self._writer.get().Error()

  cpdef string far_type(self):
    """
    far_type(self)

    Returns a string indicating the FAR type.
    """
    return fst.GetFarTypeString(self._writer.get().Type())

  # Dictionary-like assignment.
  def __setitem__(self, key, _Fst fst):
    self.add(key, fst)


## Cleanup operations for module entrance and exit.


# Masks fst_error_fatal flags while this module is running, returning to the
# previous state upon module exit.


cdef bool _fst_error_fatal_old = fst.FLAGS_fst_error_fatal
fst.FLAGS_fst_error_fatal = False


@atexit.register
def _reset_fst_error_fatal():
  fst.FLAGS_fst_error_fatal = _fst_error_fatal_old
