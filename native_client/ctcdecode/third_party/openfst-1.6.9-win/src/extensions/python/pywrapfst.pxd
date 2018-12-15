# See www.openfst.org for extensive documentation on this weighted
# finite-state transducer library.


from libc.time cimport time
from libc.time cimport time_t

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from libcpp.string cimport string
from basictypes cimport int32
from basictypes cimport int64
from basictypes cimport uint32
from basictypes cimport uint64
cimport fst as fst
from ios cimport stringstream


# Exportable helper functions.


cdef string tostring(data, encoding=?) except *

cdef string weight_tostring(data, encoding=?) except *

cdef fst.ComposeFilter _get_compose_filter(
    const string &compose_filter) except *

cdef fst.DeterminizeType _get_determinize_type(const string &det_type) except *

cdef fst.QueueType _get_queue_type(const string &queue_type) except *

cdef fst.RandArcSelection _get_rand_arc_selection(
    const string &replace_label_type) except *

cdef fst.ReplaceLabelType _get_replace_label_type(
    const string &replace_label_type, bool epsilon_on_replace) except *


# Weight.


cdef fst.WeightClass _get_WeightClass_or_One(const string &weight_type,
                                             weight_string) except *

cdef fst.WeightClass _get_WeightClass_or_Zero(const string &weight_type,
                                              weight_string) except *


cdef class Weight(object):

  cdef unique_ptr[fst.WeightClass] _weight

  cdef void _check_weight(self) except *

  cpdef Weight copy(self)

  cpdef string to_string(self)

  cpdef string type(self)


cdef Weight _Zero(weight_type)

cdef Weight _One(weight_type)

cdef Weight _NoWeight(weight_type)

cdef Weight _plus(Weight lhs, Weight rhs)

cdef Weight _times(Weight lhs, Weight rhs)

cdef Weight _divide(Weight lhs, Weight rhs)

cdef Weight _power(Weight lhs, size_t n)


# SymbolTable.

ctypedef fst.SymbolTable * SymbolTable_ptr


cdef class _SymbolTable(object):

  cdef fst.SymbolTable *_table

  cpdef int64 available_key(self)

  cpdef bytes checksum(self)

  cpdef SymbolTable copy(self)

  cpdef int64 get_nth_key(self, ssize_t pos) except *

  cpdef bytes labeled_checksum(self)

  cpdef bool member(self, key)

  cpdef string name(self)

  cpdef size_t num_symbols(self)

  cpdef void write(self, filename) except *

  cpdef void write_text(self, filename) except *


cdef class _EncodeMapperSymbolTable(_SymbolTable):

  cdef shared_ptr[fst.EncodeMapperClass] _encoder


cdef class _FstSymbolTable(_SymbolTable):

  cdef shared_ptr[fst.FstClass] _fst


cdef class _MutableSymbolTable(_SymbolTable):

  cpdef int64 add_symbol(self, symbol, int64 key=?)

  cpdef void add_table(self, _SymbolTable syms)

  cpdef void set_name(self, new_name) except *


cdef class _MutableFstSymbolTable(_MutableSymbolTable):

  cdef shared_ptr[fst.MutableFstClass] _mfst


cdef class SymbolTable(_MutableSymbolTable):

  cdef unique_ptr[fst.SymbolTable] _smart_table


cdef _EncodeMapperSymbolTable _init_EncodeMapperSymbolTable(
    fst.SymbolTable *table, shared_ptr[fst.EncodeMapperClass] encoder)


cdef _FstSymbolTable _init_FstSymbolTable(fst.SymbolTable *table,
                                          shared_ptr[fst.FstClass] ifst)


cdef _MutableFstSymbolTable _init_MutableFstSymbolTable(fst.SymbolTable *table,
    shared_ptr[fst.MutableFstClass] ifst)


cdef SymbolTable _init_SymbolTable(fst.SymbolTable *table)



cdef class SymbolTableIterator(object):

  cdef shared_ptr[fst.SymbolTable] _table
  cdef unique_ptr[fst.SymbolTableIterator] _siter

  cpdef bool done(self)

  cpdef void next(self)

  cpdef void reset(self)

  cpdef string symbol(self)

  cpdef int64 value(self)


# EncodeMapper.


cdef class EncodeMapper(object):

  cdef shared_ptr[fst.EncodeMapperClass] _encoder

  cpdef string arc_type(self)

  cpdef uint32 flags(self)

  cpdef _EncodeMapperSymbolTable input_symbols(self)

  cpdef _EncodeMapperSymbolTable output_symbols(self)

  cpdef uint64 properties(self, uint64 mask)

  cpdef void set_input_symbols(self, _SymbolTable syms) except *

  cpdef void set_output_symbols(self, _SymbolTable syms) except *

  cpdef string weight_type(self)


# Fst.


ctypedef fst.FstClass * FstClass_ptr
ctypedef fst.MutableFstClass * MutableFstClass_ptr
ctypedef fst.VectorFstClass * VectorFstClass_ptr


cdef class _Fst(object):

  cdef shared_ptr[fst.FstClass] _fst

  cpdef string arc_type(self)

  cpdef ArcIterator arcs(self, int64 state)

  cpdef _Fst copy(self)

  cpdef void draw(self, filename, _SymbolTable isymbols=?,
                  _SymbolTable osymbols=?, SymbolTable ssymbols=?,
                  bool acceptor=?, title=?, double width=?,
                  double height=?, bool portrait=?, bool vertical=?,
                  double ranksep=?, double nodesep=?, int32 fontsize=?,
                  int32 precision=?, float_format=?,
                  bool show_weight_one=?)

  cpdef Weight final(self, int64 state)

  cpdef string fst_type(self)

  cpdef _FstSymbolTable input_symbols(self)

  cpdef size_t num_arcs(self, int64 state) except *

  cpdef size_t num_input_epsilons(self, int64 state) except *

  cpdef size_t num_output_epsilons(self, int64 state) except *

  cpdef _FstSymbolTable output_symbols(self)

  cpdef uint64 properties(self, uint64 mask, bool test)

  cpdef int64 start(self)

  cpdef StateIterator states(self)

  cpdef string text(self, _SymbolTable isymbols=?, _SymbolTable osymbols=?,
                    _SymbolTable ssymbols=?, bool acceptor=?,
                    bool show_weight_one=?, missing_sym=?)

  cpdef bool verify(self)

  cpdef string weight_type(self)

  cpdef void write(self, filename) except *

  cpdef bytes write_to_string(self)


cdef class _MutableFst(_Fst):

  cdef shared_ptr[fst.MutableFstClass] _mfst

  cdef void _check_mutating_imethod(self) except *

  cdef void _add_arc(self, int64 state, Arc arc) except *

  cpdef int64 add_state(self) except *

  cdef void _arcsort(self, sort_type=?) except *

  cdef void _closure(self, bool closure_plus=?) except *

  cdef void _concat(self, _Fst ifst) except *

  cdef void _connect(self) except *

  cdef void _decode(self, EncodeMapper) except *

  cdef void _delete_arcs(self, int64 state, size_t n=?) except *

  cdef void _delete_states(self, states=?) except *

  cdef void _encode(self, EncodeMapper) except *

  cdef void _invert(self) except *

  cdef void _minimize(self, float delta=?, bool allow_nondet=?) except *

  cpdef MutableArcIterator mutable_arcs(self, int64 state)

  cpdef int64 num_states(self)

  cdef void _project(self, bool project_output=?) except *

  cdef void _prune(self, float delta=?, int64 nstate=?, weight=?) except *

  cdef void _push(self, float delta=?, bool remove_total_weight=?,
                  bool to_final=?) except *

  cdef void _relabel_pairs(self, ipairs=?, opairs=?) except *

  cdef void _relabel_tables(self, _SymbolTable old_isymbols=?,
      _SymbolTable new_isymbols=?, unknown_isymbol=?,
      bool attach_new_isymbols=?,
      _SymbolTable old_osymbols=?, _SymbolTable new_osymbols=?,
      unknown_osymbol=?, bool attach_new_osymbols=?) except *

  cdef void _reserve_arcs(self, int64 state, size_t n) except *

  cdef void _reserve_states(self, int64 n) except *

  cdef void _reweight(self, potentials, bool to_final=?) except *

  cdef void _rmepsilon(self, queue_type=?, bool connect=?, weight=?,
                       int64 nstate=?, float delta=?) except *

  cdef void _set_final(self, int64 state, weight=?) except *

  cdef void _set_properties(self, uint64 props, uint64 mask)

  cdef void _set_start(self, int64 state) except *

  cdef void _set_input_symbols(self, _SymbolTable syms) except *

  cdef void _set_output_symbols(self, _SymbolTable syms) except *

  cdef void _topsort(self) except *

  cdef void _union(self, _Fst ifst) except *


# Fst construction helpers.


cdef _Fst _init_Fst(FstClass_ptr tfst)

cdef _MutableFst _init_MutableFst(MutableFstClass_ptr tfst)

cdef _Fst _init_XFst(FstClass_ptr tfst)

cdef _MutableFst _create_Fst(arc_type=?)

cpdef _Fst _read(filename)

cpdef _Fst _read_from_string(state)


# Iterators.


cdef class Arc(object):

  cdef unique_ptr[fst.ArcClass] _arc

  cpdef Arc copy(self)


cdef Arc _init_Arc(const fst.ArcClass &arc)


cdef class ArcIterator(object):

  cdef shared_ptr[fst.FstClass] _fst
  cdef unique_ptr[fst.ArcIteratorClass] _aiter

  cpdef bool done(self)

  cpdef uint32 flags(self)

  cpdef void next(self)

  cpdef size_t position(self)

  cpdef void reset(self)

  cpdef void seek(self, size_t a)

  cpdef void set_flags(self, uint32 flags, uint32 mask)

  cpdef object value(self)


cdef class MutableArcIterator(object):

  cdef shared_ptr[fst.MutableFstClass] _mfst
  cdef unique_ptr[fst.MutableArcIteratorClass] _aiter

  cpdef bool done(self)

  cpdef uint32 flags(self)

  cpdef void next(self)

  cpdef size_t position(self)

  cpdef void reset(self)

  cpdef void seek(self, size_t a)

  cpdef void set_flags(self, uint32 flags, uint32 mask)

  cpdef void set_value(self, Arc arc)

  cpdef object value(self)


cdef class StateIterator(object):

  cdef shared_ptr[fst.FstClass] _fst
  cdef unique_ptr[fst.StateIteratorClass] _siter

  cpdef bool done(self)

  cpdef void next(self)

  cpdef void reset(self)

  cpdef int64 value(self)


# Constructive operations on Fst.


cdef _Fst _map(_Fst ifst, float delta=?, map_type=?, double power=?, weight=?)

cpdef _Fst arcmap(_Fst ifst, float delta=?, map_type=?, double power=?,
                  weight=?)

cpdef _MutableFst compose(_Fst ifst1, _Fst ifst2, compose_filter=?,
                          bool connect=?)

cpdef _Fst convert(_Fst ifst, fst_type=?)

cpdef _MutableFst determinize(_Fst ifst, float delta=?, det_type=?,
                              int64 nstate=?, int64 subsequential_label=?,
                              weight=?, bool increment_subsequential_label=?)

cpdef _MutableFst difference(_Fst ifst1, _Fst ifst2, compose_filter=?,
                             bool connect=?)

cpdef _MutableFst disambiguate(_Fst ifst, float delta=?, int64 nstate=?,
                               int64 subsequential_label=?, weight=?)

cpdef _MutableFst epsnormalize(_Fst ifst, bool eps_norm_output=?)

cpdef bool equal(_Fst ifst1, _Fst ifst2, float delta=?)

cpdef bool equivalent(_Fst ifst1, _Fst ifst2, float delta=?) except *

cpdef _MutableFst intersect(_Fst ifst1, _Fst ifst2, compose_filter=?,
                            bool connect=?)

cpdef bool isomorphic(_Fst ifst1, _Fst ifst2, float delta=?)

cpdef _MutableFst prune(_Fst ifst, float delta=?, int64 nstate=?,
                        weight=?)

cpdef _MutableFst push(_Fst ifst, float delta=?, bool push_weights=?,
                       bool push_labels=?, bool remove_common_affix=?,
                       bool remove_total_weight=?, bool to_final=?)

cpdef bool randequivalent(_Fst ifst1, _Fst ifst2, int32 npath=?,
                          float delta=?, time_t seed=?, select=?,
                          int32 max_length=?) except *

cpdef _MutableFst randgen(_Fst ifst, int32 npath=?, time_t seed=?,
                          select=?, int32 max_length=?,
                          bool remove_total_weight=?, bool weighted=?)

cdef fst.ReplaceLabelType _get_replace_label_type(string rlt,
    bool epsilon_on_replace) except *

cpdef _MutableFst replace(pairs, call_arc_labeling=?, return_arc_labeling=?,
                          bool epsilon_on_replace=?, int64 return_label=?)

cpdef _MutableFst reverse(_Fst ifst, bool require_superinitial=?)

cdef vector[fst.WeightClass] *_shortestdistance(_Fst ifst, float delta=?,
                                                int64 nstate=?, queue_type=?,
                                                bool reverse=?) except *

cpdef _MutableFst shortestpath(_Fst ifst, float delta=?, int32 nshortest=?,
                               int64 nstate=?, queue_type=?, bool unique=?,
                               weight=?)

cpdef _Fst statemap(_Fst ifst, map_type)

cpdef _MutableFst synchronize(_Fst ifst)


# Compiler.


cdef class Compiler(object):

  cdef unique_ptr[stringstream] _sstrm
  cdef string _fst_type
  cdef string _arc_type
  cdef const fst.SymbolTable *_isymbols
  cdef const fst.SymbolTable *_osymbols
  cdef const fst.SymbolTable *_ssymbols
  cdef bool _acceptor
  cdef bool _keep_isymbols
  cdef bool _keep_osymbols
  cdef bool _keep_state_numbering
  cdef bool _allow_negative_labels

  cpdef _Fst compile(self)

  cpdef void write(self, expression)


# FarReader.

cdef class FarReader(object):

  cdef unique_ptr[fst.FarReaderClass] _reader

  cpdef string arc_type(self)

  cpdef bool done(self)

  cpdef bool error(self)

  cpdef string far_type(self)

  cpdef bool find(self, key) except *

  cpdef _Fst get_fst(self)

  cpdef string get_key(self)

  cpdef void next(self)

  cpdef void reset(self)


# FarWriter.

cdef class FarWriter(object):

  cdef unique_ptr[fst.FarWriterClass] _writer

  cpdef string arc_type(self)

  cdef void close(self)

  cpdef void add(self, key, _Fst ifst) except *

  cpdef bool error(self)

  cpdef string far_type(self)
