// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to create a partition of states.

#ifndef FST_PARTITION_H_
#define FST_PARTITION_H_

#include <algorithm>
#include <vector>


#include <fst/queue.h>


namespace fst {
namespace internal {

template <typename T>
class PartitionIterator;

// Defines a partitioning of elements, used to represent equivalence classes
// for FST operations like minimization. T must be a signed integer type.
//
// The elements are numbered from 0 to num_elements - 1.
// Initialize(num_elements) sets up the class for a given number of elements.
// We maintain a partition of these elements into classes. The classes are also
// numbered from zero; you can add a class with AddClass(), or add them in bulk
// with AllocateClasses(num_classes). Initially the elements are not assigned
// to any class; you set up the initial mapping from elements to classes by
// calling Add(element_id, class_id). You can also move an element to a
// different class by calling Move(element_id, class_id).
//
// We also support a rather specialized interface that allows you to efficiently
// split classes in the Hopcroft minimization algorithm. This maintains a
// binary partition of each class.  Let's call these, rather arbitrarily, the
// 'yes' subset and the 'no' subset of each class, and assume that by default,
// each element of a class is in its 'no' subset. When one calls
// SplitOn(element_id), element_id is moved to the 'yes' subset of its class.
// (If it was already in the 'yes' set, it just stays there). The aim is to
// enable (later) splitting the class in two in time no greater than the time
// already spent calling SplitOn() for that class. We keep a list of the classes
// which have nonempty 'yes' sets, as visited_classes_. When one calls
// FinalizeSplit(Queue *l), for each class in visited_classes_ whose 'yes'
// and 'no' sets are both nonempty, it will create a new class consisting of
// the smaller of the two subsets (and this class will be added to the queue),
// and the old class will now be the larger of the two subsets. This call also
// resets all the yes/no partitions so that everything is in the 'no' subsets.
//
// One cannot use the Move() function if SplitOn() has been called without
// a subsequent call to FinalizeSplit()
template <typename T>
class Partition {
 public:
  Partition() {}

  explicit Partition(T num_elements) { Initialize(num_elements); }

  // Creates an empty partition for num_elements. This means that the elements
  // are not assigned to a class (i.e class_index = -1); you should set up the
  // number of classes using AllocateClasses() or AddClass(), and allocate each
  // element to a class by calling Add(element, class_id).
  void Initialize(size_t num_elements) {
    elements_.resize(num_elements);
    classes_.reserve(num_elements);
    classes_.clear();
    yes_counter_ = 1;
  }

  // Adds a class; returns new number of classes.
  T AddClass() {
    auto num_classes = classes_.size();
    classes_.resize(num_classes + 1);
    return num_classes;
  }

  // Adds 'num_classes' new (empty) classes.
  void AllocateClasses(T num_classes) {
    classes_.resize(classes_.size() + num_classes);
  }

  // Adds element_id to class_id. element_id should already have been allocated
  // by calling Initialize(num_elements)---or the constructor taking
  // num_elements---with num_elements > element_id. element_id must not
  // currently be a member of any class; once elements have been added to a
  // class, use the Move() method to move them from one class to another.
  void Add(T element_id, T class_id) {
    auto &this_element = elements_[element_id];
    auto &this_class = classes_[class_id];
    ++this_class.size;
    // Adds the element to the 'no' subset of the class.
    auto no_head = this_class.no_head;
    if (no_head >= 0) elements_[no_head].prev_element = element_id;
    this_class.no_head = element_id;
    this_element.class_id = class_id;
    // Adds to the 'no' subset of the class.
    this_element.yes = 0;
    this_element.next_element = no_head;
    this_element.prev_element = -1;
  }

  // Moves element_id from 'no' subset of its current class to 'no' subset of
  // class class_id. This may not work correctly if you have called SplitOn()
  // [for any element] and haven't subsequently called FinalizeSplit().
  void Move(T element_id, T class_id) {
    auto elements = &(elements_[0]);
    auto &element = elements[element_id];
    auto &old_class = classes_[element.class_id];
    --old_class.size;
    // Excises the element from the 'no' list of its old class, where it is
    // assumed to be.
    if (element.prev_element >= 0) {
      elements[element.prev_element].next_element = element.next_element;
    } else {
      old_class.no_head = element.next_element;
    }
    if (element.next_element >= 0) {
      elements[element.next_element].prev_element = element.prev_element;
    }
    // Adds to new class.
    Add(element_id, class_id);
  }

  // Moves element_id to the 'yes' subset of its class if it was in the 'no'
  // subset, and marks the class as having been visited.
  void SplitOn(T element_id) {
    auto elements = &(elements_[0]);
    auto &element = elements[element_id];
    if (element.yes == yes_counter_) {
      return;  // Already in the 'yes' set; nothing to do.
    }
    auto class_id = element.class_id;
    auto &this_class = classes_[class_id];
    // Excises the element from the 'no' list of its class.
    if (element.prev_element >= 0) {
      elements[element.prev_element].next_element = element.next_element;
    } else {
      this_class.no_head = element.next_element;
    }
    if (element.next_element >= 0) {
      elements[element.next_element].prev_element = element.prev_element;
    }
    // Adds the element to the 'yes' list.
    if (this_class.yes_head >= 0) {
      elements[this_class.yes_head].prev_element = element_id;
    } else {
      visited_classes_.push_back(class_id);
    }
    element.yes = yes_counter_;
    element.next_element = this_class.yes_head;
    element.prev_element = -1;
    this_class.yes_head = element_id;
    this_class.yes_size++;
  }

  // This should be called after one has possibly called SplitOn for one or more
  // elements, thus moving those elements to the 'yes' subset for their class.
  // For each class that has a nontrivial split (i.e., it's not the case that
  // all members are in the 'yes' or 'no' subset), this function creates a new
  // class containing the smaller of the two subsets of elements, leaving the
  // larger group of elements in the old class. The identifier of the new class
  // will be added to the queue provided as the pointer L. This method then
  // moves all elements to the 'no' subset of their class.
  template <class Queue>
  void FinalizeSplit(Queue *queue) {
    for (const auto &visited_class : visited_classes_) {
      const auto new_class = SplitRefine(visited_class);
      if (new_class != -1 && queue) queue->Enqueue(new_class);
    }
    visited_classes_.clear();
    // Incrementation sets all the 'yes' members of the elements to false.
    ++yes_counter_;
  }

  const T ClassId(T element_id) const { return elements_[element_id].class_id; }

  const size_t ClassSize(T class_id) const { return classes_[class_id].size; }

  const T NumClasses() const { return classes_.size(); }

 private:
  friend class PartitionIterator<T>;

  // Information about a given element.
  struct Element {
    T class_id;      // Class ID of this element.
    T yes;           // This is to be interpreted as a bool, true if it's in the
                     // 'yes' set of this class. The interpretation as bool is
                     // (yes == yes_counter_ ? true : false).
    T next_element;  // Next element in the 'no' list or 'yes' list of this
                     // class, whichever of the two we belong to (think of
                     // this as the 'next' in a doubly-linked list, although
                     // it is an index into the elements array). Negative
                     // values corresponds to null.
    T prev_element;  // Previous element in the 'no' or 'yes' doubly linked
                     // list. Negative values corresponds to null.
  };

  // Information about a given class.
  struct Class {
    Class() : size(0), yes_size(0), no_head(-1), yes_head(-1) {}
    T size;      // Total number of elements in this class ('no' plus 'yes'
                 // subsets).
    T yes_size;  // Total number of elements of 'yes' subset of this class.
    T no_head;   // Index of head element of doubly-linked list in 'no' subset.
                 // Everything is in the 'no' subset until you call SplitOn().
                 // -1 means no element.
    T yes_head;  // Index of head element of doubly-linked list in 'yes' subset.
                 // -1 means no element.
  };

  // This method, called from FinalizeSplit(), checks whether a class has to
  // be split (a class will be split only if its 'yes' and 'no' subsets are
  // both nonempty, but one can assume that since this function was called, the
  // 'yes' subset is nonempty). It splits by taking the smaller subset and
  // making it a new class, and leaving the larger subset of elements in the
  // 'no' subset of the old class. It returns the new class if created, or -1
  // if none was created.
  T SplitRefine(T class_id) {
    auto yes_size = classes_[class_id].yes_size;
    auto size = classes_[class_id].size;
    auto no_size = size - yes_size;
    if (no_size == 0) {
      // All members are in the 'yes' subset, so we don't have to create a new
      // class, just move them all to the 'no' subset.
      classes_[class_id].no_head = classes_[class_id].yes_head;
      classes_[class_id].yes_head = -1;
      classes_[class_id].yes_size = 0;
      return -1;
    } else {
      auto new_class_id = classes_.size();
      classes_.resize(classes_.size() + 1);
      auto &old_class = classes_[class_id];
      auto &new_class = classes_[new_class_id];
      // The new_class will have the values from the constructor.
      if (no_size < yes_size) {
        // Moves the 'no' subset to new class ('no' subset).
        new_class.no_head = old_class.no_head;
        new_class.size = no_size;
        // And makes the 'yes' subset of the old class ('no' subset).
        old_class.no_head = old_class.yes_head;
        old_class.yes_head = -1;
        old_class.size = yes_size;
        old_class.yes_size = 0;
      } else {
        // Moves the 'yes' subset to the new class (to the 'no' subset)
        new_class.size = yes_size;
        new_class.no_head = old_class.yes_head;
        // Retains only the 'no' subset in the old class.
        old_class.size = no_size;
        old_class.yes_size = 0;
        old_class.yes_head = -1;
      }
      auto elements = &(elements_[0]);
      // Updates the 'class_id' of all the elements we moved.
      for (auto e = new_class.no_head; e >= 0; e = elements[e].next_element) {
        elements[e].class_id = new_class_id;
      }
      return new_class_id;
    }
  }

  // elements_[i] contains all info about the i'th element.
  std::vector<Element> elements_;
  // classes_[i] contains all info about the i'th class.
  std::vector<Class> classes_;
  // Set of visited classes to be used in split refine.
  std::vector<T> visited_classes_;
  // yes_counter_ is used in interpreting the 'yes' members of class Element.
  // If element.yes == yes_counter_, we interpret that element as being in the
  // 'yes' subset of its class. This allows us to, in effect, set all those
  // bools to false at a stroke by incrementing yes_counter_.
  T yes_counter_;
};

// Iterates over members of the 'no' subset of a class in a partition. (When
// this is used, everything is in the 'no' subset).
template <typename T>
class PartitionIterator {
 public:
  using Element = typename Partition<T>::Element;

  PartitionIterator(const Partition<T> &partition, T class_id)
      : partition_(partition),
        element_id_(partition_.classes_[class_id].no_head),
        class_id_(class_id) {}

  bool Done() { return element_id_ < 0; }

  const T Value() { return element_id_; }

  void Next() { element_id_ = partition_.elements_[element_id_].next_element; }

  void Reset() { element_id_ = partition_.classes_[class_id_].no_head; }

 private:
  const Partition<T> &partition_;
  T element_id_;
  T class_id_;
};

}  // namespace internal
}  // namespace fst

#endif  // FST_PARTITION_H_
