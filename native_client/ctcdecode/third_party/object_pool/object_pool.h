#ifndef GODEFV_MEMORY_OBJECT_POOL_H
#define GODEFV_MEMORY_OBJECT_POOL_H

#include "unique_ptr.h"
#include <memory>
#include <vector>
#include <array>

namespace godefv{

// Forward declaration
template<class Object, template<class T> class Allocator = std::allocator, std::size_t ChunkSize = 1024>
class object_pool_t;

//! Custom deleter to recycle the deleted pointers of the object_pool_t. 
template<class Object, template<class T> class Allocator = std::allocator, std::size_t ChunkSize = 1024>
struct object_pool_deleter_t{
private:
	object_pool_t<Object, Allocator, ChunkSize>* object_pool_ptr;
public:
	explicit object_pool_deleter_t(decltype(object_pool_ptr) input_object_pool_ptr) :
		object_pool_ptr(input_object_pool_ptr)
	{}

	void operator()(Object* object_ptr)
	{
		object_pool_ptr->delete_object(object_ptr);
	}
};

//! Allocates instances of Object efficiently (constant time and log((maximum number of Objects used at the same time)/ChunkSize) calls to malloc in the whole lifetime of the object pool). 
//! When an instance returned by the object pool is destroyed, its allocated memory is recycled by the object pool. Defragmenting the object pool to free memory is not possible. 
template<class Object, template<class T> class Allocator, std::size_t ChunkSize>
class object_pool_t{
	//! An object slot is an uninitialized memory space of the same size as Object. 
	//! It is initially "free". It can then be "used" to construct an Object in place and the pointer to it is returned by the object pool. When the pointer is destroyed, the object slot is "recycled" and can be used again but it is not "free" anymore because "free" object slots are contiguous in memory.
	using object_slot_t=std::array<char, sizeof(Object)>;

	//! To minimize calls to malloc, the object slots are allocated in chunks. 
	//! For example, if ChunkSize=8, a chunk may look like this : |used|recycled|used|used|recycled|free|free|free|. In this example, if more than 5 new Object are now asked from the object pool, at least one new chunk of 8 object slots will be allocated.
	using chunk_t=std::array<object_slot_t, ChunkSize>; 
	Allocator<chunk_t> chunk_allocator; //!< This allocator can be used to have aligned memory if required.
	std::vector<unique_ptr_t<chunk_t, decltype(chunk_allocator)>> memory_chunks; 

	//! Recycled object slots are tracked using a stack of pointers to them. When an object slot is recycled, a pointer to it is pushed in constant time. When a new object is constructed, a recycled object slot can be found and poped in constant time.
	std::vector<object_slot_t*> recycled_object_slots;

	object_slot_t* free_object_slots_begin; 
	object_slot_t* free_object_slots_end; 

	//! When a pointer provided by the ObjectPool is deleted, its memory is converted to an object slot to be recycled. 
	void delete_object(Object* object_ptr){
		object_ptr->~Object();
		recycled_object_slots.push_back(reinterpret_cast<object_slot_t*>(object_ptr));
	}
	friend object_pool_deleter_t<Object, Allocator, ChunkSize>;

public:
	using object_t = Object;
	using deleter_t = object_pool_deleter_t<Object, Allocator, ChunkSize>;
	using object_unique_ptr_t = std::unique_ptr<object_t, deleter_t>; //!< The type returned by the object pool.

	object_pool_t(Allocator<chunk_t> const& allocator = Allocator<chunk_t>{}) :
		chunk_allocator{ allocator },
		free_object_slots_begin{ free_object_slots_end } // At the begining, set the 2 iterators at the same value to simulate a full pool.
	{}

	//! Returns a unique pointer to an object_t using an unused object slot from the object pool. 
	template<class... Args> object_unique_ptr_t make_unique(Args&&... vars){
		auto construct_object_unique_ptr=[&](object_slot_t* object_slot){
			return object_unique_ptr_t{ new (reinterpret_cast<object_t*>(object_slot)) object_t{ std::forward<Args>(vars)... } , deleter_t{ this } };
		};

		// If a recycled object slot is available, use it.
		if (!recycled_object_slots.empty())
		{
			auto object_slot = recycled_object_slots.back();
			recycled_object_slots.pop_back();
			return construct_object_unique_ptr(object_slot);
		}

		// If the pool is full: add a new chunk.
		if (free_object_slots_begin == free_object_slots_end)
		{
			memory_chunks.emplace_back(chunk_allocator);
			auto& new_chunk = memory_chunks.back();
			free_object_slots_begin=new_chunk->data();
			free_object_slots_end  =free_object_slots_begin+new_chunk->size();
		}

		// We know that there is now at least one free object slot, use it.
		return construct_object_unique_ptr(free_object_slots_begin++);
	}

	//! Returns the total number of object slots (free, recycled, or used).
	std::size_t capacity() const{
		return memory_chunks.size()*ChunkSize;
	}

	//! Returns the number of currently used object slots.
	std::size_t size() const{
		return capacity() - static_cast<std::size_t>(free_object_slots_end-free_object_slots_begin) - recycled_object_slots.size();
	}
};

} /* namespace godefv */

#endif /* GODEFV_MEMORY_OBJECT_POOL_H */
