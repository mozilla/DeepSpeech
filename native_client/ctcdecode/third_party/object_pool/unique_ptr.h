#ifndef GODEFV_MEMORY_ALLOCATED_UNIQUE_PTR_H
#define GODEFV_MEMORY_ALLOCATED_UNIQUE_PTR_H

#include <memory>

namespace godefv{

//! A deleter to deallocate memory which have been allocated by the given allocator.
template<class Allocator> 
struct allocator_deleter_t
{
	allocator_deleter_t(Allocator const& allocator) :
		mAllocator{ allocator }
	{}

	void operator()(typename Allocator::value_type* ptr)
	{
		mAllocator.deallocate(ptr, 1);
	}

private:
	Allocator mAllocator;
};

//! A smart pointer like std::unique_ptr but templated on an allocator instead of a deleter.
//! The deleter is deduced from the given allocator.
template<class T, class Allocator = std::allocator<T>>
struct unique_ptr_t : public std::unique_ptr<T, allocator_deleter_t<Allocator>>
{
	using base_t = std::unique_ptr<T, allocator_deleter_t<Allocator>>;

	unique_ptr_t(Allocator allocator = Allocator{}) :
		base_t{ allocator.allocate(1), allocator_deleter_t<Allocator>{ allocator } }
	{}
};

} // namespace godefv 

#endif // GODEFV_MEMORY_ALLOCATED_UNIQUE_PTR_H 
