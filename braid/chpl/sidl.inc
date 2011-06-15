// -*- chpl -*- 
// This fragment will preprocessed to yield sidl.chpl during compile time.

module sidl {
  // FIXME: autogenerate this from sidl.sidl
  _extern record sidl_BaseInterface__object {};

# include <array.inc>

// Start: Borrowed Array related items  
# include <barray.inc>
// End: Borrowed Array related items  

// Start: Distributed Array related items  
# include <darray.inc>
// End: Distributed Array related items  
}
