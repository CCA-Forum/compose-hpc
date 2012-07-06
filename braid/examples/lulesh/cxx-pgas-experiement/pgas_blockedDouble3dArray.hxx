// 
// File:          pgas_blockedDouble3dArray.hxx
// Symbol:        pgas.blockedDouble3dArray-v1.0
// Symbol Type:   class
// Babel Version: 2.0.0 (Revision: 7481 trunk)
// Description:   Client-side glue code for pgas.blockedDouble3dArray
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pgas_blockedDouble3dArray_hxx
#define included_pgas_blockedDouble3dArray_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace pgas { 

  class blockedDouble3dArray;
} // end namespace pgas

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::pgas::blockedDouble3dArray >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#include <stdint.h>
#ifndef included_pgas_blockedDouble3dArray_IOR_h
#include "pgas_blockedDouble3dArray_IOR.h"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
namespace sidl {
  namespace rmi {
    class Call;
    class Return;
    class Ticket;
  }
  namespace rmi {
    class InstanceHandle;
  }
}
namespace pgas { 

  /**
   * Symbol "pgas.blockedDouble3dArray" (version 1.0)
   */
  class blockedDouble3dArray: public virtual ::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // Special methods for throwing exceptions
    // 

  private:
    static 
    void
    throwException0(
      const char* methodName,
      struct sidl_BaseInterface__object *_exception
    )
      // throws:
    ;
  public:
    typedef struct pgas_blockedDouble3dArray__object ior_t;
    typedef struct pgas_blockedDouble3dArray__external ext_t;
    typedef struct pgas_blockedDouble3dArray__sepv sepv_t;


    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:

    /**
     * allocate a blocked cubic array of doubles in sizesizesize
     */
    void
    allocate (
      /* in */int32_t size
    )
    ;

    /**
     * user defined non-static method
     */
    double
    get (
      /* in */int32_t idx1,
      /* in */int32_t idx2,
      /* in */int32_t idx3
    )
    ;

    /**
     * user defined non-static method
     */
    void
    set (
      /* in */int32_t idx1,
      /* in */int32_t idx2,
      /* in */int32_t idx3,
      /* in */double val
    )
    ;


    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

    // default constructor
    blockedDouble3dArray() { }
    // static constructor
    static ::pgas::blockedDouble3dArray _create();


#ifdef WITH_RMI

    // RMI constructor
    static ::pgas::blockedDouble3dArray _create( /*in*/ const std::string& url 
      );

    // RMI connect
    static inline ::pgas::blockedDouble3dArray _connect( /*in*/ const 
      std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::pgas::blockedDouble3dArray _connect( /*in*/ const std::string& url,
      /*in*/ const bool ar  );


#endif /*WITH_RMI*/

    // default destructor
    virtual ~blockedDouble3dArray () { }

    // copy constructor
    blockedDouble3dArray ( const blockedDouble3dArray& original );

    // assignment operator
    blockedDouble3dArray& operator= ( const blockedDouble3dArray& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    blockedDouble3dArray ( blockedDouble3dArray::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    blockedDouble3dArray ( blockedDouble3dArray::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    inline void _set_ior( ior_t* ptr ) throw () { 
      d_self = reinterpret_cast< void*>(ptr);

      if( ptr != NULL ) {
      } else {
      }
    }

    virtual int _set_ior_typesafe( struct sidl_BaseInterface__object *obj,
                                   const ::std::type_info &argtype );

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "pgas.blockedDouble3dArray";}

    static struct pgas_blockedDouble3dArray__object* _cast(const void* src);

    // execute member function by name
    void _exec(const std::string& methodName,
               ::sidl::rmi::Call& inArgs,
               ::sidl::rmi::Return& outArgs);

    /**
     * Get the URL of the Implementation of this object (for RMI)
     */
    ::std::string
    _getURL() // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to enable/disable method hooks invocation.
     */
    void
    _set_hooks (
      /* in */bool enable
    )
    // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to enable/disable interface contract enforcement.
     */
    void
    _set_contracts (
      /* in */bool enable,
      /* in */const ::std::string& enfFilename,
      /* in */bool resetCounters
    )
    // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to dump contract enforcement statistics.
     */
    void
    _dump_stats (
      /* in */const ::std::string& filename,
      /* in */const ::std::string& prefix
    )
    // throws:
    //    ::sidl::RuntimeException
    ;

    // return true iff object is remote
    bool _isRemote() const { 
      ior_t* self = const_cast<ior_t*>(_get_ior() );
      struct sidl_BaseInterface__object *throwaway_exception;
      return (*self->d_epv->f__isRemote)(self, &throwaway_exception) == TRUE;
    }

    // return true iff object is local
    bool _isLocal() const {
      return !_isRemote();
    }

  protected:
    // Pointer to external (DLL loadable) symbols (shared among instances)
    static const ext_t * s_ext;

    // Global cache for _get_sepv()
    static sepv_t *_sepv;

  public:
    static const ext_t * _get_ext() throw ( ::sidl::NullIORException );

  }; // end class blockedDouble3dArray
} // end namespace pgas

extern "C" {


#pragma weak pgas_blockedDouble3dArray__connectI

  /**
   * RMI connector function for the class. (no addref)
   */
  struct pgas_blockedDouble3dArray__object*
  pgas_blockedDouble3dArray__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::pgas::blockedDouble3dArray > {
    typedef array< ::pgas::blockedDouble3dArray > cxx_array_t;
    typedef ::pgas::blockedDouble3dArray cxx_item_t;
    typedef struct pgas_blockedDouble3dArray__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct pgas_blockedDouble3dArray__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::pgas::blockedDouble3dArray > > iterator;
    typedef const_array_iter< array_traits< ::pgas::blockedDouble3dArray > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::pgas::blockedDouble3dArray >: public interface_array< 
    array_traits< ::pgas::blockedDouble3dArray > > {
  public:
    typedef interface_array< array_traits< ::pgas::blockedDouble3dArray > > 
      Base;
    typedef array_traits< ::pgas::blockedDouble3dArray >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::pgas::blockedDouble3dArray >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::pgas::blockedDouble3dArray >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::pgas::blockedDouble3dArray >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::pgas::blockedDouble3dArray >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct pgas_blockedDouble3dArray__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::pgas::blockedDouble3dArray >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::pgas::blockedDouble3dArray >&
    operator =( const array< ::pgas::blockedDouble3dArray >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#endif
