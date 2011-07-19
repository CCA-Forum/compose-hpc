

    use sidl;
    _extern record blas_VectorUtils__object {
    };
    
    _extern proc blas_VectorUtils__createObject(d_data: int, inout ex: 
      sidl_BaseInterface__object): blas_VectorUtils__object;
    
    // All the static methods of class VectorUtils
    module VectorUtils_static {
    
    proc helper_daxpy( in n: int(32), in alpha: real(64), in X: sidl.Array(real(64), sidl_double__array), in Y: sidl.Array(real(64), sidl_double__array), inout _babel_param_ex: BaseException) {
        var _ex:sidl_BaseInterface__object;
        _extern proc blas_VectorUtils_helper_daxpy_stub( in n: int(32), in alpha: real(64), in 
        X: sidl_double__array, in Y: sidl_double__array, inout _ex: 
        sidl_BaseInterface__object);
        blas_VectorUtils_helper_daxpy_stub( n, alpha, X.self, Y.self, _ex);
        _extern proc IS_NULL(inout aRef): bool;
        if (! IS_NULL( _ex)) {
          _babel_param_ex = new BaseException( _ex);
        }

    }
    
    
    }
    class VectorUtils /**/ {
    var self: blas_VectorUtils__object;
        /**
         * Constructor
         */
        proc VectorUtils( inout _babel_param_ex: BaseException) {
            var ex: sidl_BaseInterface__object;
            this.self = blas_VectorUtils__createObject(0, ex);
            _extern proc IS_NULL(inout aRef): bool;
            if (IS_NULL(ex)) {
               _babel_param_ex = new BaseException(ex);
            }
            _extern proc blas_VectorUtils_addRef_stub( in self: blas_VectorUtils__object, inout 
        _ex: sidl_BaseInterface__object);
            blas_VectorUtils_addRef_stub( this.self, ex);
        }
        
        /**
         * Constructor for wrapping an existing object
         */
        proc VectorUtils( in obj: blas_VectorUtils__object) {
            var ex: sidl_BaseInterface__object;
            this.self = obj;
            _extern proc blas_VectorUtils_addRef_stub( in self: blas_VectorUtils__object, inout 
        _ex: sidl_BaseInterface__object);
            blas_VectorUtils_addRef_stub( this.self, ex);
        }
        
        /**
         * Destructor
         */
        proc ~VectorUtils() {
            var ex: sidl_BaseInterface__object;
            _extern proc blas_VectorUtils_deleteRef_stub( in self: blas_VectorUtils__object, inout 
        _ex: sidl_BaseInterface__object);
            blas_VectorUtils_deleteRef_stub( this.self, ex);
        }
        
        
    };
