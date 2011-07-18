

use sidl;
_extern record blas_VectorUtils__object {};
_extern proc blas_VectorUtils__createObject(d_data: int, inout ex: 
  sidl_BaseInterface__object): blas_VectorUtils__object;

    _extern proc blas_VectorUtils_helper_daxpy_stub( in n: int(32), in alpha: real(64), in 
        X: sidl_double__array, inout Y: sidl_double__array, inout ex: 
        sidl_BaseInterface__object);
    
// All the static methods of class VectorUtils
module VectorUtils_static {

    proc helper_daxpy( in n: int(32), in alpha: real(64), in X: sidl.Array(real(64), sidl_double__array), inout Y: sidl.Array(real(64), sidl_double__array)) {
        var ex:sidl_BaseInterface__object;
        var _IOR_Y:sidl_double__array;
        _IOR_Y = Y.ior;
        blas_VectorUtils_helper_daxpy_stub( n, alpha, X.ior, _IOR_Y, ex);
        Y = new sidl.Array( real(64), sidl_double__array, _IOR_Y);
    }
    
    
}
class VectorUtils {
var self: blas_VectorUtils__object;
    /**
     * Constructor
     */
    proc VectorUtils() {
          var ex: sidl_BaseInterface__object;
          this.self = blas_VectorUtils__createObject(0, ex);
    }
    
    /**
     * Constructor for wrapping an existing object
     */
    proc VectorUtils( in obj: blas_VectorUtils__object) {
          this.self = obj;
    }
    
    
};
;
