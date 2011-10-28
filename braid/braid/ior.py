# type definitions for the conversion code generator
import ir, sidl
bool     = ir.pt_bool
char     = ir.pt_char
int      = ir.pt_int
long     = ir.pt_long
float    = ir.pt_float
double   = ir.pt_double
fcomplex = ir.pt_fcomplex
dcomplex = ir.pt_dcomplex
string   = ir.pt_string
void_ptr = ir.void_ptr
# dummy types for aggregate types
enum     = ir.enum
array    = sidl.array
rarray   = sidl.rarray
struct   = ir.struct
void     = ir.pt_void
pointer_type = ir.pointer_type 
