## @package chapel.chpl 
# type definitions for the conversion code generator
import ir, sidlir
bool     = 'chpl', ir.pt_bool
char     = 'chpl', ir.pt_char
int      = 'chpl', ir.pt_int
long     = 'chpl', ir.pt_long
float    = 'chpl', ir.pt_float
double   = 'chpl', ir.pt_double
fcomplex = 'chpl', ir.pt_fcomplex
dcomplex = 'chpl', ir.pt_dcomplex
string   = 'chpl', ir.pt_string
void_ptr = 'chpl', ir.void_ptr
void     = 'chpl', ir.pt_void
# dummy types for aggregate types
enum     = 'chpl', ir.enum
array    = 'chpl', sidlir.array
rarray   = 'chpl', sidlir.rarray
struct   = 'chpl', sidlir.struct
pointer_type = 'chpl', ir.pointer_type
new_array = 'chpl', 'new_array'
ext      = 'chpl', 'extendable'
opaque   = 'chpl', sidlir.pt_opaque
