## @package chapel.chpl 
# type definitions for the conversion code generator
import ir, sidl
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
array    = 'chpl', sidl.array
rarray   = 'chpl', sidl.rarray
struct   = 'chpl', sidl.struct
pointer_type = 'chpl', ir.pointer_type
