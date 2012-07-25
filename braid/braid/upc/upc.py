## @package UPC.upc
# type definitions for the conversion code generator
import ir, sidl
bool     = 'UPC', ir.pt_bool
char     = 'UPC', ir.pt_char
int      = 'UPC', ir.pt_int
long     = 'UPC', ir.pt_long
float    = 'UPC', ir.pt_float
double   = 'UPC', ir.pt_double
fcomplex = 'UPC', ir.pt_fcomplex
dcomplex = 'UPC', ir.pt_dcomplex
string   = 'UPC', ir.pt_string
void_ptr = 'UPC', ir.void_ptr
void     = 'UPC', ir.pt_void
# dummy types for aggregate types
enum     = 'UPC', ir.enum
array    = 'UPC', sidl.array
rarray   = 'UPC', sidl.rarray
struct   = 'UPC', sidl.struct
pointer_type = 'UPC', ir.pointer_type
new_array = 'UPC', 'new_array'
