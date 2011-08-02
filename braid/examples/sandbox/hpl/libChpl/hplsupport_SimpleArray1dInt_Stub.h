#ifndef __HPLSUPPORT_SIMPLEARRAY1DINT_STUB_H__
#define __HPLSUPPORT_SIMPLEARRAY1DINT_STUB_H__
// Package header (enums, etc...)
#include <stdint.h>
#include <hplsupport.h>
#include <hplsupport_SimpleArray1dInt_IOR.h>
typedef struct hplsupport_SimpleArray1dInt__object _hplsupport_SimpleArray1dInt__object;
typedef _hplsupport_SimpleArray1dInt__object* hplsupport_SimpleArray1dInt__object;
#ifndef _CHPL_SIDL_BASETYPES
#define _CHPL_SIDL_BASETYPES
typedef struct sidl_BaseInterface__object _sidl_BaseInterface__object;
typedef _sidl_BaseInterface__object* sidl_BaseInterface__object;
hplsupport_SimpleArray1dInt__object hplsupport_SimpleArray1dInt__createObject(hplsupport_SimpleArray1dInt__object copy, sidl_BaseInterface__object* ex);
#endif

#endif
