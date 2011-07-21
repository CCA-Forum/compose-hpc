/*
 * File:          hplsupport_BlockCyclicDistArray2dDouble_fSkel.c
 * Symbol:        hplsupport.BlockCyclicDistArray2dDouble-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 7138  trunk)
 * Description:   Server-side glue code for hplsupport.BlockCyclicDistArray2dDouble
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */


#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_fStub_h
#include "hplsupport_BlockCyclicDistArray2dDouble_fStub.h"
#endif
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "sidlfortran.h"
#ifndef included_sidl_String_h
#include "sidl_String.h"
#endif
#ifndef included_sidl_CastException_h
#include "sidl_CastException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#include <stdio.h>
#ifndef included_sidlf90array_h
#include "sidlf90array.h"
#endif
#include "sidl_header.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_Exception_h
#include "sidl_Exception.h"
#endif
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include "sidl_Loader.h"
#endif
#include "hplsupport_BlockCyclicDistArray2dDouble_IOR.h"
#include "hplsupport_BlockCyclicDistArray2dDouble_fAbbrev.h"
#include "sidl_BaseException_IOR.h"
#include "sidl_BaseInterface_IOR.h"
#include "sidl_ClassInfo_IOR.h"
#include "sidl_RuntimeException_IOR.h"
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidlOps_h
#include "sidlOps.h"
#endif
/*
 * Includes for all method dependencies.
 */

#ifndef included_hplsupport_BlockCyclicDistArray2dDouble_fStub_h
#include "hplsupport_BlockCyclicDistArray2dDouble_fStub.h"
#endif
#ifndef included_sidl_BaseClass_fStub_h
#include "sidl_BaseClass_fStub.h"
#endif
#ifndef included_sidl_BaseInterface_fStub_h
#include "sidl_BaseInterface_fStub.h"
#endif
#ifndef included_sidl_ClassInfo_fStub_h
#include "sidl_ClassInfo_fStub.h"
#endif
#ifndef included_sidl_RuntimeException_fStub_h
#include "sidl_RuntimeException_fStub.h"
#endif
#include <string.h>
#include <stdio.h>

#ifdef WITH_RMI
struct sidl_BaseInterface__object* 
  skel_hplsupport_BlockCyclicDistArray2dDouble_fconnect_sidl_BaseInterface(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return sidl_BaseInterface__connectI(url, ar, _ex);
}

#endif /*WITH_RMI*/
void
hplsupport_BlockCyclicDistArray2dDouble__load_skel_c
(
  struct sidl_BaseInterface__object* *exception
);

void hplsupport_BlockCyclicDistArray2dDouble__call_load(void) { 
  struct sidl_BaseInterface__object *throwaway = NULL;
  struct sidl_BaseInterface__object **exception = &throwaway;
  hplsupport_BlockCyclicDistArray2dDouble__load_skel_c(exception);
}


#ifdef __cplusplus
extern "C" {
#endif


void hplsupport_BlockCyclicDistArray2dDouble__set_epv_bindc(struct 
  hplsupport_BlockCyclicDistArray2dDouble__epv *);

void
hplsupport_BlockCyclicDistArray2dDouble__set_epv(struct 
  hplsupport_BlockCyclicDistArray2dDouble__epv * epv) {
  hplsupport_BlockCyclicDistArray2dDouble__set_epv_bindc(epv);
}
#ifdef __cplusplus
}
#endif

