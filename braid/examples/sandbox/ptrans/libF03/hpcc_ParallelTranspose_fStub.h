/*
 * File:          hpcc_ParallelTranspose_fStub.h
 * Symbol:        hpcc.ParallelTranspose-v0.1
 * Symbol Type:   class
 * Babel Version: 2.0.0 (Revision: 0  )
 * Description:   Client-side documentation text for hpcc.ParallelTranspose
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_hpcc_ParallelTranspose_fStub_h
#define included_hpcc_ParallelTranspose_fStub_h

/**
 * Symbol "hpcc.ParallelTranspose" (version 0.1)
 */

#ifndef included_hpcc_ParallelTranspose_IOR_h
#include "hpcc_ParallelTranspose_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak hpcc_ParallelTranspose__connectI

/**
 * RMI connector function for the class. (no addref)
 */
struct hpcc_ParallelTranspose__object*
hpcc_ParallelTranspose__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
