
IMPLHDRS =
IMPLSRCS = pgas_Impl.chpl
IORHDRS = pgas_MPIinitializer_IOR.h pgas_blockedDouble3dArray_IOR.h #FIXME Array_IOR.h
IORSRCS = pgas_MPIinitializer_IOR.c pgas_blockedDouble3dArray_IOR.c
SKELSRCS = pgas_MPIinitializer_Skel.c pgas_blockedDouble3dArray_Skel.c
STUBHDRS = pgas_MPIinitializer_Stub.h pgas_MPIinitializer_cStub.h pgas_blockedDouble3dArray_Stub.h pgas_blockedDouble3dArray_cStub.h
STUBSRCS = pgas_MPIinitializer_cStub.c pgas_blockedDouble3dArray_cStub.c
