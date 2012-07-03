
use pgas;
use sidl;
extern record pgas_MPIinitializer__object { var d_data: opaque; };
extern proc pgas_MPIinitializer__createObject(d_data: int, out ex: sidl_BaseInterface__object): pgas_MPIinitializer__object;


use pgas;
use sidl;
extern record pgas_blockedDouble3dArray__object { var d_data: opaque; };
extern proc pgas_blockedDouble3dArray__createObject(d_data: int, out ex: sidl_BaseInterface__object): pgas_blockedDouble3dArray__object;


