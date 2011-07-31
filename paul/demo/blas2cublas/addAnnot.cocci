// Script that adds annotations to all blas calls in a source file.



@cbcall7@
identifier CblasCall ~= "^\(sdot\|ddot\|dsdot\|sdsdot\|\
			cblas_sdot\|cblas_ddot\|cblas_dsdot\|cblas_sdsdot\)$";

@@



+ xy=10;
CblasCall(...);

