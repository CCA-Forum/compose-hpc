// Script that adds annotations to all blas calls in a source file.

@initialize:python@

count = 0

@cbcall@
position cbc;
identifier CblasCall ~= "^\(cblas_sgemm\|cblas_dgemm\|cblas_cgemm\|\
		       cblas_cgemm3m\|cblas_zgemm\|cblas_zgemm3m\)$";
@@

CblasCall@cbc(...);

@script:python cbcs@
cbc << cbcall.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall.cbc;
identifier cbcs.init_val;
identifier cbcall.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall1@
position cbc;
identifier CblasCall ~= "^\(cblas_ssymm\|cblas_dsymm\|cblas_csymm\|cblas_zsymm\)$";
@@

CblasCall@cbc(...);

@script:python cbcs1@
cbc << cbcall1.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall1.cbc;
identifier cbcs1.init_val;
identifier cbcall1.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall2@
position cbc;
identifier CblasCall ~= "^\(cblas_sscal\|cblas_dscal\|\
		       cblas_cscal\|cblas_zscal\|cblas_csscal\|cblas_zdscal\)$";
@@

CblasCall@cbc(...);

@script:python cbcs2@
cbc << cbcall2.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall2.cbc;
identifier cbcs2.init_val;
identifier cbcall2.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall3@
position cbc;
identifier CblasCall ~= "^\(cblas_ssyrk\|cblas_ssyr2k\|cblas_dsyrk\|cblas_csyrk\|\
			cblas_dsyr2k\|cblas_csyr2k\|cblas_zsyr2k\|cblas_zsyrk\)$";
@@

CblasCall@cbc(...);

@script:python cbcs3@
cbc << cbcall3.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall3.cbc;
identifier cbcs3.init_val;
identifier cbcall3.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall4@
position cbc;
identifier CblasCall ~= "^\(cblas_strmm\|cblas_ztrmm\|cblas_dtrmm\|cblas_ctrmm\)$";
@@

CblasCall@cbc(...);

@script:python cbcs4@
cbc << cbcall4.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall4.cbc;
identifier cbcs4.init_val;
identifier cbcall4.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall5@
position cbc;
identifier CblasCall ~= "^\(cblas_strsm\|cblas_ztrsm\|cblas_dtrsm\|cblas_ctrsm\)$";
@@

CblasCall@cbc(...);

@script:python cbcs5@
cbc << cbcall5.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall5.cbc;
identifier cbcs5.init_val;
identifier cbcall5.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall6@
position cbc;
identifier CblasCall ~= "^\(cblas_zhemm\|cblas_chemm\|cblas_cherk\|cblas_zherk\|cblas_cher2k\|cblas_zher2k\)$";
@@

CblasCall@cbc(...);

@script:python cbcs6@
cbc << cbcall6.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall6.cbc;
identifier cbcs6.init_val;
identifier cbcall6.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall7@
position cbc;
identifier CblasCall ~= "^\(sdot\|ddot\|dsdot\|sdsdot\|\
			cblas_sdot\|cblas_ddot\|cblas_dsdot\|cblas_sdsdot\)$";

@@


CblasCall@cbc(...);


@script:python cbcs7@
cbc << cbcall7.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall7.cbc;
identifier cbcs7.init_val;
identifier cbcall7.CblasCall;
@@

+ init_val;

CblasCall@cbc(...);



@cbcall71@
position cbc;
identifier CblasCall ~= "^\(sdoti\|ddoti\|\
			cblas_sdoti\|cblas_ddoti\)$";

@@

CblasCall@cbc(...);


@script:python cbcs71@
cbc << cbcall71.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall71.cbc;
identifier cbcs71.init_val;
identifier cbcall71.CblasCall;
@@


+ init_val;

CblasCall@cbc(...);


@cbcall72@
position cbc;
identifier CblasCall ~= "^\(cdotu_sub\|cdotui_sub\|zdotu_sub\|zdotui_sub\|\
			cblas_cdotu_sub\|cblas_cdotui_sub\|cblas_zdotu_sub\|\
		       cblas_zdotui_sub\)$";

@@


CblasCall@cbc(...);


@script:python cbcs72@
cbc << cbcall72.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall72.cbc;
identifier cbcs72.init_val;
identifier cbcall72.CblasCall;
@@


+ init_val;

CblasCall@cbc(...);


@cbcall73@
position cbc;
identifier CblasCall ~= "^\(cdotc_sub\|cdotci_sub\|zdotc_sub\|zdotci_sub\|\
		       cblas_cdotc_sub\|cblas_cdotci_sub\|\|cblas_zdotc_sub\|cblas_zdotci_sub\)$";
@@


CblasCall@cbc(...);


@script:python cbcs73@
cbc << cbcall73.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall73.cbc;
identifier cbcs73.init_val;
identifier cbcall73.CblasCall;
@@


+ init_val;

CblasCall@cbc(...);



@cbcall8@
position cbc;
identifier CblasCall ~= "^\(snrm2\|sasum\|dnrm2\|dasum\|\
		       scnrm2\|scasum\|dznrm2\|dzasum\|\
		       isamax\|idamax\|icamax\|izamax\|\
		       isamin\|idamin\|icamin\|izamin\|\
		       cblas_snrm2\|cblas_sasum\|cblas_dnrm2\|cblas_dasum\|\
		       cblas_scnrm2\|cblas_scasum\|cblas_dznrm2\|cblas_dzasum\|\
		       cblas_isamax\|cblas_idamax\|cblas_icamax\|cblas_izamax\|\
		       cblas_isamin\|cblas_idamin\|cblas_icamin\|cblas_izamin\)$";
@@


CblasCall@cbc(...);


@script:python cbcs8@
cbc << cbcall8.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall8.cbc;
identifier cbcs8.init_val;
identifier cbcall8.CblasCall;
@@


+ init_val;

CblasCall@cbc(...);


@cbcall9@
position cbc;
identifier CblasCall ~= "^\(sgemv\|dgemv\|cgemv\|zgemv\|\
		       cblas_sgemv\|cblas_dgemv\|cblas_cgemv\|cblas_zgemv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs9@
cbc << cbcall9.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall9.cbc;
identifier cbcs9.init_val;
identifier cbcall9.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);


@cbcall10@
position cbc;
identifier CblasCall ~= "^\(sgbmv\|dgbmv\|cgbmv\|zgbmv\|\
		       cblas_sgbmv\|cblas_dgbmv\|cblas_cgbmv\|cblas_zgbmv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs10@
cbc << cbcall10.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall10.cbc;
identifier cbcs10.init_val;
identifier cbcall10.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall11@
position cbc;
identifier CblasCall ~= "^\(strmv\|dtrmv\|ctrmv\|ztrmv\|\
		       cblas_strmv\|cblas_dtrmv\|cblas_ctrmv\|cblas_ztrmv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs11@
cbc << cbcall11.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall11.cbc;
identifier cbcs11.init_val;
identifier cbcall11.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall12@
position cbc;
identifier CblasCall ~= "^\(stpmv\|dtpmv\|ctpmv\|ztpmv\|\
		       cblas_stpmv\|cblas_dtpmv\|cblas_ctpmv\|cblas_ztpmv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs12@
cbc << cbcall12.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall12.cbc;
identifier cbcs12.init_val;
identifier cbcall12.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall13@
position cbc;
identifier CblasCall ~= "^\(stbsv\|dtbsv\|ctbsv\|ztbsv\|\
		       cblas_stbsv\|cblas_dtbsv\|cblas_ctbsv\|cblas_ztbsv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs13@
cbc << cbcall13.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall13.cbc;
identifier cbcs13.init_val;
identifier cbcall13.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall14@
position cbc;
identifier CblasCall ~= "^\(stpsv\|dtpsv\|ctpsv\|ztpsv\|\
		       cblas_stpsv\|cblas_dtpsv\|cblas_ctpsv\|cblas_ztpsv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs14@
cbc << cbcall14.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall14.cbc;
identifier cbcs14.init_val;
identifier cbcall14.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall15@
position cbc;
identifier CblasCall ~= "^\(stbmv\|dtbmv\|ctbmv\|ztbmv\|\
		       cblas_stbmv\|cblas_dtbmv\|cblas_ctbmv\|cblas_ztbmv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs15@
cbc << cbcall15.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall15.cbc;
identifier cbcs15.init_val;
identifier cbcall15.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall16@
position cbc;
identifier CblasCall ~= "^\(strsv\|dtrsv\|ctrsv\|ztrsv\|\
		       cblas_strsv\|cblas_dtrsv\|cblas_ctrsv\|cblas_ztrsv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs16@
cbc << cbcall16.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall16.cbc;
identifier cbcs16.init_val;
identifier cbcall16.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall17@
position cbc;
identifier CblasCall ~= "^\(strsv\|dtrsv\|ctrsv\|ztrsv\|\
		       cblas_strsv\|cblas_dtrsv\|cblas_ctrsv\|cblas_ztrsv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs17@
cbc << cbcall17.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall17.cbc;
identifier cbcs17.init_val;
identifier cbcall17.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall18@
position cbc;
identifier CblasCall ~= "^\(srotg\|drotg\|crotg\|zrotg\|\
		       srotmg\|srot\|sroti\|srotm\|\
		       drotmg\|drot\|drotm\|droti\|\
		       csrot\|zdrot\|cblas_srotg\|cblas_drotg\|cblas_crotg\|cblas_zrotg\|\
		       cblas_srotmg\|cblas_srot\|cblas_sroti\|cblas_srotm\|\
		       cblas_drotmg\|cblas_drot\|cblas_drotm\|cblas_droti\|\
		       cblas_csrot\|cblas_zdrot\)$";
@@


CblasCall@cbc(...);


@script:python cbcs18@
cbc << cbcall18.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall18.cbc;
identifier cbcs18.init_val;
identifier cbcall18.CblasCall;
@@


+ init_val;

CblasCall@cbc(...);


@cbcall19@
position cbc;
identifier CblasCall ~= "^\(sswap\|dswap\|cswap\|zswap\|\
		       cblas_sswap\|cblas_dswap\|cblas_cswap\|cblas_zswap\)$";
@@

CblasCall@cbc(...);

@script:python cbcs19@
cbc << cbcall19.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall19.cbc;
identifier cbcs19.init_val;
identifier cbcall19.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall20@
position cbc;
identifier CblasCall ~= "^\(scopy\|dcopy\|ccopy\|zcopy\|\
		       cblas_scopy\|cblas_dcopy\|cblas_ccopy\|cblas_zcopy\)$";
@@

CblasCall@cbc(...);

@script:python cbcs20@
cbc << cbcall20.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall20.cbc;
identifier cbcs20.init_val;
identifier cbcall20.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall21@
position cbc;
identifier CblasCall ~= "^\(saxpy\|daxpy\|caxpy\|zaxpy\|\
		       cblas_saxpy\|cblas_daxpy\|cblas_caxpy\|cblas_zaxpy\)$";
@@

CblasCall@cbc(...);

@script:python cbcs21@
cbc << cbcall21.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall21.cbc;
identifier cbcs21.init_val;
identifier cbcall21.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall22@
position cbc;
identifier CblasCall ~= "^\(saxpby\|daxpby\|caxpby\|zaxpby\|\
		       cblas_saxpby\|cblas_daxpby\|cblas_caxpby\|cblas_zaxpby\)$";
@@

CblasCall@cbc(...);

@script:python cbcs22@
cbc << cbcall22.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall22.cbc;
identifier cbcs22.init_val;
identifier cbcall22.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall23@
position cbc;
identifier CblasCall ~= "^\(saxpyi\|daxpyi\|caxpyi\|zaxpyi\|\
		       cblas_saxpyi\|cblas_daxpyi\|cblas_caxpyi\|cblas_zaxpyi\)$";
@@

CblasCall@cbc(...);

@script:python cbcs23@
cbc << cbcall23.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall23.cbc;
identifier cbcs23.init_val;
identifier cbcall23.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall24@
position cbc;
identifier CblasCall ~= "^\(sgthr\|dgthr\|cgthr\|zgthr\|\
		       cblas_sgthr\|cblas_dgthr\|cblas_cgthr\|cblas_zgthr\)$";
@@

CblasCall@cbc(...);

@script:python cbcs24@
cbc << cbcall24.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall24.cbc;
identifier cbcs24.init_val;
identifier cbcall24.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall25@
position cbc;
identifier CblasCall ~= "^\(sgthrz\|dgthrz\|cgthrz\|zgthrz\|\
		       cblas_sgthrz\|cblas_dgthrz\|cblas_cgthrz\|cblas_zgthrz\)$";
@@

CblasCall@cbc(...);

@script:python cbcs25@
cbc << cbcall25.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall25.cbc;
identifier cbcs25.init_val;
identifier cbcall25.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall26@
position cbc;
identifier CblasCall ~= "^\(ssctr\|dsctr\|csctr\|zsctr\|\
		       cblas_ssctr\|cblas_dsctr\|cblas_csctr\|cblas_zsctr\)$";
@@

CblasCall@cbc(...);

@script:python cbcs26@
cbc << cbcall26.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall26.cbc;
identifier cbcs26.init_val;
identifier cbcall26.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall27@
position cbc;
identifier CblasCall ~= "^\(ssymv\|dsymv\|ssbmv\|dsbmv\|sspmv\|dspmv\|\
		       cblas_ssymv\|cblas_dsymv\|cblas_ssbmv\|cblas_dsbmv\|cblas_sspmv\|cblas_dspmv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs27@
cbc << cbcall27.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall27.cbc;
identifier cbcs27.init_val;
identifier cbcall27.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall28@
position cbc;
identifier CblasCall ~= "^\(chpmv\|chemv\|chbmv\|zhemv\|zhbmv\|zhpmv\|\
		       cblas_chpmv\|cblas_chemv\|cblas_chbmv\|cblas_zhemv\|cblas_zhbmv\|cblas_zhpmv\)$";
@@

CblasCall@cbc(...);

@script:python cbcs28@
cbc << cbcall28.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall28.cbc;
identifier cbcs28.init_val;
identifier cbcall28.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall29@
position cbc;
identifier CblasCall ~= "^\(sger\|dger\|cgeru\|zgeru\|zgerc\|cgerc\|\
			cblas_sger\|cblas_dger\|cblas_cgeru\|cblas_zgeru\|cblas_zgerc\|cblas_cgerc\)$";
@@

CblasCall@cbc(...);

@script:python cbcs29@
cbc << cbcall29.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall29.cbc;
identifier cbcs29.init_val;
identifier cbcall29.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall30@
position cbc;
identifier CblasCall ~= "^\(ssyr\|dsyr\|ssyr2\|dsyr2\|\
		       cblas_ssyr\|cblas_dsyr\|cblas_ssyr2\|cblas_dsyr2\)$";
@@

CblasCall@cbc(...);

@script:python cbcs30@
cbc << cbcall30.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall30.cbc;
identifier cbcs30.init_val;
identifier cbcall30.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall31@
position cbc;
identifier CblasCall ~= "^\(sspr\|dspr\|sspr2\|dspr2\|\
		       cblas_sspr\|cblas_dspr\|cblas_sspr2\|cblas_dspr2\)$";
@@

CblasCall@cbc(...);

@script:python cbcs31@
cbc << cbcall31.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall31.cbc;
identifier cbcs31.init_val;
identifier cbcall31.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall32@
position cbc;
identifier CblasCall ~= "^\(shpr\|dhpr\|shpr2\|dhpr2\|\
		       cblas_shpr\|cblas_dhpr\|cblas_shpr2\|cblas_dhpr2\)$";
@@

CblasCall@cbc(...);

@script:python cbcs32@
cbc << cbcall32.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall32.cbc;
identifier cbcs32.init_val;
identifier cbcall32.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);

@cbcall33@
position cbc;
identifier CblasCall ~= "^\(cher\|cher2\|zher\|zher2\|\
		       cblas_cher\|cblas_cher2\|cblas_zher\|cblas_zher2\)$";
@@

CblasCall@cbc(...);

@script:python cbcs33@
cbc << cbcall33.cbc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position cbcall33.cbc;
identifier cbcs33.init_val;
identifier cbcall33.CblasCall;
@@


+ init_val;
CblasCall@cbc(...);











