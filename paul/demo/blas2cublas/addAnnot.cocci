// Script that adds annotations to all blas calls in a source file.

@initialize:python@

count = 0

@bcall@
position bc;
identifier blasCall ~= "^\(sgemm\|ssymm\|ssyrk\|ssyr2k\|\
		       strmm\|strsm\|dgemm\|dsymm\|dsyrk\|\
		       dsyr2k\|dtrmm\|dtrsm\|cgemm\|\
		       cgemm3m\|csymm\|csyrk\|csyr2k\|\
		       ctrmm\|ctrsm\|zgemm\|zgemm3m\|\
		       zsymm\|zsyrk\|zsyr2k\|ztrmm\|\
		       ztrsm\|chemm\|cherk\|cher2k\|\
		       zhemm\|zherk\|zher2k\|\
		       dcabs1\|scabs1\|sdot\|\
		       sdoti\|ddot\|ddoti\|dsdot\|\
		       sdsdot\|cdotu_sub\|cdotui_sub\|\
		       cdotc_sub\|cdotci_sub\|zdotu_sub\|\
		       zdotui_sub\|zdotc_sub\|zdotci_sub\|\
		       snrm2\|sasum\|dnrm2\|dasum\|\
		       scnrm2\|scasum\|dznrm2\|dzasum\|\
		       isamax\|idamax\|icamax\|izamax\|\
		       isamin\|idamin\|icamin\|izamin\|\
		       sswap\|scopy\|saxpy\|saxpby\|\
		       saxpyi\|sgthr\|sgthrz\|ssctr\|\
		       srotg\|dswap\|dcopy\|daxpy\|\
		       daxpby\|daxpyi\|dgthr\|dgthrz\|\
		       dsctr\|drotg\|cswap\|ccopy\|\
		       caxpy\|caxpby\|caxpyi\|cgthr\|\
		       cgthrz\|csctr\|crotg\|zswap\|\
		       zcopy\|zaxpy\|zaxpby\|zaxpyi\|\
		       zgthr\|zgthrz\|zsctr\|zrotg\|\
		       srotmg\|srot\|sroti\|srotm\|\
		       drotmg\|drot\|drotm\|droti\|\
		       csrot\|zdrot\|sscal\|dscal\|\
		       cscal\|zscal\|csscal\|zdscal\|\
		       sgemv\|sgbmv\|strmv\|stbmv\|\
		       stpmv\|strsv\|stbsv\|stpsv\|\
		       dgemv\|dgbmv\|dtrmv\|dtbmv\|\
		       dtpmv\|dtrsv\|dtbsv\|dtpsv\|\
		       cgemv\|cgbmv\|ctrmv\|ctbmv\|\
		       ctpmv\|ctrsv\|ctbsv\|ctpsv\|\
		       zgemv\|zgbmv\|ztrmv\|ztbmv\|\
		       ztpmv\|ztrsv\|ztbsv\|ztpsv\|\
		       ssymv\|ssbmv\|sspmv\|sger\|\
		       ssyr\|sspr\|ssyr2\|sspr2\|\
		       dsymv\|dsbmv\|dspmv\|dger\|\
		       dsyr\|dspr\|dsyr2\|dspr2\|\
		       chemv\|chbmv\|chpmv\|cgeru\|\
		       cgerc\|cher\|cher2\|chpr\|\
		       chpr2\|zhemv\|zhbmv\|zhpmv\|\
		       zgeru\|zgerc\|zher\|zhpr\|\
		       zher2\|zhpr2\)$";

@@

blasCall@bc(...);

@script:python bcs@
bc << bcall.bc;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position bcall.bc;
identifier bcs.init_val;
identifier bcall.blasCall;
@@

+ init_val;
blasCall@bc(...);


@cbcall@
position cbc;
identifier CblasCall ~= "^\(cblas_sgemm\|cblas_ssymm\|cblas_ssyrk\|cblas_ssyr2k\|\
		       cblas_strmm\|cblas_strsm\|cblas_dgemm\|cblas_dsymm\|cblas_dsyrk\|\
		       cblas_dsyr2k\|cblas_dtrmm\|cblas_dtrsm\|cblas_cgemm\|\
		       cblas_cgemm3m\|cblas_csymm\|cblas_csyrk\|cblas_csyr2k\|\
		       cblas_ctrmm\|cblas_ctrsm\|cblas_zgemm\|cblas_zgemm3m\|\
		       cblas_zsymm\|cblas_zsyrk\|cblas_zsyr2k\|cblas_ztrmm\|\
		       cblas_ztrsm\|cblas_chemm\|cblas_cherk\|cblas_cher2k\|\
		       cblas_zhemm\|cblas_zherk\|cblas_zher2k\|\
		       cblas_dcabs1\|cblas_scabs1\|cblas_sdot\|\
		       cblas_sdoti\|cblas_ddot\|cblas_ddoti\|cblas_dsdot\|\
		       cblas_sdsdot\|cblas_cdotu_sub\|cblas_cdotui_sub\|\
		       cblas_cdotc_sub\|cblas_cdotci_sub\|cblas_zdotu_sub\|\
		       cblas_zdotui_sub\|cblas_zdotc_sub\|cblas_zdotci_sub\|\
		       cblas_snrm2\|cblas_sasum\|cblas_dnrm2\|cblas_dasum\|\
		       cblas_scnrm2\|cblas_scasum\|cblas_dznrm2\|cblas_dzasum\|\
		       cblas_isamax\|cblas_idamax\|cblas_icamax\|cblas_izamax\|\
		       cblas_isamin\|cblas_idamin\|cblas_icamin\|cblas_izamin\|\
		       cblas_sswap\|cblas_scopy\|cblas_saxpy\|cblas_saxpby\|\
		       cblas_saxpyi\|cblas_sgthr\|cblas_sgthrz\|cblas_ssctr\|\
		       cblas_srotg\|cblas_dswap\|cblas_dcopy\|cblas_daxpy\|\
		       cblas_daxpby\|cblas_daxpyi\|cblas_dgthr\|cblas_dgthrz\|\
		       cblas_dsctr\|cblas_drotg\|cblas_cswap\|cblas_ccopy\|\
		       cblas_caxpy\|cblas_caxpby\|cblas_caxpyi\|cblas_cgthr\|\
		       cblas_cgthrz\|cblas_csctr\|cblas_crotg\|cblas_zswap\|\
		       cblas_zcopy\|cblas_zaxpy\|cblas_zaxpby\|cblas_zaxpyi\|\
		       cblas_zgthr\|cblas_zgthrz\|cblas_zsctr\|cblas_zrotg\|\
		       cblas_srotmg\|cblas_srot\|cblas_sroti\|cblas_srotm\|\
		       cblas_drotmg\|cblas_drot\|cblas_drotm\|cblas_droti\|\
		       cblas_csrot\|cblas_zdrot\|cblas_sscal\|cblas_dscal\|\
		       cblas_cscal\|cblas_zscal\|cblas_csscal\|cblas_zdscal\|\
		       cblas_sgemv\|cblas_sgbmv\|cblas_strmv\|cblas_stbmv\|\
		       cblas_stpmv\|cblas_strsv\|cblas_stbsv\|cblas_stpsv\|\
		       cblas_dgemv\|cblas_dgbmv\|cblas_dtrmv\|cblas_dtbmv\|\
		       cblas_dtpmv\|cblas_dtrsv\|cblas_dtbsv\|cblas_dtpsv\|\
		       cblas_cgemv\|cblas_cgbmv\|cblas_ctrmv\|cblas_ctbmv\|\
		       cblas_ctpmv\|cblas_ctrsv\|cblas_ctbsv\|cblas_ctpsv\|\
		       cblas_zgemv\|cblas_zgbmv\|cblas_ztrmv\|cblas_ztbmv\|\
		       cblas_ztpmv\|cblas_ztrsv\|cblas_ztbsv\|cblas_ztpsv\|\
		       cblas_ssymv\|cblas_ssbmv\|cblas_sspmv\|cblas_sger\|\
		       cblas_ssyr\|cblas_sspr\|cblas_ssyr2\|cblas_sspr2\|\
		       cblas_dsymv\|cblas_dsbmv\|cblas_dspmv\|cblas_dger\|\
		       cblas_dsyr\|cblas_dspr\|cblas_dsyr2\|cblas_dspr2\|\
		       cblas_chemv\|cblas_chbmv\|cblas_chpmv\|cblas_cgeru\|\
		       cblas_cgerc\|cblas_cher\|cblas_cher2\|cblas_chpr\|\
		       cblas_chpr2\|cblas_zhemv\|cblas_zhbmv\|cblas_zhpmv\|\
		       cblas_zgeru\|cblas_zgerc\|cblas_zher\|cblas_zhpr\|\
		       cblas_zher2\|cblas_zhpr2\)$";
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

@bcallMisc@
position bcm;
identifier blasCallMisc ~= "^\(sgemm_\|ssymm_\|ssyrk_\|ssyr2k_\|\
		       strmm_\|strsm_\|dgemm_\|dsymm_\|dsyrk_\|\
		       dsyr2k_\|dtrmm_\|dtrsm_\|cgemm_\|\
		       cgemm3m_\|csymm_\|csyrk_\|csyr2k_\|\
		       ctrmm_\|ctrsm_\|zgemm_\|zgemm3m_\|\
		       zsymm_\|zsyrk_\|zsyr2k_\|ztrmm_\|\
		       ztrsm_\|chemm_\|cherk_\|cher2k_\|\
		       zhemm_\|zherk_\|zher2k_\|\
		       dcabs1_\|scabs1_\|sdot_\|\
		       sdoti_\|ddot_\|ddoti_\|dsdot_\|\
		       sdsdot_\|cdotu_sub_\|cdotui_sub_\|\
		       cdotc_sub_\|cdotci_sub_\|zdotu_sub_\|\
		       zdotui_sub_\|zdotc_sub_\|zdotci_sub_\|\
		       snrm2_\|sasum_\|dnrm2_\|dasum_\|\
		       scnrm2_\|scasum_\|dznrm2_\|dzasum_\|\
		       isamax_\|idamax_\|icamax_\|izamax_\|\
		       isamin_\|idamin_\|icamin_\|izamin_\|\
		       sswap_\|scopy_\|saxpy_\|saxpby_\|\
		       saxpyi_\|sgthr_\|sgthrz_\|ssctr_\|\
		       srotg_\|dswap_\|dcopy_\|daxpy_\|\
		       daxpby_\|daxpyi_\|dgthr_\|dgthrz_\|\
		       dsctr_\|drotg_\|cswap_\|ccopy_\|\
		       caxpy_\|caxpby_\|caxpyi_\|cgthr_\|\
		       cgthrz_\|csctr_\|crotg_\|zswap_\|\
		       zcopy_\|zaxpy_\|zaxpby_\|zaxpyi_\|\
		       zgthr_\|zgthrz_\|zsctr_\|zrotg_\|\
		       srotmg_\|srot_\|sroti_\|srotm_\|\
		       drotmg_\|drot_\|drotm_\|droti_\|\
		       csrot_\|zdrot_\|sscal_\|dscal_\|\
		       cscal_\|zscal_\|csscal_\|zdscal_\|\
		       sgemv_\|sgbmv_\|strmv_\|stbmv_\|\
		       stpmv_\|strsv_\|stbsv_\|stpsv_\|\
		       dgemv_\|dgbmv_\|dtrmv_\|dtbmv_\|\
		       dtpmv_\|dtrsv_\|dtbsv_\|dtpsv_\|\
		       cgemv_\|cgbmv_\|ctrmv_\|ctbmv_\|\
		       ctpmv_\|ctrsv_\|ctbsv_\|ctpsv_\|\
		       zgemv_\|zgbmv_\|ztrmv_\|ztbmv_\|\
		       ztpmv_\|ztrsv_\|ztbsv_\|ztpsv_\|\
		       ssymv_\|ssbmv_\|sspmv_\|sger_\|\
		       ssyr_\|sspr_\|ssyr2_\|sspr2_\|\
		       dsymv_\|dsbmv_\|dspmv_\|dger_\|\
		       dsyr_\|dspr_\|dsyr2_\|dspr2_\|\
		       chemv_\|chbmv_\|chpmv_\|cgeru_\|\
		       cgerc_\|cher_\|cher2_\|chpr_\|\
		       chpr2_\|zhemv_\|zhbmv_\|zhpmv_\|\
		       zgeru_\|zgerc_\|zher_\|zhpr_\|\
		       zher2_\|zhpr2_\)$";

@@

blasCallMisc@bcm(...);

@script:python bcms@
bcm << bcallMisc.bcm;
init_val;
@@
count = count + 1
coccinelle.init_val = "/*%c BLAS_TO_CUBLAS prefix=device%d */" % ('%',count)

@@
position bcallMisc.bcm;
identifier bcms.init_val;
identifier bcallMisc.blasCallMisc;
@@

+ init_val;
blasCallMisc@bcm(...);








