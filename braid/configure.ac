AC_PREREQ(2.65)
dnl gantlet expects the following statement to be exactly one line.
AC_INIT([Braid],[0.2.6],[components@llnl.gov],[braid],[http://compose-hpc.sourceforge.net/])
AC_CONFIG_SRCDIR([COPYRIGHT])
AC_CONFIG_MACRO_DIR([m4])
CCA_DIAGNOSE_INIT(toolcheck.log)
AC_CONFIG_AUX_DIR([config])
AC_CANONICAL_TARGET

AM_INIT_AUTOMAKE([foreign 1.11 dist-bzip2 dist-xz parallel-tests
  color-tests silent-rules std-options])
dnl do `make V=1' or `./configure --disable-silent-rules' for a verbose build
AM_SILENT_RULES([yes])
dnl To regenerate configure and .in files, configure with --enable-maintainer-mode
AM_MAINTAINER_MODE

dnl AC_CONFIG_HEADERS(runtime/sidl/ignore_me.h)
AC_DEFINE([BRAID_VERSION_HEX],[0x00021000],
	[Hexified form of release number: 2 hex digits each for major, minor, bugfix, and alpha/beta/candidate/distributed ])

############################################################
#
# Additional configure options.
#
dnl LLNL_CROSS_COMPILING
dnl LLNL_AUTO_DISABLE
LLNL_ENABLE_REGRESSION
LLNL_ENABLE_DOCUMENTATION


############################################################
#
# Critical Applications.
#
AC_PROG_CC
AC_PROG_LEX
AC_PROG_INSTALL
AC_PROG_CPP
AC_PROG_AWK
AC_PROG_SED
AC_PROG_LN_S
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL
AC_MSG_CHECKING([for perl])
AX_WITH_PROG(PERL, perl) # used as sed replacement in the testsuite
AS_IF([test -n "$PERL" -a -x "$PERL"], [], AC_MSG_ERROR([perl is missing]))

############################################################
#
# Python Support.
#
AC_ARG_ENABLE([doc_only], 
   AS_HELP_STRING([--enable-doc-only], 
   [use this if you *only* want to run `make doc' and don't have a recent enough python installation]),
   [AM_PATH_PYTHON([2.4],, [:])], 
   [AM_PATH_PYTHON([2.6])])
AC_SUBST([CONFIGURED_PYTHONPATH], ["$PYTHONPATH"])
AS_IF([test x"$PYTHON_VERSION" == x2.6])
AM_CONDITIONAL([PYTHON_VERSION_2_6], [test x"$PYTHON_VERSION" == x2.6])
CCA_DIAGNOSE_BEGIN([PYTHON])
LLNL_LIBXML_CONFIG
LLNL_CONFIRM_BABEL_PYTHON_SUPPORT
CCA_DIAGNOSE_END([PYTHON])

# DOXYGEN
AX_WITH_PROG(DOXYGEN, doxygen)
AS_IF([test -n "$DOXYGEN" -a -x "$DOXYGEN"], 
      [DOXYGEN_VERSION=`$DOXYGEN --version`
       AX_COMPARE_VERSION([$DOXYGEN_VERSION], ge, [1.6.0], 
           [],
           [AC_SUBST([DOXYGEN], [])
	    AM_CONDITIONAL(ENABLE_DOCUMENTATION, false)
            AC_MSG_NOTICE([doxygen < 1.6 is too old])])
      ], 
      AC_MSG_NOTICE([doxygen is missing]))

# DOT
AX_WITH_PROG(DOT, dot)
AS_IF([test -n "$DOT" -a -x "$DOT"], 
      [:], 
      [AM_CONDITIONAL(ENABLE_DOCUMENTATION, false)
       AC_MSG_NOTICE([dot is missing -> documentation was disabled])])


export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${prefix}/lib/pkgconfig
############################################################
#
# Babel
#
PKG_CHECK_MODULES(BABEL, [babel >= 2.0.0], 
    [AC_SUBST([HAVE_BABEL], 1)
     AC_SUBST([BABEL_PREFIX], [`$PKG_CONFIG babel --variable prefix`])
     AC_SUBST([SIDL_PATH], ["$BABEL_PREFIX/share"])
     AC_SUBST([BABEL_SRCDIR],  [`$BABEL_PREFIX/bin/babel-config --query-var=abs_top_srcdir`])
     AC_SUBST([BABEL_INCLUDEDIR],  [`$BABEL_PREFIX/bin/babel-config --query-var=includedir`])
     AC_SUBST([BABEL_VERSION], [`$BABEL_PREFIX/bin/babel-config --query-var=VERSION`])
     BABEL_LANGS=`$BABEL_PREFIX/bin/babel-config --query-var=BABEL_SUPPORTED_LANGUAGES`
     AC_SUBST([BABEL_SUPPORTED_LANGUAGES], [$BABEL_LANGS])
     AC_MSG_RESULT([using $BABEL_PREFIX/bin/babel])],
     [AC_MSG_NOTICE([Did not find Babel version 2 or later! 
Please remember to set PKG_CONFIG_PATH accordingly.])
      AC_SUBST([HAVE_BABEL], 0)])

AM_CONDITIONAL([BABEL_LANG_C], [echo $BABEL_LANGS |grep -q c])
AM_CONDITIONAL([BABEL_LANG_CXX], [echo $BABEL_LANGS |grep -q cxx])
AM_CONDITIONAL([BABEL_LANG_F77], [echo $BABEL_LANGS |grep -q f77])
AM_CONDITIONAL([BABEL_LANG_F90], [echo $BABEL_LANGS |grep -q f90])
AM_CONDITIONAL([BABEL_LANG_F03], [echo $BABEL_LANGS |grep -q f03])
AM_CONDITIONAL([BABEL_LANG_JAVA], [echo $BABEL_LANGS |grep -q java])
AM_CONDITIONAL([BABEL_LANG_PYTHON], [echo $BABEL_LANGS |grep -q python])

############################################################
#
# Chapel
#
AX_WITH_PROG(CHPL, chpl)
AS_IF([test -n "$CHPL" -a -x "$CHPL"],
    [CHPL_VERSION=`$CHPL --version |grep Version |$AWK '{print \$3}'`
     AC_MSG_RESULT([using $CHPL (version $CHPL_VERSION)])

     AX_COMPARE_VERSION([$CHPL_VERSION], ge, [1.6.0], 
         [AC_SUBST([CHPL], ["$CHPL"])
 	  CHPL_ROOT=`dirname \`dirname \\\`dirname $CHPL\\\`\``
          AC_SUBST([CHPL_ROOT], [$CHPL_ROOT])
          AC_SUBST([CHPL_VERSION], ["$CHPL_VERSION"])],
	 [AC_MSG_NOTICE([Chapel version < 1.6 is too old])
          AC_SUBST([CHPL], [])]
     )
     AS_IF([test -n "$CHPL_COMM" ],
          [AC_SUBST([CHPL_COMM], [$CHPL_COMM])
           AC_MSG_NOTICE([Chapel: CHPL_COMM=$CHPL_COMM])],
	  [AC_SUBST([CHPL_COMM], ["none"])
           AC_MSG_NOTICE([
========================================================================
The environment variable CHPL_COMM was not set, defaulting to "none".
Please see the file $CHPL_ROOT/doc/README.multilocale for more details.
========================================================================
])
	  ])
    ],
    [AC_MSG_NOTICE([Did not find Chapel compiler in PATH!])
     AC_SUBST([CHPL], [])])

AM_CONDITIONAL([BABEL_LANG_CHPL], [test -n "$CHPL"])

############################################################
#
# UPC
#
AM_PROG_UPC
AS_IF([test -n "$UPC"], 
      [UPC_ROOT=`dirname \`dirname \\\`which $UPC\\\`\``
       AC_SUBST([UPC_ROOT], [$UPC_ROOT])
       AS_IF([$UPC --version |head -n 1 |grep -q Berkeley],
             [
              ###########################################
              # Berkeley UPC
              ###########################################
	      AC_SUBST([UPC_VERSION],  [`$UPC --version |head -n 1 |$AWK '{ print $NF }'`])
	      AC_MSG_NOTICE([detected Berkeley UPC version $UPC_VERSION])
	      AC_SUBST([UPC_CFLAGS],   [`$UPC $UPCFLAGS --echo-var GASNET_CFLAGS`])
	      AC_SUBST([UPC_CPPFLAGS], [`$UPC $UPCFLAGS --echo-var UPCR_CPPFLAGS`])
	      AC_SUBST([UPC_LDFLAGS],  [`$UPC $UPCFLAGS --echo-var UPCR_LDFLAGS`])
	      AC_SUBST([UPC_LIBS],     [`$UPC $UPCFLAGS --echo-var UPCR_LIBS`])
	      AC_SUBST([UPCC_POSTTRANS_CPPFLAGS], [`$UPC $UPC_FLAGS --echo-var UPCC_POSTTRANS_CPPFLAGS`])

	      # compile an empty file and extract the auto-generated
	      # startup code for inclusion in the braid upc runtime
	      AC_MSG_CHECKING([UPC runtime startup code])
	      AC_LANG([C])
	      AC_LANG_CONFTEST([AC_LANG_SOURCE([int main(int argc, char** argv) { return 0; }])])
	      mv conftest.c upc_stub.upc || AC_MSG_FAILURE([no conftest?])
	      $UPC $UPCFLAGS --save-all-temps upc_stub.upc -o upc_stub || AC_MSG_FAILURE([invoking $UPC])
	      $MKDIR_P upc || AC_MSG_FAILURE([mkdir])
	      UPC_RUNTIME_FILES="upc/upc_stub_startup_tmp.c upc/upcr_trans_extra.c upc/upc_stub.trans.c"
	      mv upc_stub_temps/upc_stub_startup_tmp.c \
	         upc_stub_temps/upcr_trans_extra.c \
		 upc_stub_temps/upc_stub.trans.c \
		 upc/ || AC_MSG_FAILURE([
Unexpected startup code layout. You are probably an unsupported version of UPC. 
Please see the README for a list of supported versions.])
       	      $RM -r upc_stub* || AC_MSG_FAILURE([removing temp. files])
	      $SED -i 's/user_main/_braid_disabled/' upc/upc_stub.trans.c || AC_MSG_FAILURE([sed])
	      AC_MSG_RESULT([ok])
	      ], [
	       ###########################################
	       # Unknown UPC implementation
               ###########################################
	       AC_SUBST([UPC_VERSION], [`$UPC --version`])
	       AC_MSG_WARN([$UPC is not yet fully supported by BRAID. Right now BRAID only supports the Berkeley UPC compiler.])
              ])
      ],
      [AC_MSG_NOTICE([Did not find UPC compiler in PATH!])
       AC_SUBST([UPC], [])
      ])
AM_CONDITIONAL([BABEL_LANG_UPC], [test -n "$UPC"])

############################################################
#
# SWI-Prolog (compile-time-only dependency)
#
PKG_CHECK_MODULES(PL, [pl >= 5.8.0],
    [AC_SUBST([SWIPL_PREFIX], [`$PKG_CONFIG pl --variable prefix`])
     AS_IF([test -x $SWIPL_PREFIX/bin/swipl],
           [AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/swipl"])],
           [AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/pl"])])
     AC_SUBST([SWIPL_VERSION], [`${SWIPL} --version | $AWK '{print $3}'`])
     AC_MSG_RESULT([using ${SWIPL}])
    ],
    [# alternate pkg-config name
     PKG_CHECK_MODULES(SWIPL, [swipl >= 5.8.0], 
        [AC_SUBST([SWIPL_PREFIX], [`$PKG_CONFIG swipl --variable prefix`])
         AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/swipl"])
         AC_SUBST([SWIPL_VERSION], [`${SWIPL} --version | $AWK '{print $3}'`])
	],
	[AC_MSG_NOTICE([Did not find SWI-Prolog version 5.8 or later! 
Please remember to set PKG_CONFIG_PATH accordingly.
Note: You won't need to have this installed 
      unless you intend to modify the IR grammar.])
         AC_SUBST([SWIPL], [""])
        ])
    ])

AC_SUBST([BUILD_MACHINE], ["`hostname`"])
AC_SUBST([CONFIGURE_DATE], ["`date`"])

AM_CONDITIONAL(ENABLE_INTEROP_REGRESSION, 
               [test $HAVE_BABEL -eq 1 && test -n $CHPL_VERSION])

dnl this needs to come before Makefile is generated!
AS_IF([test -n "${BABEL_PREFIX}"],
      [echo "Generating dependencies for regression test suite"
       sh $srcdir/regression/interop/gen-test-deps.sh $BABEL_SRCDIR $srcdir \
         >$srcdir/regression/interop/tests.mk 2>/dev/null
       grep "^# Total" $srcdir/regression/interop/tests.mk])

sh $srcdir/gen-runtime-deps.sh $srcdir
echo "UPC_RUNTIME_EXTRA=$UPC_RUNTIME_FILES">>${srcdir}/upc/Makefile.include

AC_CONFIG_FILES([Makefile 
  braid/braid 
  braid/rote2braid 
  braid/config.py 
  braid/pyburg 
  Doxyfile
  patch-chapel-compiler.sh
  regression/interop/build-client.sh
  regression/interop/build-server.sh
  regression/interop/build-tracker.sh
  regression/interop/driver.sh
], [chmod +x patch-chapel-compiler.sh])

AC_OUTPUT

AC_MSG_RESULT([==============================================================])
AC_MSG_RESULT([BRAID Configuration Summary                                   ])
AC_MSG_RESULT([==============================================================])
AS_IF([test -z "${SWIPL}"], 
      AC_MSG_RESULT([    SWI-Prolog ....... NO]),
      AC_MSG_RESULT([    SWI-Prolog ...... YES (${SWIPL_VERSION} in ${SWIPL})]))
AS_IF([test -z "${BABEL_PREFIX}"], 
      AC_MSG_RESULT([    Babel ............ NO]),
      AC_MSG_RESULT([    Babel ........... YES (${BABEL_VERSION} in ${BABEL_PREFIX})]))
AS_IF([test -z "${CHPL_ROOT}"], 
      AC_MSG_RESULT([    Chapel ........... NO]),
      [AC_MSG_RESULT([    Chapel .......... YES (${CHPL_VERSION} in ${CHPL_ROOT})])
       AC_MSG_RESULT([  **You may need to patch you Chapel compiler now by running ./patch-chapel-compiler.sh])])
AS_IF([test -z "${UPC}"], 
      AC_MSG_RESULT([    UPC .............. NO]),
      [AC_MSG_RESULT([    UPC ............. YES (${UPC_VERSION} in ${UPC_ROOT})])])
AC_MSG_RESULT([==============================================================])
AC_MSG_RESULT([BRAID configured successfully!])
