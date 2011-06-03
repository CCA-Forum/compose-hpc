AC_PREREQ(2.65)
AC_INIT([Braid],[0.2.1],[components@llnl.gov],
        [braid],[http://compose-hpc.sourceforge.net/])
AC_CONFIG_SRCDIR([COPYRIGHT])
AC_CONFIG_MACRO_DIR([m4])
CCA_DIAGNOSE_INIT(toolcheck.log)
AC_CONFIG_AUX_DIR([config])
AC_CANONICAL_TARGET

AM_INIT_AUTOMAKE([foreign no-dependencies no-installinfo no-installman 
  no-texinfo.tex 1.11 dist-bzip2 parallel-tests color-tests silent-rules 
  std-options])
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
# Python Support.
#
AM_PATH_PYTHON([2.6])
AM_CONDITIONAL([PYTHON_VERSION_2_6], [test x"$PYTHON_VERSION" == x2.6])
CCA_DIAGNOSE_BEGIN([PYTHON])
LLNL_LIBXML_CONFIG
LLNL_CONFIRM_BABEL_PYTHON_SUPPORT
CCA_DIAGNOSE_END([PYTHON])

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
AS_IF([test -x "$PERL"], [], AC_MSG_ERROR([perl is missing]))


# DOXYGEN
AX_WITH_PROG(DOXYGEN, doxygen) # used as sed replacement in the testsuite
AS_IF([test -x "$DOXYGEN"], 
      [DOXYGEN_VERSION=`$DOXYGEN --version`
       AX_COMPARE_VERSION([$DOXYGEN_VERSION], ge, [1.6.0], 
           [],
           [AC_SUBST([DOXYGEN], [])
	    AM_CONDITIONAL(ENABLE_DOCUMENTATION, false)
            AC_MSG_NOTICE([doxygen < 1.6 is too old])])
      ], 
      AC_MSG_NOTICE([doxygen is missing]))


export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${prefix}/lib/pkgconfig
############################################################
#
# Babel
#
PKG_CHECK_MODULES(BABEL, [babel >= 2.0.0], 
    [AC_SUBST([HAVE_BABEL], 1)
     AC_SUBST([BABEL_PREFIX], [`$PKG_CONFIG babel --variable prefix`])
     AC_SUBST([SIDL_PATH], ["$BABEL_PREFIX/share"])
     AC_SUBST([BABEL_SRCDIR], 
              [`$BABEL_PREFIX/bin/babel-config --query-var=abs_top_srcdir`])
     AC_MSG_RESULT([using $BABEL_PREFIX/bin/babel])],
     [AC_MSG_NOTICE([Did not find Babel version 2 or later! 
Please remember to set PKG_CONFIG_PATH accordingly.])
      AC_SUBST([HAVE_BABEL], 0)])

############################################################
#
# Chapel
#
AX_WITH_PROG(CHAPEL, chpl)
AS_IF([test -x "${CHAPEL}"],
    [CHAPEL_VERSION=`$CHAPEL --version |grep Version |awk '{print \$3}'`
     AC_MSG_RESULT([using $CHAPEL (version $CHAPEL_VERSION)])
     AX_COMPARE_VERSION([$CHAPEL_VERSION], ge, [1.3.0], 
         [AC_SUBST([CHAPEL], ["$CHAPEL"])
          AC_SUBST([CHAPEL_ROOT], 
                   ["`dirname \`dirname \\\`dirname $CHAPEL\\\`\``"])
          AC_SUBST([CHAPEL_VERSION], ["$CHAPEL_VERSION"])],
	 [AC_MSG_NOTICE([Chapel version < 1.3 is too old])
          AC_SUBST([CHAPEL], ['echo "**ERROR: No Chapel compiler" ; false'])])
    ],
    [AC_MSG_NOTICE([Did not find Chapel compiler in PATH!])
      AC_SUBST([CHAPEL], ['echo "**ERROR: No Chapel compiler" ; false'])])

############################################################
#
# SWI-Prolog (compile-time-only dependency)
#
PKG_CHECK_MODULES(PL, [pl >= 5.8.0],
    [AC_SUBST([SWIPL_PREFIX], [`$PKG_CONFIG pl --variable prefix`])
     AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/swipl"])
     AC_MSG_RESULT([using ${SWIPL}])
    ],
    [# alternate pkg-config name
     PKG_CHECK_MODULES(SWIPL, [swipl >= 5.8.0], 
        [AC_SUBST([SWIPL_PREFIX], [`$PKG_CONFIG swipl --variable prefix`])
         AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/swipl"])
	],
	[AC_MSG_NOTICE([Did not find SWI-Prolog version 5.8 or later! 
Please remember to set PKG_CONFIG_PATH accordingly.])
         AC_SUBST([SWIPL], [""])
        ])
    ])

AM_CONDITIONAL(ENABLE_INTEROP_REGRESSION, 
               [test $HAVE_BABEL -eq 1 && test -n $CHAPEL_VERSION])

AC_CONFIG_FILES([Makefile 
  braid/braid 
  braid/config.py 
  Doxyfile
  regression/interop/build-tracker.sh
  regression/interop/driver.sh
])

AC_OUTPUT

