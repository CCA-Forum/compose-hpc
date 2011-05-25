AC_PREREQ(2.64)
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

LLNL_CHECK_AUTOMAKE([1.11])
LLNL_CHECK_AUTOCONF([2.64])

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
AC_PROG_LN_S
AC_PROG_INSTALL
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${prefix}/lib/pkgconfig
PKG_PROG_PKG_CONFIG
############################################################
#
# Babel
#
AC_MSG_CHECKING([for babel])
PKG_CHECK_EXISTS([babel >= 2.0.0], 
    [AC_SUBST([HAVE_BABEL], 1)
     AC_SUBST([BABEL_PREFIX], [`$PKG_CONFIG babel --variable prefix`])
     AC_SUBST([SIDL_PATH], ["$BABEL_PREFIX/share"])
     AC_SUBST([BABEL_SRCDIR], 
              [`$BABEL_PREFIX/bin/babel-config --query-var=abs_top_srcdir`])
     AC_MSG_RESULT([$BABEL_PREFIX/bin/babel])],
     [AC_MSG_RESULT([no])
      AC_MSG_NOTICE([Did not find Babel version 2 or later! 
Please remember to set PKG_CONFIG_PATH accordingly.])
      AC_SUBST([HAVE_BABEL], 0)])

############################################################
#
# Chapel
#
AC_MSG_CHECKING([for chapel])
CHAPEL=`which chpl`
AS_IF([test -x "${CHAPEL}"],
    [CHAPEL_VERSION=`$CHAPEL --version |grep Version |awk '{print \$3}'`
     AC_MSG_RESULT([${CHAPEL} (version ${CHAPEL_VERSION})])
     AC_SUBST([CHAPEL], ["${CHAPEL}"])
     AC_SUBST([CHAPEL_ROOT], 
              ["`dirname \`dirname \\\`dirname ${CHAPEL}\\\`\``"])
     AC_SUBST([CHAPEL_VERSION], ["${CHAPEL_VERSION}"])],
    [AC_MSG_RESULT([no])
     AC_MSG_NOTICE([Did not find Chapel!])
      AC_SUBST([CHAPEL], ['echo "**ERROR: No Chapel compiler" ; false'])])

############################################################
#
# SWI-Prolog (compile-time-only dependency)
#
AC_MSG_CHECKING([for swipl])
PKG_CHECK_EXISTS([pl >= 5.8.0],
    [AC_SUBST([SWIPL_PREFIX], [`$PKG_CONFIG pl --variable prefix`])
     AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/swipl"])
     AC_MSG_RESULT([${SWIPL}])
    ],
    [PKG_CHECK_EXISTS([swipl >= 5.8.0], 
        [AC_SUBST([SWIPL_PREFIX], [`$PKG_CONFIG swipl --variable prefix`])
         AC_SUBST([SWIPL], ["${SWIPL_PREFIX}/bin/swipl"])
	],
	[AC_MSG_RESULT([no])
         AC_MSG_NOTICE([Did not find SWI-Prolog version 5.8 or later! 
Please remember to set PKG_CONFIG_PATH accordingly.])
         AC_SUBST([SWIPL], [""])
        ])
    ])

AC_CONFIG_FILES([Makefile braid/braid braid/config.py Doxyfile regression/interop/driver.sh])

AC_OUTPUT

