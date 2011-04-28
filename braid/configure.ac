AC_PREREQ(2.64)
AC_INIT([Braid],[0.2.1],[adrian@llnl.gov])
AC_CONFIG_SRCDIR([COPYRIGHT])
AC_CONFIG_MACRO_DIR([m4])
CCA_DIAGNOSE_INIT(toolcheck.log)
AC_CONFIG_AUX_DIR([config])
AC_CANONICAL_TARGET

AM_INIT_AUTOMAKE([foreign no-dependencies no-installinfo no-installman no-texinfo.tex 1.11 dist-bzip2 parallel-tests color-tests silent-rules std-options])
dnl do `make V=1' or `./configure --disable-silent-rules' for a verbose build
AM_SILENT_RULES([yes])
dnl To regenerate configure and .in files, configure with --enable-maintainer-mode
AM_MAINTAINER_MODE

dnl AC_CONFIG_HEADERS(runtime/sidl/ignore_me.h)
AC_DEFINE([BABEL_VERSION_HEX],[0x01099000],
	[Hexified form of release number: 2 hex digits each for major, minor, bugfix, and alpha/beta/candidate/distributed ])

LLNL_CHECK_AUTOMAKE([1.11])
LLNL_CHECK_AUTOCONF([2.64])

############################################################
#
# Additional configure options.
#
LLNL_CROSS_COMPILING
LLNL_AUTO_DISABLE
LLNL_ENABLE_REGRESSION
LLNL_ENABLE_DOCUMENTATION

############################################################
#
# Python Support.
#
AM_PATH_PYTHON([2.6])
AM_CONDITIONAL([PYTHON_VERSION_2_6], [test x"$PYTHON_VERSION" == x2.6])
CCA_DIAGNOSE_BEGIN([PYTHON])
LLNL_CONFIRM_BABEL_PYTHON_SUPPORT
CCA_DIAGNOSE_END([PYTHON])

AC_SUBST([SWIPL], [`which swipl`])

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
PKG_CHECK_EXISTS([babel >= 2.0.0], 
    [AC_SUBST([HAVE_BABEL], 1)
     AC_SUBST([BABEL_PREFIX], [`$PKG_CONFIG babel --variable prefix`])
     AC_SUBST([SIDL_PATH], ["$BABEL_PREFIX/share"])
     AC_MSG_RESULT([Found Babel])],
     [AC_MSG_NOTICE([Did not find Babel version 2 or later! Please remember to set PKG_CONFIG_PATH accordingly.])
      AC_SUBST([HAVE_BABEL], 0)])


AC_CONFIG_FILES([Makefile braid/braid braid/config.py Doxyfile])

AC_OUTPUT

