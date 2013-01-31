#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.63)
AC_INIT(libpaul, 0.2, matt@galois.com)
AM_INIT_AUTOMAKE([foreign 1.11 dist-bzip2])
dnl do `make V=1' or `./configure --disable-silent-rules' for a verbose build
AM_SILENT_RULES([yes])
dnl To regenerate configure and .in files, configure with --enable-maintainer-mode
AM_MAINTAINER_MODE
AC_OUTPUT([Makefile src/Makefile tests/Makefile])
AC_CONFIG_SRCDIR([src/paul/Dynamic.h])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_MACRO_DIR([m4])

# Checks for programs.
#AC_PROG_YACC
#AS_IF([test "x`which $YACC`" = x],[AC_MSG_FAILURE([Please install bison])])
AC_PROG_CXX
AS_IF([test "x`which $CXX`" = x],[AC_MSG_FAILURE([Please install g++])])
AC_PROG_CC
AS_IF([test "x`which $CC`" = x],[AC_MSG_FAILURE([Please install gcc])])
AC_PROG_LEX
AS_IF([test "x`which $LEX`" = x],[AC_MSG_FAILURE([Please install flex])])

# Some versions of Debian ship with a libtool that defaults to not
# linking dependencies specified by .la libraries. It seems that
# patching libtool is the only viable remedy for this.
AC_PROG_SED
sed -i "s/\(.*link_all_deplibs\s*,\s*\$1\s*)\)\s*=\s*no/\1=unknown/g" m4/libtool.m4
AC_MSG_RESULT([patched libtool to force it to link against all deplibs])
# the patching needs to happen before AC_PROG_LIBTOOL creates ./libtool
AC_PROG_LIBTOOL

# ROSE PATH
AS_IF([test "x`which roseCompiler`" = x],
      [ROSEDIR="/opt/rose"],
      [TMP=`which roseCompiler`
       TMP=`dirname $TMP`
       ROSEDIR=`dirname $TMP`])
AC_ARG_WITH(rosedir,
  [  --with-rosedir=DIR           ROSE install path (default in /opt/rose)],
  [ROSEDIR="$withval"])
AC_SUBST(ROSEDIR)

ROSE_INCLUDE_DIR="$ROSEDIR/include"
AC_ARG_WITH(rose_includedir,
  [  --with-rose-includedir=DIR   ROSE include path (default in /opt/rose/include)],
  [ROSE_INCLUDE_DIR="$withval"])
AC_SUBST(ROSE_INCLUDE_DIR)
AC_DEFINE_UNQUOTED(ROSE_INCLUDE_DIR,["${ROSE_INCLUDE_DIR}"], [Include directory for ROSE headers])

ROSE_LIB_DIR="$ROSEDIR/lib"
AC_ARG_WITH(rose_libdir,
  [  --with-rose-libdir=DIR       ROSE library path (default in /opt/rose/lib)],
  [ROSE_LIB_DIR="$withval"])
AC_SUBST(ROSE_LIB_DIR)

# BOOST PATH
BOOSTDIR='/usr'
AC_ARG_WITH(boostdir,
  [  --with-boostdir=DIR          Boost install path (default in /usr)],
  [BOOSTDIR="$withval"])
AC_SUBST(BOOSTDIR)

BOOST_INCLUDE_DIR="$BOOSTDIR/include"
AC_ARG_WITH(boost_includedir,
  [  --with-boost-includedir=DIR  Boost include path (default in /usr/include)],
  [BOOST_INCLUDE_DIR="$withval"])
AC_SUBST(BOOST_INCLUDE_DIR)

BOOST_LIB_DIR="$BOOSTDIR/lib"
AC_ARG_WITH(boost_libdir,
  [  --with-boost-libdir=DIR      Boost library path (default in /usr/lib)],
  [BOOST_LIB_DIR="$withval"])
AC_SUBST(BOOST_LIB_DIR)

# additional Boost stuff: it encodes the compiler version into its lib names
BOOST_COMPILER_STRING=""
AC_ARG_WITH(boost_compiler_string,
  [  --with-boost-compiler-string=STR    Boost compiler string (default: "")],
  [BOOST_COMPILER_STRING="$withval"])
AC_SUBST(BOOST_COMPILER_STRING)


# Checks for libraries.
#-----------------------------------
AC_CHECK_FILE(${BOOST_LIB_DIR}/libboost_filesystem${BOOST_COMPILER_STRING}.so,
    AC_MSG_RESULT([found Boost (hopefully)]),
    [echo "Could not find 'libboost_filesystem${BOOST_COMPILER_STRING}.so' in '${BOOST_LIB_DIR}'."
     echo "Please consider using one of the --with-boost* options."
     exit -1 ])

AC_CHECK_FILE([${ROSE_LIB_DIR}/librose.so],
    AC_MSG_RESULT([found ROSE]),
    [echo "Could not find '-lrose' in '-L${ROSE_LIB_DIR}'."
     echo "Please consider using one of the --with-rose* options."
     exit -1 ],
    [-Wl,--rpath -Wl,${BOOST_LIB_DIR} -L${ROSE_LIB_DIR}])

# BXL
#-----------------------------------
AS_IF([test -z "`which bxl-f`"],
      [AC_MSG_WARN([
 * * *
If you want to support user-customizeable annotations, we suggest to
also install the bxl language to a location in your PATH. To install
bxl in "$prefix/bin/bxl", run
   cd ../bxl && ./configure --prefix=$prefix && make install && cd -
 * * *
])
      BXL="not found!"],
      [BXL=`which bxl`])



AC_OUTPUT

# SUMMARY

echo
echo "------------------------------------------------------------------------"
echo "  [libpaul] configuration summary:"
echo "    ROSE .................." $ROSEDIR
echo "    BXL ..................." $BXL
echo "------------------------------------------------------------------------"
