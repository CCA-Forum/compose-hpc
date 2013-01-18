AC_PREREQ(2.65)
AC_INIT([BXL],[1.0],[adrian@llnl.gov],[bxl],[http://compose-hpc.sourceforge.net/])
AC_CONFIG_SRCDIR([README])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([config])
AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE([foreign 1.11 dist-bzip2 dist-xz silent-rules std-options])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE
AM_PATH_PYTHON([2.6])
AS_IF([$PYTHON -c "import pyparsing"], 
      [:], 
      [AC_MSG_ERROR([Could not find the Python module "pyparsing"])])

AC_CONFIG_FILES([Makefile bxl])

AC_OUTPUT
AC_MSG_RESULT([BXL configured successfully!])
