#run!/bin/sh

echo "Compiling Braid -> BLAS tests."

# Begin of main driver

set -x
echo "--set up environment"
BABEL=/Users/imam1/softwares/bin/babel
BRAID="env PYTHONPATH=../../../../braid:../../../../braid/.libs:../../../../../compose-hpc/braid/contrib/argparse-1.1:../../../../../compose-hpc/braid/contrib/ply: /usr/bin/python ../../../../../compose-hpc/braid/braid/braid.py"
sidl=blas.sidl

echo "--generate client"
cd runChapel						||exit 1

for libdir in `ls -1 -d ../lib*`; do
  language=`echo $libdir|sed 's/^...lib//g'`
  case $language in
      F*) extralib="-lgfortran" ;;
      *)  extralib= ;;
  esac
  echo "--compiling client [ $language ]"
  echo "Current directory: `pwd`"
  make SERVER="${libdir}/libimpl*.la" IMPL="daxpy_hybird_block_cyclic" EXTRA_LDFLAGS="$extralib" OUTFILE="runChapel2$language"                   ||exit 1

  # generate wrapper scripts
  case "$language" in
      Java)
	  cat >runChapel2Java.sh <<EOF
env LD_LIBRARY_PATH="${libjvm_dir}:${libjava_dir}:/Users/imam1/softwares/lib:${LD_LIBRARY_PATH}:$libdir" \
    CLASSPATH="../libJava:/Users/imam1/softwares/lib/sidl-2.0.0.jar:/Users/imam1/softwares/lib/sidlstub_2.0.0.jar:.:$libdir" \
    SIDL_DLL_PATH="../libJava/libimpl.scl;/Users/imam1/softwares/lib/libsidlstub_java.scl;/Users/imam1/softwares/lib/libsidl.scl;../../../runtime/sidlx/libsidlx.scl;../../output/libC/libOutput.scl" \
      ./runChapel2$language 
EOF
	  ;;
      Python)
	  cat >runChapel2Python.sh <<EOF
env LD_LIBRARY_PATH="/Users/imam1/softwares/lib:${LD_LIBRARY_PATH}" \
    PYTHONPATH="../libPython:`/usr/bin/python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(prefix='/Users/imam1/softwares',plat_specific=1) + ':' + get_python_lib(prefix='/Users/imam1/softwares')"`:$PYTHONPATH" \
    SIDL_DLL_PATH="../libPython/libimpl.scl;../../output/libC/libOutput.scl;/Users/imam1/softwares/lib/libsidlx.scl" \
    ./runChapel2$language
EOF
	  ;;
      *)
	  cat >runChapel2$language.sh <<EOF
env LD_LIBRARY_PATH="/Users/imam1/softwares/lib:${LD_LIBRARY_PATH}" \
    SIDL_DLL_PATH="$libdir/libimpl.scl;../../output/libC/libOutput.scl;/Users/imam1/softwares/lib/libsidlx.scl" \
    ./runChapel2$language
EOF
	  ;;
  esac
  chmod u+x runChapel2$language.sh
done

