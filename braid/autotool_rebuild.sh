#!/bin/bash

machine=`hostname`;

case $machine in 
  up*)          export PATH=/usr/apps/babel/dev_2008/bin:${PATH}
                AUTORECONF=autoreconf
                ;;
  lemur.*)      AUTORECONF=/opt/local/bin/autoreconf ;;
  ingot*)       AUTORECONF=/usr/apps/babel/dev_tools/bin/autoreconf ;;
  tux314|tux316)       AUTORECONF=/usr/bin/autoreconf ;;
  tux*)         export PATH=/usr/casc/babel/apps/autotools_2009/bin:${PATH}
	        AUTORECONF=/usr/casc/babel/apps/autotools_2009/bin/autoreconf
	        ;;
  driftcreek*)  AUTORECONF=autoreconf ;;
  *)            AUTORECONF=autoreconf ;;
esac

echo "**** $AUTORECONF ****"
$AUTORECONF -i
