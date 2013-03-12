for i in *.spec
do
  name=${i%\.*}
  rm -Rf $name
done
