for i in *.spec
do
  echo "=============================================================="
  name=${i%\.*}
  echo $name
  cd $name
  ../../../dist/build/rulegen/rulegen -c ../../../example.config -s ${name}_pre.trm -t ${name}_post.trm -S pre.dot -T post.dot -W weave.dot -o ${name}.str
  cd ..
done
