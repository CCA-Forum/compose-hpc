for i in *.spec
do
  name=${i%\.*}
  echo $name
  mkdir $name
  cd $name
  perl ../../../patch_gen.pl ../${name}.spec ${name}_pre.c ${name}_post.c
  /Users/matt/termify.sh ${name}_pre.c ${name}_pre.trm
  /Users/matt/termify.sh ${name}_post.c ${name}_post.trm
  cd ..
done
