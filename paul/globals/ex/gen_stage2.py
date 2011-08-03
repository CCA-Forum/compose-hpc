import pickle
funcset= set()
try:
  pkl_file = open('data.pkl', 'rb')
  tmpset=pickle.load(pkl_file)
  pkl_file.close()
  funcset = tmpset
except:
  print "pickle load failed"
  exit(2)

stage2 = open("stage2.cocci","a+",1)
for id in funcset:
  print "ID%s" % (id)
  print >> stage2, """
@fix_%s@
expression P;
@@
-%s(P)
+%s( ((struct dummy_insert *)4096), P)

""" % (id, id,id)
stage2.close()

