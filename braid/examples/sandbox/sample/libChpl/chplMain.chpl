
use distarray_BlockDistArray2dInt_chplImpl;

_extern proc client_main();

proc main() {
  writeln("chplMain.main() starts...");
  main_dummy_calls();
  client_main();
  writeln("chplMain.main() ends.");
}