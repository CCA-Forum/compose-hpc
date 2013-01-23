/* %CONTRACT INIT      foo: bar; */
/* %CONTRACT INVARIANT foo: bar;*/
/* %CONTRACT FINAL     foo: bar;*/
/* %CONTRACT REQUIRE 
    pos_weights: ((weights!=NULL) and (n>0)) 
			implies pce_all(weights>0, n); 
 */
/* %CONTRACT ENSURE 
    side_effect_free: is pure;
 */

int
main(int argc, char **argv) {
 int res = 0;
 return res;
}
