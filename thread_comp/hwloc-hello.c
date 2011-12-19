/* Topology detection program 
 */

#include <hwloc.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
static void print_children(hwloc_topology_t topology, hwloc_obj_t obj, 
                           int depth){
    char string[128];
    unsigned i;
    
    hwloc_obj_snprintf(string, sizeof(string), topology, obj, "#", 0);
    printf("%*s%s\n", 2*depth, "", string);
    for (i = 0; i < obj->arity; i++) {
        print_children(topology, obj->children[i], depth + 1);
    }
}

int main(void)
{
    int depth;
    unsigned i, n;
    unsigned long size;
    int levels;
    char string[128];
    char hname[128];
    int topodepth;
    hwloc_topology_t topology;
    hwloc_cpuset_t cpuset;
    hwloc_obj_t obj;
    int my_pu;
    
    gethostname(hname, 128);
    //printf("%s\n\n", hname); 
    
    /* Allocate and initialize topology object. */
    hwloc_topology_init(&topology);

    /* Perform the topology detection. */
    hwloc_topology_load(topology);

    /* Optionally, get some additional topology information
       in case we need the topology depth later. */
    topodepth = hwloc_topology_get_depth(topology);

    depth = hwloc_get_type_or_below_depth(topology, HWLOC_OBJ_CORE);

    /* Get last core. */
    obj = hwloc_get_obj_by_depth(topology, depth,
                   hwloc_get_nbobjs_by_depth(topology, depth) - 1);
    if (obj) {
        /* Get a copy of its cpuset that we may modify. */
        cpuset = hwloc_bitmap_dup(obj->cpuset);

        /* Get only one logical processor (in case the core is
           SMT/hyperthreaded). */
        hwloc_bitmap_singlify(cpuset);
        my_pu = hwloc_get_cpubind(topology, cpuset, 0);
        printf("%s: i am bound to PU#: %d \n", hname, my_pu);

        /* Free our cpuset copy */
        hwloc_bitmap_free(cpuset);
    }

    /* Destroy topology object. */
    hwloc_topology_destroy(topology);

    return 0;
}
