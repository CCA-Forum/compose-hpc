==========================
MINITERMITE HELPER SCRIPTS
==========================

This directory contains some helpers that are useful for using
minitermite and the aterm analyzers in environments where ROSE is not
locally installed.  For example, users of unsupported systems (e.g.,
recent version of MacOS X) can use a remote server that has ROSE built
to perform source to term mapping (and vice versa) without much pain
via these scripts.  They require a small amount of setup and
configuration.

SERVER SIDE:
------------

On the server side (where ROSE lives), once you have built ROSE,
modify the compose_rose_setup.sh script to make sure it points at the
proper ROSE, BOOST, and Java locations.  This shell script is used to
set up the environment such that ROSE-based programs can be run.

Next, create a sandbox directory.  This is a working scratch directory
where files are copied and generated into.  The example files assume
that this is a directory called "sandbox" in the root of your home
directory.

Finally, copy three scripts to the server and put them in your home
directory:

compose_rose_setup.sh
src2term.sh
term2src.sh

CLIENT SIDE:
------------

On the system that will call out to the server to do the term and
source conversion, you need to place the scripts:

termify.sh
sourcify.sh

These require only one environment variable to be set indicating the
username and hostname of the remote host.  Note that it is HIGHLY
recommended that you use SSH keys to avoid having to type your
password over and over and over during the process of scp'ing and
ssh'ing within the scripts.

USAGE:
------

Say you have a file called "test.c".  One test to make sure this works
is to try to termify it, and then convert that back to source with
sourcify.sh.

% cat > test.c
void foo() {
  int x = 5;
}
^D

% termify.sh test.c test.trm
% sourcify.sh test.trm test_new.c

Modulo some differences in formatting, the contents of test_new.c
should be equivalent to test.c.  If you have any errors, you likely
have to check your paths, permissions, and hostnames.
