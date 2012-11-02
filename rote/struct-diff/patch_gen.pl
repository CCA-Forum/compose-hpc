#!/usr/bin/perl
#
# this script is used to write pre/post inputs to the
# rule generator in a compact form.
#

print $#ARGV."\n";

if ($#ARGV != 2) {
	print "USAGE: patch_gen.pl changespec prefile postfile\n";
	exit(1);
}

$input_fname = $ARGV[0];
$pre_fname = $ARGV[1];
$post_fname = $ARGV[2];

open(INFILE, $input_fname) || die "Error opening $input_fname.";

@input = ();
@output = ();

while (<INFILE>) {
	if (/^\+/) {
		s/^\+//;
		push(@output,$_);
	} elsif (/^\-/) {
		s/^\-//;
		push(@input,$_);
	} else {
		push(@input,$_);
		push(@output,$_);
	}
}

close(INFILE);

open(PREFILE, "> $pre_fname") || die "Error opening $pre_fname.";
open(POSTFILE, "> $post_fname") || die "Error opening $post_fname.";

foreach $line (@input) {
	print PREFILE $line;
}
foreach $line (@output) {
	print POSTFILE $line;
}

close(PREFILE);
close(POSTFILE);