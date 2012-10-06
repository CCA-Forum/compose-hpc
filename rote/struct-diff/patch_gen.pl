#!/usr/bin/perl

if ($#ARGV != 2) {
	print "USAGE: patch_gen.pl patcher_file\n";
	exit(1);
}

$fname = $ARGV[1];

open(INFILE, $fname) || die "Error opening $fname.";

@input = ();
@output = ();

while (<INFILE>) {
	if (/^\+/) {
		s/^\+//;
		push(@output,$_);
	} else if (/^\-/) {
		s/^\-//;
		push(@input,$_);
	} else {
		push(@input,$_);
		push(@output,$_);
	}
}

close(INFILE);
