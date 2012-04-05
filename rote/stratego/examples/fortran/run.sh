#!/usr/bin/env bash

src2term -Iinput -i input/test.F -o input.term
# WARNING
# Please be aware that the replacement strings for ! and # may
# theoretically conflict with a string that appears verbatim in the
# sources.
sed -i \
    -e s/\"/\\\\\"/g \
    -e s/\'/\"/g \
    -e "s/,::/,\'::\'/g" \
    -e 's/!/STRATEGO-BANG/g' \
    -e 's/#/STRATEGO-OCTOTHORPE/g' \
    input.term
./identity -i input.term > output.term
sed -i \
    -e "s/\([^\\]\)\"/\\1\'/g" \
    -e s/\\\\\"/\"/g \
    -e 's/STRATEGO-BANG/!/g' \
    -e 's/STRATEGO-OCTOTHORPE/#/g' \
    output.term
echo '.' >> output.term
term2src output.term

