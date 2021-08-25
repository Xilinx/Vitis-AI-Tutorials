DESCRIPTION = "json-c autotools recipe"
HOMEPAGE = "https://github.com/json-c/json-c/"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://COPYING;md5=de54b60fbbc35123ba193fea8ee216f2"
SRC_URI[md5sum] = "20dba7bf773599a0842745a2fe5b7cd3"

S = "${WORKDIR}/json-c-json-c-0.13.1-20180305"

SRC_URI = "https://github.com/json-c/json-c/archive/json-c-0.13.1-20180305.tar.gz"

inherit autotools
