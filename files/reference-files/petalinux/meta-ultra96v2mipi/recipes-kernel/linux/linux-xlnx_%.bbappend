FILESEXTRAPATHS_prepend := "${THISDIR}/${PN}:"

SRC_URI += "file://bsp.cfg"
SRC_URI += "file://fix_u96v2_pwrseq_simple.patch"

