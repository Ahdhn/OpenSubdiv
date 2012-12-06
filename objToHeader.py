#!/usr/bin/env python2.7

import sys

def main(argv):
	assert len(argv) > 1, "Usage: %s model0.obj [ model1.obj model2.obj ... ]" % argv[0]

	for obj_filename in argv[1:]:
		basename = obj_filename[:-4]
		header_filename = basename + ".h"

		with open(obj_filename, 'r') as objfile:
			lines = objfile.readlines()

		with open(header_filename, 'w') as header:
			print >>header, "static char const * %s =" % basename
			for line in lines:
				print >>header, '"%s"' % (line[:-2] + "\\n")
			print >>header, ";"

	print "done"

if __name__ == '__main__':
	main(sys.argv)
