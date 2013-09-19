from sys import argv, stderr
from optparse import OptionParser
from networkx import expected_degree_graph
from scipy.stats import poisson

def gen_graph(mu, num_nodes):
    expected_degrees = poisson.rvs(mu, size=num_nodes)
    return expected_degree_graph(expected_degrees)

def main(argv):
    parser = OptionParser()
    parser.add_option("-n", type="int", dest="num_nodes", default=20)
    parser.add_option("-mu", type="int", dest="mu", default=4)
    (options, args) = parser.parse_args()

    print >>stderr, "Generating graph with %d nodes." % options.num_nodes
    print >>stderr, "Using mu = %d." % options.mu
    gen_graph( options.mu, options.num_nodes )
    print >>stderr, "Done."


if __name__ == '__main__':
    import sys
    main(sys.argv)
