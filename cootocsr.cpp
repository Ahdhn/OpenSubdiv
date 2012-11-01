/**
 * How to build a sparse matrix in CSR format using boost.
 */

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/foreach.hpp>

#define foreach BOOST_FOREACH

using namespace std;
using namespace boost::numeric::ublas;

int main () {

    // insert values into coo matrix
    coordinate_matrix<double> m0 (3, 3, 3 * 3);
    for (unsigned i = 1; i < m0.size1 (); ++ i)
        for (unsigned j = i; j < m0.size2 (); ++ j)
            m0 (i, j) = 3 * i + j;

    // compress coo matrix to csr format
    compressed_matrix<double> m1 (m0);

    // access vectors
    cout << "row ptrs: ";
    foreach(double d,  m1.index1_data())
        cout << d << " ";
    cout << endl;

    cout << "col ptrs: ";
    foreach(double d,  m1.index2_data())
        cout << d << " ";
    cout << endl;

    cout << "vals: ";
    foreach(double d,  m1.value_data())
        cout << d << " ";
    cout << endl;
}
