#include <gtest/gtest.h>
#include <cmath> // pow
#include <Eigen/Dense>
#include "../src/InputInterface.h"
#include "../src/Decomposition.h"
#include "../src/matrix_definition.h"

namespace {

// helper function for getting a digit of a decimal number
// index is from the back and zero-based
int get_digit(double x, int index) {
    int n = (int) x;
    int larger = (int) pow(10, index+1);
    int smaller = (int) pow(10, index);
    return (int) (n%larger)/smaller;
    //return (int)(n%pow(10,index+1))/pow(10,index);
}

// Test whether global rows are calculated correctly
TEST(GlobalInputInterfaceTest, GlobalRowsTest) {
  std::vector<int> dimensions = {3,4,5,6};
  std::vector<bool> reduced(4, false);
  int fields = 1;

  reduced[2] = true;
  InputInterface<float, double> interface1(dimensions, reduced, fields);
  EXPECT_EQ(3*4*6, interface1.global_rows);

  reduced[3] = true;
  InputInterface<float, double> interface2(dimensions, reduced, fields);
  EXPECT_EQ(3*4, interface2.global_rows);

  reduced[0] = true;
  InputInterface<float, double> interface3(dimensions, reduced, fields);
  EXPECT_EQ(4, interface3.global_rows);
}

// The fixture for testing class Foo.
class SingleFileInputInterfaceTest : public ::testing::Test {
 protected:

  SingleFileInputInterfaceTest() {
    std::vector<int> dimensions = {3,4,5,3};
    std::vector<bool> reduced(4, false);
    reduced[0] = true;
    reduced[1] = true;
    int fields = 1;
    InputInterface<float, double> interface(dimensions, reduced, fields);

    std::vector<std::string> filenames = {"../test/test_x1.out"};
    bool multiple = false;
    matrix = interface.read(filenames, multiple);
  }

  // virtual ~InputInterfaceTest() { }
  // virtual void SetUp() { }
  // virtual void TearDown() { }

  // Objects declared here can be used by all tests in the test case for Foo.
  Matrix<double> matrix;
};

TEST_F(SingleFileInputInterfaceTest, RowsEqualTest) {
    for (int i=0; i<matrix.rows(); i++) {
        int n2 = get_digit(matrix(i,0),2);
        int n3 = get_digit(matrix(i,0),3);
        for (int j=1; j<matrix.cols(); j++) {
            EXPECT_EQ(n2, get_digit(matrix(i,j),2));
            EXPECT_EQ(n3, get_digit(matrix(i,j),3));
        }
    }
}

TEST_F(SingleFileInputInterfaceTest, ColsEqualTest) {
    for (int j=0; j<matrix.cols(); j++) {
        int n0 = get_digit(matrix(0,j),0);
        int n1 = get_digit(matrix(0,j),1);
        for (int i=1; i<matrix.rows(); i++) {
            EXPECT_EQ(n0, get_digit(matrix(i,j),0));
            EXPECT_EQ(n1, get_digit(matrix(i,j),1));
        }
    }
}

TEST_F(SingleFileInputInterfaceTest, RowsIncreasingTest) {
    for (int i=0; i<matrix.rows(); i++) {
        int n0 = get_digit(matrix(i,0),0);
        int n1 = get_digit(matrix(i,0),1);
        int n = 10*n1+n0;
        for (int j=1; j<matrix.cols(); j++) {
            n0 = get_digit(matrix(i,j),0);
            n1 = get_digit(matrix(i,j),1);
            int n_new = 10*n1+n0;
            EXPECT_GT(n_new, n);
            n = n_new;
        }
    }
}

TEST_F(SingleFileInputInterfaceTest, ColsIncreasingTest) {
    for (int j=0; j<matrix.cols(); j++) {
        int n2 = get_digit(matrix(0,j),2);
        int n3 = get_digit(matrix(0,j),3);
        int n = 10*n3+n2;
        for (int i=1; i<matrix.rows(); i++) {
            n2 = get_digit(matrix(i,j),2);
            n3 = get_digit(matrix(i,j),3);
            int n_new = 10*n3+n2;
            EXPECT_GT(n_new, n);
            n = n_new;
        }
    }
}

class MultiFileInputInterfaceTest : public ::testing::Test {
 protected:

  MultiFileInputInterfaceTest() {
    std::vector<int> dimensions = {3,4,5,6};
    std::vector<bool> reduced(4, false);
    reduced[1] = true;
    reduced[3] = true;
    int fields = 1;
    InputInterface<float, double> interface(dimensions, reduced, fields);

    std::vector<std::string> filenames = {"../test/test_x.lst"};
    bool multiple = true;
    matrix = interface.read(filenames, multiple);
    //std::cout << matrix << std::endl;
  }

  // virtual ~InputInterfaceTest() { }
  // virtual void SetUp() { }
  // virtual void TearDown() { }

  // Objects declared here can be used by all tests in the test case for Foo.
  Matrix<double> matrix;
};

TEST_F(MultiFileInputInterfaceTest, RowsEqualTest) {
    for (int i=0; i<matrix.rows(); i++) {
        int n0 = get_digit(matrix(i,0),0);
        int n2 = get_digit(matrix(i,0),2);
        for (int j=1; j<matrix.cols(); j++) {
            EXPECT_EQ(n0, get_digit(matrix(i,j),0));
            EXPECT_EQ(n2, get_digit(matrix(i,j),2));
        }
    }
}

TEST_F(MultiFileInputInterfaceTest, ColsEqualTest) {
    for (int j=0; j<matrix.cols(); j++) {
        int n1 = get_digit(matrix(0,j),1);
        int n3 = get_digit(matrix(0,j),3);
        for (int i=1; i<matrix.rows(); i++) {
            EXPECT_EQ(n1, get_digit(matrix(i,j),1));
            EXPECT_EQ(n3, get_digit(matrix(i,j),3));
        }
    }
}

TEST_F(MultiFileInputInterfaceTest, RowsIncreasingTest) {
    for (int i=0; i<matrix.rows(); i++) {
        int n1 = get_digit(matrix(i,0),1);
        int n3 = get_digit(matrix(i,0),3);
        int n = 10*n3+n1;
        for (int j=1; j<matrix.cols(); j++) {
            n1 = get_digit(matrix(i,j),1);
            n3 = get_digit(matrix(i,j),3);
            int n_new = 10*n3+n1;
            EXPECT_GT(n_new, n);
            n = n_new;
        }
    }
}

TEST_F(MultiFileInputInterfaceTest, ColsIncreasingTest) {
    for (int j=0; j<matrix.cols(); j++) {
        int n0 = get_digit(matrix(0,j),0);
        int n2 = get_digit(matrix(0,j),2);
        int n = 10*n2+n0;
        for (int i=1; i<matrix.rows(); i++) {
            n0 = get_digit(matrix(i,j),0);
            n2 = get_digit(matrix(i,j),2);
            int n_new = 10*n2+n0;
            EXPECT_GT(n_new, n);
            n = n_new;
        }
    }
}

class MultiFieldInputInterfaceTest : public ::testing::Test {
 protected:

  MultiFieldInputInterfaceTest() {
    std::vector<int> dimensions = {3,4,5,6};
    std::vector<bool> reduced(4, false);
    reduced[0] = true;
    reduced[2] = true;
    int fields = 2;
    InputInterface<float, double> interface(dimensions, reduced, fields);

    std::vector<std::string> filenames = {"../test/test_x.lst","../test/test_y.lst"};
    bool multiple = true;
    matrix = interface.read(filenames, multiple);
    //std::cout << matrix << std::endl << std::endl;
  }

  // virtual ~InputInterfaceTest() { }
  // virtual void SetUp() { }
  // virtual void TearDown() { }

  // Objects declared here can be used by all tests in the test case for Foo.
  Matrix<double> matrix;
};

TEST_F(MultiFieldInputInterfaceTest, RowsEqualTest) {
    for (int i=0; i<matrix.rows(); i++) {
        int n1 = get_digit(matrix(i,0),1);
        int n2 = get_digit(matrix(i,0),3);
        for (int j=1; j<matrix.cols(); j++) {
            EXPECT_EQ(n1, get_digit(matrix(i,j),1));
            EXPECT_EQ(n2, get_digit(matrix(i,j),3));
        }
    }
}

TEST_F(MultiFieldInputInterfaceTest, ColsEqualTest) {
    for (int j=0; j<matrix.cols(); j++) {
        int n1 = get_digit(matrix(0,j),0);
        int n2 = get_digit(matrix(0,j),2);
        for (int i=1; i<matrix.rows(); i++) {
            EXPECT_EQ(n1, get_digit(matrix(i,j),0));
            EXPECT_EQ(n2, get_digit(matrix(i,j),2));
        }
    }
}

TEST_F(MultiFieldInputInterfaceTest, RowsIncreasingTest) {
    for (int i=0; i<matrix.rows(); i++) {
        int n1 = get_digit(matrix(i,0),0);
        int n2 = get_digit(matrix(i,0),2);
        int n = 10*n2+n1;
        for (int j=1; j<matrix.cols(); j++) {
            n1 = get_digit(matrix(i,j),0);
            n2 = get_digit(matrix(i,j),2);
            int n_new = 10*n2+n1;
            EXPECT_GT(n_new, n);
            n = n_new;
        }
    }
}

TEST_F(MultiFieldInputInterfaceTest, ColsIncreasingTest) {
    for (int j=0; j<matrix.cols(); j++) {
        int n1 = get_digit(matrix(0,j),1);
        int n2 = get_digit(matrix(0,j),3);
        int n3 = get_digit(matrix(0,j),4);
        int n = 100*n3+10*n2+n1;
        for (int i=1; i<matrix.rows(); i++) {
            n1 = get_digit(matrix(i,j),1);
            n2 = get_digit(matrix(i,j),3);
            n3 = get_digit(matrix(i,j),4);
            int n_new = 100*n3+10*n2+n1;
            EXPECT_GT(n_new, n);
            n = n_new;
        }
    }
}

}  // namespace

int main(int argc, char **argv) {

  // initialize mpi
  MPI::Init(argc, argv);

  // run tests
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  MPI::Finalize();
  return result;
}
