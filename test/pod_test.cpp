#include <gtest/gtest.h>
#include "../src/InputInterface.h"

namespace {

// The fixture for testing class Foo.
class InputInterfaceTest : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  InputInterfaceTest() {
    // You can do set-up work for each test here.
  }

  virtual ~InputInterfaceTest() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case for Foo.
};

// Tests that the Foo::Bar() method does Abc.
TEST_F(InputInterfaceTest, GlobalRowsTest) {
  std::vector<int> dimensions(4, 20);
  std::vector<bool> reduced(4, false);
  reduced[2] = true;
  int fields(2);
  InputInterface<float, double> interface(dimensions, reduced, fields);

  EXPECT_EQ(20*20*20*2, interface.global_rows);
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
