Tests for the POD program
=========================

Description of test files
-------------------------

* binary arrays of integer numbers saved as floats
* four dimensions, length 3, 4, 5, and 6 (fastest to slowest)
* last (slowest) dimension split into three files: 1..3, 4..5, and 6
* fastest dimension increments last digit, slowest dimension increments first digit
* one more digit for variables: 1 and 2
* test_x1.out, test_x2.out, test_x3.out, test_y1.out, test_y2.out, test_y3.out
* summaries: test_x.lst, test_y.lst
