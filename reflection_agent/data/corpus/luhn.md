# Luhn checksum

- The Luhn algorithm validates identification numbers such as payment card numbers with a mod 10 checksum.
- Luhn: starting from the rightmost digit (the check digit) and moving left, double every second digit.
- Luhn: when doubling a digit gives a number greater than 9, subtract 9 from it (or add its two digits).
- Luhn: the number is valid when the sum of all the digits, after doubling, is a multiple of 10.
- Luhn input handling: spaces are usually ignored; any other non-digit character makes the number invalid.
- Luhn edge case: a number with fewer than two digits is invalid, because it has no payload besides the check digit.
- Luhn example: 79927398713 is valid, and changing its last digit to 0 makes it invalid.
