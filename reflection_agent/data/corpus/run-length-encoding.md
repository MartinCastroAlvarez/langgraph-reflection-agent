# Run-length encoding

- Run-length encoding (RLE) replaces each run of a repeated character with the run length followed by the character.
- RLE example: "aaabcc" encodes to "3a1b2c" when every run carries its count, including runs of length 1.
- RLE edge case: the empty string encodes to the empty string.
- RLE decoding reads a number (possibly several digits, as in "12a") and repeats the next character that many times.
- RLE round trip: decode(encode(s)) == s for any string without digits.
