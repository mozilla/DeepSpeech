#! /usr/bin/awk -f
# A script to extract the actual suppression info from the output of (for example) valgrind --leak-check=full --show-reachable=yes --error-limit=no --gen-suppressions=all ./minimal
# The desired bits are between ^{ and ^} (including the braces themselves).
# The combined output should either be appended to /usr/lib/valgrind/default.supp, or placed in a .supp of its own
# If the latter, either tell valgrind about it each time with --suppressions=<filename>, or add that line to ~/.valgrindrc

# NB This script uses the |& operator, which I believe is gawk-specific. In case of failure, check that you're using gawk rather than some other awk

# The script looks for suppressions. When it finds one it stores it temporarily in an array,
# and also feeds it line by line to the external app 'md5sum' which generates a unique checksum for it.
# The checksum is used as an index in a different array. If an item with that index already exists the suppression must be a duplicate and is discarded.

BEGIN { suppression=0; md5sum = "md5sum" }
  # If the line begins with '{', it's the start of a supression; so set the var and initialise things
  /^{/  {
           suppression=1;  i=0; next 
        }
  # If the line begins with '}' its the end of a suppression
  /^}/  {
          if (suppression)
           { suppression=0;
             close(md5sum, "to")  # We've finished sending data to md5sum, so close that part of the pipe
             ProcessInput()       # Do the slightly-complicated stuff in functions
             delete supparray     # We don't want subsequent suppressions to append to it!
           }
     }
  # Otherwise, it's a normal line. If we're inside a supression, store it, and pipe it to md5sum. Otherwise it's cruft, so ignore it
     { if (suppression)
         { 
            supparray[++i] = $0
            print |& md5sum
         }
     }


 function ProcessInput()
 {
    # Pipe the result from md5sum, then close it     
    md5sum |& getline result
    close(md5sum)
    # gawk can't cope with enormous ints like $result would be, so stringify it first by prefixing a definite string
    resultstring = "prefix"result

    if (! (resultstring in chksum_array) )
      { chksum_array[resultstring] = 0;  # This checksum hasn't been seen before, so add it to the array
        OutputSuppression()              # and output the contents of the suppression
      }
 }

 function OutputSuppression()
 {
  # A suppression is surrounded by '{' and '}'. Its data was stored line by line in the array  
  print "{"  
  for (n=1; n <= i; ++n)
    { print supparray[n] }
  print "}" 
 }
