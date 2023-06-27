# SWMM Optimization
The following code optimizes the usage of LID in a watershed based on antifragility or reliability of the system. The code creates the SWMM input file based on the guessed variables, evaluates system perormance, and repeats as necessary.
The SWMM input file needs to be broken into two pieces, so the synthetic rainfall data can be creted and inserted into a final input file. The code requires swmm5.exe and swmm5.dll files to be present in the same folder as it essentially runs a third party SWMM model in order to simulate a watershed.
