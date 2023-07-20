== Manual of the tracking performance evaluation software ==

This aim of this software is to provide measures of particle tracking performance
It is provided by the organizers of the ISB'2012 particle tracking challenge

Usage: java -jar trackingPerformanceEvaluation.jar [-hrcdo] ...

If no input is specified then a dedicate GUI is shown to set up the compuationOtherwise, the following options can be used:
	 -h, -help, displays the manual of the software
	 -r specifies the input file for the reference tracks [Mandatory]
	 -c specifies the input file for the candidate tracks [Mandatory]
	 -o specifies the output file for the results [Optional]
	 -d specifies the maximum distance between detections [Optional]

If no output file is specified the results are sent to the standard output stream.
If no maximum distance is specified, then the default value: 5, is used.

Examples of usage:

java -jar trackingPerformanceEvaluation.jar
	 displays the dedicated GUI for tracking performance computation

java -jar trackingPerformanceEvaluation.jar -r referenceTracks.xml -c candidateTracks.xml
	 compute the tracking measures for the tracks from file candidateTracks.xml,
	 with respect to the reference tracks in referenceTracks.xml,
	 results are sent to the standard output stream
	 The input files have to follow the XML standard of the ISBI'2012 Particle Tracking Challenge.

java -jar trackingPerformanceEvaluation.jar -r referenceTracks.xml -c candidateTracks.xml -d 3
	 Same as above, but the value of the gate distance is 3

java -jar trackingPerformanceEvaluation.jar -r referenceTracks.xml -c candidateTracks.xml -d 3 -o results.txt
	 Same as above, but the results are output to the text file results.txt

For more details visit the ISBI'2012 Tracking Challenge website or contact the organizers.
Author: Nicolas Chenouard
Date: Feb 5 2012