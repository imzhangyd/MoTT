import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * Tracking performance evaluation with no GUI.
 * 
 * @version February 5, 2012
 * 
 * @author Nicolas Chenouard
 * */

public class BatchPerformanceEvaluation
{
	final static double DEFAULT_GATE = 5.0;
	File referenceFile = null;
	File candidateFile = null;
	File outputFile = null;
	double maxDist = DEFAULT_GATE;
	DistanceTypes distType = DistanceTypes.DISTANCE_EUCLIDIAN;

	String initLog = "";
	String pairingLog = "";
	String exportLog = "";

	ArrayList<TrackPair> trackPairs = new ArrayList<TrackPair>();

	ArrayList<TrackSegment> referenceTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> candidateTracks = new ArrayList<TrackSegment>();

	ArrayList<TrackSegment> recoveredTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> correctTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> missedTracks = new ArrayList<TrackSegment>();
	ArrayList<TrackSegment> spuriousTracks = new ArrayList<TrackSegment>();

	/**
	 * initialize the computation
	 * @param args table of input arguments of the program. See the manual for a description.
	 * @return true if the initialization step was successful
	 * */
	public boolean init(String[] args)
	{
		boolean candidateFileFound = false;
		boolean referenceFileFound = false;
		boolean outputFileFound = false;
		referenceFile = null;
		candidateFile = null;
		outputFile = null;
		initLog = "";
		double dist = DEFAULT_GATE;

		int i = 0;
		while (i<args.length)
		{
			if (args[i].equals("-h") || args[i].equals("-help") || args[i].equals("--h") || args[i].equals("--help"))
			{
				initLog = "";
				return false;
			}
			if (args[i].equals("-r"))
			{
				if (args.length>i)
				{
					referenceFile = new File(args[i+1]);
					referenceFileFound = true;
				}
			}
			if (args[i].equals("-c"))
			{				
				if (args.length>i)
				{
					candidateFile = new File(args[i+1]);
					candidateFileFound = true;
				}
			}
			if (args[i].equals("-o"))
			{
				if (args.length>i)
				{
					outputFile = new File(args[i+1]);
					outputFileFound = true;
				}
			}
			if (args[i].equals("-d"))
			{
				if (args.length>i)
					try {
						dist = Double.parseDouble(args[i+1]);
					} catch (Exception e) {
						initLog = "Invalid distance";
						return false;
					}
			}
			i++;
		}
		// check if input arguments are okay
		if (!referenceFileFound) 
		{
			initLog = "No reference tracks file";
			return false;
		}
		if (referenceFile == null || !referenceFile.exists())
		{
			initLog = "Reference tracks file not found";
			return false;
		}
		if (!candidateFileFound) 
		{
			initLog = "No candidate tracks file";
			return false;
		}
		if (candidateFile == null || !candidateFile.exists())
		{
			initLog = "Candidate tracks file not found";
			return false;
		}
		if (dist<0)
		{
			initLog = "Maximum distance is negative";
			return false;
		}
		else
			maxDist = dist;
		if (outputFileFound)
		{
			if (outputFile == null)
			{
				initLog = "Output file not found";
				return false;
			}
		}
		// try to load reference tracks
		referenceTracks.clear();
		try{
			referenceTracks.addAll(TrackExportAndImportUtilities.importTracksFile(referenceFile));
		}
		catch (Exception e)
		{
			initLog = "Loading reference tracks failed";
			return false;
		}
		initLog = initLog + "Num reference tracks loaded "+referenceTracks.size()+"\n";

		// try to load candidate tracks
		candidateTracks.clear();
		try{
			candidateTracks.addAll(TrackExportAndImportUtilities.importTracksFile(candidateFile));
		}
		catch (Exception e)
		{
			initLog = "Loading candidate tracks failed";
			return false;
		}
		initLog = initLog + "Num candidate tracks loaded "+candidateTracks.size()+"\n";
		return true;
	}

	/**
	 * Pair the candidate tracks with reference tracks
	 * */
	public boolean pairTracks()
	{
		pairingLog = "";
		// now compute the scores
		OneToOneMatcher matcher = new OneToOneMatcher(referenceTracks, candidateTracks);

		ArrayList<TrackPair> pairs = new ArrayList<TrackPair>();
		try {
			pairs.addAll(matcher.pairTracks(maxDist, distType));
		} catch (Exception e) {
			pairingLog = e.getMessage();
			return false;
		}
		pairingLog = "pairing succeeded";
		// remove spurious candidate tracks
		recoveredTracks.clear();
		correctTracks.clear();
		missedTracks.clear();
		spuriousTracks.clear();
		for (TrackPair tp:pairs)
		{
			if (tp.candidateTrack.getDetectionList().isEmpty())
			{
				tp.candidateTrack = null;
				missedTracks.add(tp.referenceTrack);
			}
			else
			{
				recoveredTracks.add(tp.referenceTrack);
				correctTracks.add(tp.candidateTrack);
			}
		}
		for (TrackSegment ts:candidateTracks)
		{
			if (!correctTracks.contains(ts))
				spuriousTracks.add(ts);
		}
		trackPairs.clear();
		trackPairs.addAll(pairs);
		return true;
	}

	/**
	 * Print several tracking criteria in the standard system output stream
	 * @return true is computation of criteria and printing them was successful
	 * */
	public boolean printPerformanceMeasures()
	{
		try{
			PerformanceAnalyzer analyzer = new PerformanceAnalyzer(referenceTracks, candidateTracks, trackPairs);
			System.out.println(analyzer.getPairedTracksDistance(distType, maxDist)+"\t : pairing distance");
			System.out.println(analyzer.getPairedTracksNormalizedDistance(distType, maxDist)+"\t : normalized pairing score (alpha)");
			System.out.println(analyzer.getFullTrackingScore(distType, maxDist)+"\t : full normalized score (beta)");
			System.out.println(analyzer.getNumRefTracks()+"\t : number of reference tracks");
			System.out.println(analyzer.getNumCandidateTracks()+"\t : number of candidate tracks");
			double tracksSimilarity = (double)analyzer.getNumPairedTracks()/((double)analyzer.getNumRefTracks() + (double)analyzer.getNumSpuriousTracks());
			System.out.println(tracksSimilarity+"\t : Similarity between tracks (Jaccard)");
			System.out.println(analyzer.getNumPairedTracks()+"\t : number of paired tracks");
			System.out.println(analyzer.getNumMissedTracks()+"\t : number of missed tracks (out of "+analyzer.getNumRefTracks()+")");
			System.out.println(analyzer.getNumSpuriousTracks()+"\t : number of spurious tracks)");
			System.out.println(analyzer.getNumRefDetections()+"\t : number of reference detections");
			System.out.println(analyzer.getNumCandidateDetections()+"\t : number of candidate detections");
			double detectionsSimilarity = (double) analyzer.getNumPairedDetections(maxDist)/((double) analyzer.getNumRefDetections() + (double)analyzer.getNumWrongDetections(maxDist));
			System.out.println(detectionsSimilarity+"\t : Similarity between detections (Jaccard)");
			System.out.println(analyzer.getNumPairedDetections(maxDist)+"\t : number of paired detections");
			System.out.println(analyzer.getNumMissedDetections(maxDist)+"\t : number of missed detections (out of "+analyzer.getNumRefDetections()+")");
			System.out.println(analyzer.getNumWrongDetections(maxDist)+"\t : number of spurious detections");
		}
		catch (Exception e)
		{
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * Save several tracking criteria in the File object ouputFile
	 * @return true is computation of criteria and saving them was successful
	 * */
	public boolean saveResults()
	{
		if (outputFile == null)
			return false;
		if (outputFile.exists())
		{
			exportLog = "Output file already exists. Erase it first.";
			return false;
		}

		PerformanceAnalyzer analyzer;
		try{
			analyzer = new PerformanceAnalyzer(referenceTracks, candidateTracks, trackPairs);
		}
		catch(Exception e)
		{
			e.printStackTrace();
			return false;
		}
		try {
			FileWriter outFile = new FileWriter(outputFile);
			PrintWriter out = new PrintWriter(outFile);

			out.println(analyzer.getPairedTracksDistance(distType, maxDist)+"\t : pairing distance");
			out.println(analyzer.getPairedTracksNormalizedDistance(distType, maxDist)+"\t : normalized pairing score (alpha)");
			out.println(analyzer.getFullTrackingScore(distType, maxDist)+"\t : full normalized score (beta)");
			out.println(analyzer.getNumRefTracks()+"\t : number of reference tracks");
			out.println(analyzer.getNumCandidateTracks()+"\t : number of candidate tracks");
			double tracksSimilarity = (double)analyzer.getNumPairedTracks()/((double)analyzer.getNumRefTracks() + (double)analyzer.getNumSpuriousTracks());
			out.println(tracksSimilarity+"\t : Similarity between tracks (Jaccard)");
			out.println(analyzer.getNumPairedTracks()+"\t : number of paired tracks");
			out.println(analyzer.getNumMissedTracks()+"\t : number of missed tracks (out of "+analyzer.getNumRefTracks()+")");
			out.println(analyzer.getNumSpuriousTracks()+"\t : number of spurious tracks)");
			out.println(analyzer.getNumRefDetections()+"\t : number of reference detections");
			out.println(analyzer.getNumCandidateDetections()+"\t : number of candidate detections");
			double detectionsSimilarity = (double) analyzer.getNumPairedDetections(maxDist)/((double) analyzer.getNumRefDetections() + (double)analyzer.getNumWrongDetections(maxDist));
			out.println(detectionsSimilarity+"\t : Similarity between detections (Jaccard)");
			out.println(analyzer.getNumPairedDetections(maxDist)+"\t : number of paired detections");
			out.println(analyzer.getNumMissedDetections(maxDist)+"\t : number of missed detections (out of "+analyzer.getNumRefDetections()+")");
			out.println(analyzer.getNumWrongDetections(maxDist)+"\t : number of spurious detections");
			out.close();
		} catch (IOException e){
			e.printStackTrace();
			return false;
		}
		return true;
	}

	/**
	 * @return the manual of the software
	 * */
	public static String getHelp()
	{
		String help = "== Manual of the tracking performance evaluation software ==\n\n";
		help = help + "This aim of this software is to provide measures of particle tracking performance\n";
		help = help + "It is provided by the organizers of the ISB'2012 particle tracking challenge\n";
		help = help + "\n";
		help = help + "Usage: java -jar performanceEvaluation.jar [-hrcdo] ...\n\n";
		help = help +"If no input is specified then a dedicate GUI is shown to set up the compuation";
		help = help +"Otherwise, the following options can be used:";
		help = help + "\n";
		help = help +"\t -h, -help, displays the manual of the software\n";
		help = help +"\t -r specifies the input file for the reference tracks [Mandatory]\n";
		help = help +"\t -c specifies the input file for the candidate tracks [Mandatory]\n";
		help = help +"\t -o specifies the output file for the results [Optional]\n";
		help = help +"\t -d specifies the maximum distance between detections [Optional]\n";
		help = help + "\n";
		help = help +"If no output file is specified the results are sent to the standard output stream.\n";
		help = help +"If no maximum distance is specified, then the default value: 5, is used.\n";
		help = help + "\n";		
		help = help +"Examples of usage:\n";
		help = help + "\n";		
		help = help + "java -jar performanceEvaluation.jar\n";
		help = help + "\t displays the dedicated GUI for tracking performance computation\n\n";
		help = help + "java -jar performanceEvaluation.jar -r referenceTracks.xml -c candidateTracks.xml\n";
		help = help + "\t compute the tracking measures for the tracks from file candidateTracks.xml,\n";
		help = help + "\t with respect to the reference tracks in referenceTracks.xml,\n";
		help = help + "\t results are sent to the standard output stream\n";
		help = help + "\t The input files have to follow the XML standard of the ISBI'2012 Particle Tracking Challenge.\n\n";
		help = help + "java -jar performanceEvaluation.jar -r referenceTracks.xml -c candidateTracks.xml -d 3\n";
		help = help + "\t Same as above, but the value of the gate distance is 3\n\n";
		help = help + "java -jar performanceEvaluation.jar -r referenceTracks.xml -c candidateTracks.xml -d 3 -o results.txt\n";
		help = help + "\t Same as above, but the results are output to the text file results.txt\n\n";
		help = help + "For more details visit the ISBI'2012 Tracking Challenge website or contact the organizers.\n";
		help = help + "Author: Nicolas Chenouard\n";
		help = help + "Date: Feb 5 2012\n";
		return help;
	}

	/**
	 * @return the log of the initialization step
	 * */
	public String getInitLog()
	{
		return new String(initLog);
	}

	/**
	 * @return the log of the pairing step
	 * */
	public String getPairingLog()
	{
		return new String(pairingLog);
	}

	/**
	 * Compute the tracking results from the pairs and output them
	 * @return true is computation and export of results were successful
	 * */
	public boolean outputResults() {
		exportLog = "";
		if (outputFile!=null)
			return saveResults();
		else
			return printPerformanceMeasures();
	}

	/**
	 * @return the log of criteria computation and output step
	 * */
	public String getExportLog() {
		return new String(exportLog);
	}
}
