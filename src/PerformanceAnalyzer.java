import java.util.ArrayList;

/**
 * Utilities to compute several tracking performance criteria for a given pairing between a reference and candidate set of tracks
 *
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 *
 * */

public class PerformanceAnalyzer
{
	final ArrayList<TrackSegment> referenceTracks;
	final ArrayList<TrackSegment> candidateTracks;
	final ArrayList<TrackPair> trackPairs;
	
	/**
	 * Build the analyzer
	 * @param referenceTracks the set of reference tracks
	 * @param candidateTracks the set of candidate tracks
	 * @param trackPairs the pairing between the set of tracks. Each track in the reference set has to be represented.
	 * */
	public PerformanceAnalyzer(ArrayList<TrackSegment> referenceTracks, ArrayList<TrackSegment> candidateTracks, ArrayList<TrackPair> trackPairs)
	{
		this.referenceTracks = new ArrayList<TrackSegment>();
		this.referenceTracks.addAll(referenceTracks);
		this.candidateTracks = new ArrayList<TrackSegment>();
		this.candidateTracks.addAll(candidateTracks);
		this.trackPairs = new ArrayList<TrackPair>();
		this.trackPairs.addAll(trackPairs);
	}
	
	/**
	 * @return the number of reference tracks
	 * */
	public int getNumRefTracks()
	{
		return referenceTracks.size();
	}
	

	/**
	 * @return the total number of detection for reference tracks
	 * */
	public int getNumRefDetections()
	{
		int numDetections = 0;
		for (TrackSegment ts:referenceTracks)
			numDetections+= (ts.getLastDetection().getT() - ts.getFirstDetection().getT() +1);
		return numDetections;
	}
	

	/**
	 * @return the number of candidate tracks
	 * */
	public int getNumCandidateTracks()
	{
		return candidateTracks.size();
	}
	
	/**
	 * @return the total number of detection for candidate tracks
	 * */
	public int getNumCandidateDetections()
	{
		int numDetections = 0;
		for (TrackSegment ts:candidateTracks)
			numDetections+= (ts.getLastDetection().getT() - ts.getFirstDetection().getT() +1);
		return numDetections;
	}
	
	/**
	 * @return the distance between the pairs
	 * */
	public double getPairedTracksDistance(DistanceTypes distType, double maxDist)
	{
		double distance = 0;
		for (TrackPair tp:trackPairs)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(tp.referenceTrack, tp.candidateTrack, distType, maxDist);
			distance += d.distance;
		}
		return distance;
	}
	
	/**
	 * @return the normalized distance between the pairs (alpha criterion)
	 * */
	public double getPairedTracksNormalizedDistance(DistanceTypes distType, double maxDist)
	{
		double distance = 0;
		for (TrackPair tp:trackPairs)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(tp.referenceTrack, tp.candidateTrack, distType, maxDist);
			distance += d.distance;
		}
		// divide now by the maximum distance that corresponds to reference tracks with no associated tracks
		double normalization = 0;
		for (TrackSegment ts:referenceTracks)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(ts, null, distType, maxDist);
			normalization += d.distance;
		}
		return 1d-distance/normalization;
	}
	

	/**
	 * @return the full distance between the pairs (beta criterion) that accounts for non-associated candidate tracks
	 * */
	public double getFullTrackingScore(DistanceTypes distType, double maxDist)
	{
		double distance = 0;
		for (TrackPair tp:trackPairs)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(tp.referenceTrack, tp.candidateTrack, distType, maxDist);
			distance += d.distance;
		}
		// compute the bound on the distance
		double bound = 0;
		for (TrackSegment ts:referenceTracks)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(ts, null, distType, maxDist);
			bound += d.distance;
		}
		// compute the penalty for wrong tracks
		double penalty = 0;
		for (TrackSegment ts:candidateTracks)
		{
			boolean found = false;
			for (TrackPair tp:trackPairs)
			{
				if (tp.candidateTrack==ts)
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				TrackToTrackDistance d = new TrackToTrackDistance(ts, null, distType, maxDist);
				penalty+=d.distance;
			}
		}
		
		return (bound - distance)/(bound + penalty);
	}
	
	/**
	 * @return the number of non-associated candidate tracks
	 * */
	public int getNumSpuriousTracks()
	{
		int numSpuriousTracks = 0;
		for (TrackSegment ts:candidateTracks)
		{
			boolean found = false;
			for (TrackPair tp:trackPairs)
			{
				if (tp.candidateTrack==ts)
				{
					found = true;
					break;
				}
			}
			if (!found)
				numSpuriousTracks++;
		}
		return numSpuriousTracks;
	}
	
	/**
	 * @return the number of non-associated reference tracks (or associated with a dummy track)
	 * */
	public int getNumMissedTracks()
	{
		int numMissedTrack = 0;
		for (TrackSegment ts:referenceTracks)
		{
			boolean found = false;
			for (TrackPair tp:trackPairs)
			{
				if (tp.referenceTrack==ts)
				{
					if (tp.candidateTrack!=null && !tp.candidateTrack.getDetectionList().isEmpty())
						found = true;
					break;
				}
			}
			if (!found)
				numMissedTrack++;
		}
		return numMissedTrack;
	}
	
	/**
	 * @return the number of pairs between reference and candidate tracks
	 * */
	public int getNumPairedTracks()
	{
		int numCorrectTracks = 0;
		for (TrackSegment ts:candidateTracks)
		{
			boolean found = false;
			for (TrackPair tp:trackPairs)
			{
				if (tp.candidateTrack==ts)
				{
					found = true;
					break;
				}
			}
			if (found)
				numCorrectTracks++;
		}
		return numCorrectTracks;
	}

	/**
	 * @return the total number of paired detections
	 * */
	public int getNumPairedDetections(double maxDist)
	{
		int numRecoveredDetections = 0;
		for (TrackPair tp:trackPairs)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(tp.referenceTrack, tp.candidateTrack, DistanceTypes.DISTANCE_MATCHING, maxDist);
			numRecoveredDetections += d.numMatchingDetections;
		}
		return numRecoveredDetections;
	}

	/**
	 * @return the number of detections for the reference tracks that are not paired to a candidate detection
	 * */
	public int getNumMissedDetections(double maxDist) {
		int numMissedDetections = 0;
		for (TrackPair tp:trackPairs)
		{
			TrackToTrackDistance d = new TrackToTrackDistance(tp.referenceTrack, tp.candidateTrack, DistanceTypes.DISTANCE_MATCHING, maxDist);
			numMissedDetections += d.numNonMatchedDetections;
		}
		return numMissedDetections;
	}

	/**
	 * @return the number of detections for the candidate tracks that are not paired to a reference detection
	 * */
	public int getNumWrongDetections(double maxDist)
	{
		int numSpuriousDetections = 0;
		for (TrackSegment ts:candidateTracks)
		{
			boolean found = false;
			for (TrackPair tp:trackPairs)
			{
				if (tp.candidateTrack==ts)
				{
					TrackToTrackDistance d = new TrackToTrackDistance(tp.referenceTrack, tp.candidateTrack, DistanceTypes.DISTANCE_MATCHING, maxDist);
					numSpuriousDetections += d.numWrongDetections;
					found = true;
					break;
				}
			}
			if (!found)
			{
				for (Detection d:ts.getDetectionList())
					if(d.getDetectionType()==Detection.DetectionTypes.REAL)
						numSpuriousDetections++;// virtual detections are not considered as spurious detections
		}
		}
		return numSpuriousDetections;
	}
}
