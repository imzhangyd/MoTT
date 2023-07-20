/**
 *  Creation of the best one-to-one matching between a set of reference tracks and a set of candidate tracks.
 *  Dummy tracks representing no candidate tracks associated to a reference track are allowed.
 *  
 * @version February 4, 2012
 * 
 * @author Nicolas Chenouard
 *
 */

import java.util.ArrayList;
import java.util.HashSet;

public class OneToOneMatcher {

	final ArrayList<TrackSegment> refTracks;
	final ArrayList<TrackSegment> candidateTracks;
	private ArrayList<ArrayList<TrackPair>> feasiblePairs;

	/**
	 * *
	 * @param refTracks the set of reference tracks
	 * @param candidateTracks the set of candidate tracks
	 */
	public OneToOneMatcher(ArrayList<TrackSegment> refTracks, ArrayList<TrackSegment> candidateTracks)
	{
		this.refTracks = new ArrayList<TrackSegment>();
		this.refTracks.addAll(refTracks);

		this.candidateTracks = new ArrayList<TrackSegment>();
		this.candidateTracks.addAll(candidateTracks);

		this.feasiblePairs = new ArrayList<ArrayList<TrackPair>>();
	}

	/**
	 * Compute the best pairing
	 * @param maxDist maximum Euclidian distance between two detections (gate)
	 * @param distType type of distance that is used for computing the costs of association
	 * @param maxDist the gate (maximum Euclidian distance) for distance computation
	 * @return the best pairing between tracks. All the reference tracks are paired (potentially to a dummy track), while some candidate tracks may not be in the list
	 * @throws Exception 
	 * */
	public ArrayList<TrackPair> pairTracks(double maxDist, DistanceTypes distType) throws Exception
	{
		// build the potential track pairs
		this.feasiblePairs.clear();
		for (TrackSegment ts:refTracks)
			this.feasiblePairs.add(getFeasiblePairs(ts, candidateTracks, distType, maxDist));
		// cluster track pairs
		ArrayList<TrackPairsCluster> clusters = new ArrayList<OneToOneMatcher.TrackPairsCluster>();
		try{clusters.addAll(getTrackPairClusters());}
		catch(Exception e)
		{
			throw e;
		}
		ArrayList<TrackPair> assignment = new ArrayList<TrackPair>();
		for (TrackPairsCluster cluster:clusters)
		{
			cluster.buildCostMatrix();
			// use Munkres algorithm to find the best pairing
			HungarianMatching matcher = new HungarianMatching(cluster.costs);
			try{
				boolean[][] matching = matcher.optimize();
				assignment.addAll(cluster.getAssignements(matching));
			}
			catch(Exception e)
			{
				throw e;
			}
		}
		return assignment;
	}

	/**
	 * Build the clusters of TrackPair objects
	 * @return list of TrackPairsCluster for the current set of TrackPairs
	 * */
	private ArrayList<TrackPairsCluster> getTrackPairClusters() throws Exception {
		ArrayList<TrackPairsCluster> clusters = new ArrayList<TrackPairsCluster>();
		for (ArrayList<TrackPair> trackPairsList:feasiblePairs)
		{
			if (!trackPairsList.isEmpty())
			{
				// create a cluster with tracks corresponding to the current reference track
				TrackPairsCluster currentCluster = new TrackPairsCluster();
				TrackSegment refTrack = trackPairsList.get(0).referenceTrack;
				currentCluster.usedReferenceTracks.add(refTrack);
				for (TrackPair tp:trackPairsList)
					currentCluster.usedCandidateTracks.add(tp.candidateTrack);
				currentCluster.trackPairs.addAll(trackPairsList);
				// now try to merge this cluster with others
				ArrayList<TrackPairsCluster> clustersCopy = new ArrayList<TrackPairsCluster>();
				clustersCopy.add(currentCluster);
				for (TrackPairsCluster cluster:clusters)
				{
					// check if cluster contains tracks that are used by the current cluster
					// we do not check for the reference track as it should not be used elsewhere
					boolean found = false;
					for (TrackPair tp:trackPairsList)
					{
						found = cluster.usedCandidateTracks.contains(tp.candidateTrack);
						if (found)
							break;
					}
					if (found)
					{
						// merge clusters
						currentCluster.mergeCluster(cluster);
					}
					else
					{
						// keep the investigate cluster intact
						clustersCopy.add(cluster);
					}
				}
				clusters = clustersCopy;
			}
			else
			{
				throw new Exception("There is a track cluster empty");
			}
		}
		return clusters;
	}

	/**
	 * Compute the cost matrix for the assignment between reference and candidate tracks
	 * @return a matrix of association costs between reference tracks (rows) and candidate tracks (columns)
	 * */
	private double[][] getCostMatrix() {
		double maxDist = 0;
		HashSet<TrackSegment> candidateTrackSet = new HashSet<TrackSegment>();
		for (ArrayList<TrackPair> pairList:feasiblePairs)
		{
			for (TrackPair tp:pairList)
			{
				candidateTrackSet.add(tp.candidateTrack);
				maxDist = Math.max(maxDist, tp.distance);
			}
		}
		ArrayList<TrackSegment> candidateTracksWithDummy = new ArrayList<TrackSegment>();
		candidateTracksWithDummy.addAll(candidateTrackSet);
		double[][] costs = new double[refTracks.size()][candidateTracksWithDummy.size()];
		int refIdx = 0;
		for (ArrayList<TrackPair> pairList:feasiblePairs)
		{
			for (int j = 0; j < candidateTracksWithDummy.size(); j++)
				costs[refIdx][j] = maxDist+1;
			for (TrackPair tp:pairList)
			{
				int candidateIdx = candidateTracksWithDummy.indexOf(tp.candidateTrack);
				tp.candidateIndex = candidateIdx;
				costs[refIdx][tp.candidateIndex] = tp.distance;
			}
			refIdx++;
		}
		return costs;
	}

	/**
	 * Compute the set of feasible pairs between a reference track and candidate and dummy tracks. A pair is not feasible if it does not bring improvement over the association of the reference track with a dummy track.
	 * @param ts the reference TrackSegment object
	 * @param tracks2 the set of candidate tracks
	 * @param distType type of distance that is used for computing the costs of association
	 * @param maxDist the gate (maximum Euclidian distance) for distance computation
	 * 
	 * */
	private ArrayList<TrackPair> getFeasiblePairs(TrackSegment ts, ArrayList<TrackSegment> tracks2, DistanceTypes distType, double maxDist) {
		ArrayList<TrackPair> feasiblePairs = new ArrayList<TrackPair>();
		for (TrackSegment ts2:tracks2)
		{
			TrackToTrackDistance distance = new TrackToTrackDistance(ts, ts2, distType, maxDist);
			if (distance.isMatching)
			{
				TrackPair pair = new TrackPair(ts, ts2, distance.distance, distance.firstMatchingTime, distance.lastMatchingTime);
				feasiblePairs.add(pair);
			}
		}
		// add a dummy track for representing no association
		TrackSegment dummyTrack = new TrackSegment();
		TrackToTrackDistance distance = new TrackToTrackDistance(ts, dummyTrack, distType, maxDist);
		TrackPair pair = new TrackPair(ts, new TrackSegment(), distance.distance, distance.firstMatchingTime, distance.lastMatchingTime);
		feasiblePairs.add(pair);
		return feasiblePairs;
	}

	/**
	 * cluster of TrackPair objects that share common tracks
	 * */
	class TrackPairsCluster
	{
		HashSet<TrackSegment> usedReferenceTracks = new HashSet<TrackSegment>();
		HashSet<TrackSegment> usedCandidateTracks = new HashSet<TrackSegment>();
		ArrayList<TrackPair> trackPairs = new ArrayList<TrackPair>();
		double[][] costs;
		ArrayList<TrackSegment> candidateTrackList;
		ArrayList<TrackSegment> referenceTrackList;

		public void mergeCluster(TrackPairsCluster toMerge)
		{
			this.usedCandidateTracks.addAll(toMerge.usedCandidateTracks);
			this.usedReferenceTracks.addAll(toMerge.usedReferenceTracks);
			this.trackPairs.addAll(toMerge.trackPairs);
		}

		/**
		 * Compute the cost matrix for the assignment between reference and candidate tracks of a TrackPairsCluster object
		 * */
		private void buildCostMatrix() {
			double maxDist = 0;
			for (TrackPair tp:this.trackPairs)
			{
				maxDist = Math.max(maxDist, tp.distance);
			}
			candidateTrackList = new ArrayList<TrackSegment>();
			candidateTrackList.addAll(usedCandidateTracks);
			referenceTrackList = new ArrayList<TrackSegment>();
			referenceTrackList.addAll(usedReferenceTracks);
			costs = new double[referenceTrackList.size()][candidateTrackList.size()];
			// fill costs
			for (int i = 0; i < costs.length; i++)
				for (int j = 0; j < costs[i].length; j++)
					costs[i][j] = maxDist +1;
			for (TrackPair tp:this.trackPairs)
			{
				tp.referenceIndex = referenceTrackList.indexOf(tp.referenceTrack);
				tp.candidateIndex = candidateTrackList.indexOf(tp.candidateTrack);
				costs[tp.referenceIndex][tp.candidateIndex] = tp.distance;
			}
		}

		/**
		 * Build the list of track pairs that corresponds to a given matching matrix
		 * @param matching a boolean matrix indicating the association between reference tracks (rows) and candidate tracks (columns)
		 * @return the list of TrackPair objects that correspond to the matching
		 * */
		private ArrayList<TrackPair> getAssignements(boolean[][] matching) throws Exception
		{
			ArrayList<TrackPair> assignment = new ArrayList<TrackPair>();
			for (int referenceIndex = 0; referenceIndex < referenceTrackList.size(); referenceIndex++)
			{
				boolean found = false;
				int candidateIndex = -1;
				for (int j = 0; j < matching[referenceIndex].length; j++)
				{
					if (matching[referenceIndex][j])
					{
						found = true;
						candidateIndex = j;
						break;
					}
				}
				if (!found)
					throw new Exception("No match found when building assignment");
				found = false;
				for (TrackPair tp:trackPairs)
				{
					if (tp.candidateIndex == candidateIndex && tp.referenceIndex == referenceIndex)
					{
						assignment.add(tp);
						found = true;
						break;
					}
				}
				if (!found)
					throw new Exception("Track pair not found when building assignment");
			}
			return assignment;			
		}
	}
}
