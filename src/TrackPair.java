
/**
 * A pair of tracks. Includes a reference and a candidate track
 * 
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 *
 */
public class TrackPair {
	TrackSegment referenceTrack;
	TrackSegment candidateTrack;
	double distance;
	int firstMatchingTime;
	int lastMatchingTime;
	int candidateIndex;
	int referenceIndex;
	
	/**
	 * Build the pair.
	 * @param refTrack the reference track
	 * @param cTrack the candidate track
	 * @param dist the distance between the two tracks
	 * @param firstMatchingTime the first time point at which the two tracks match (distance inferior to a gate)
	 * @param lastMatchingTime the last time point at which the two tracks match (distance inferior to a gate)
	 * */
	public TrackPair(TrackSegment refTrack, TrackSegment cTrack, double dist, int firstMatchingTime, int lastMatchingTime)
	{
		this.referenceTrack = refTrack;
		this.candidateTrack = cTrack;
		this.distance = dist;
		this.firstMatchingTime = firstMatchingTime;
		this.lastMatchingTime = lastMatchingTime;
	}
}
