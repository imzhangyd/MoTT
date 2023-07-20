/**
 *  Utility for computing the distance between two tracks
 *  
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 *
 */

public class TrackToTrackDistance
{
	double distance;
	boolean isMatching = false;
	int firstMatchingTime = -1;
	int lastMatchingTime = -1;

	int numMatchingDetections = 0;
	int numNonMatchedDetections = 0;
	int numWrongDetections = 0;
	
	/**
	 * Compute the distance between two tracks
	 * @param ts1 the first track
	 * @param ts2 the track with which to compare the first track
	 * @param distanceType the type of distance between detections that is used for the computation
	 * @param maxDist the gate that is used for computing the distance between detections
	 * */
	public TrackToTrackDistance(TrackSegment ts1, TrackSegment ts2, DistanceTypes distanceType, double maxDist)
	{

		if (ts2 == null || ts2.getDetectionList().isEmpty())
		{
			isMatching = false;
			switch (distanceType) {
			case DISTANCE_EUCLIDIAN:
				distance = maxDist*((double)(ts1.getLastDetection().getT() - ts1.getFirstDetection().getT() +1));
				break;
			case DISTANCE_MATCHING:
				distance = ((double)(ts1.getLastDetection().getT() - ts1.getFirstDetection().getT() +1));
				break;
			}
			numMatchingDetections = 0;
			numNonMatchedDetections = (ts1.getLastDetection().getT() - ts1.getFirstDetection().getT() +1);
			numWrongDetections = 0;
			return;
		}
		int t0_1 = ts1.getFirstDetection().getT();
		int tend_1 = ts1.getLastDetection().getT();
		int t0_2 = ts2.getFirstDetection().getT();
		int tend_2 = ts2.getLastDetection().getT();		

		// test if there is an intersection between segments
		if ((t0_2 >= t0_1 && t0_2 <= tend_1) || (tend_2 >= t0_1 && tend_2 <= tend_1) || (t0_2 <= t0_1 && tend_2 >= tend_1) )
		{
			numWrongDetections+=Math.max(0, t0_1 - t0_2);
			numWrongDetections+=Math.max(0, tend_2 - tend_1);
			
			numNonMatchedDetections+=Math.max(0, t0_2 - t0_1);
			numNonMatchedDetections+=Math.max(0, tend_1 - tend_2);

			int firstT = Math.max(t0_1, t0_2);
			int endT = Math.min(tend_1, tend_2);
			switch (distanceType) {
			case DISTANCE_EUCLIDIAN:
			{
				distance = maxDist*(Math.abs(t0_2 - t0_1) + Math.abs(tend_2 - tend_1));				
				for (int t = firstT; t <=endT; t++)
				{
					Detection d1 = ts1.getDetectionAtTime(t);
					Detection d2 = ts2.getDetectionAtTime(t);
					double ed = Math.sqrt((d1.getX()-d2.getX())*(d1.getX()-d2.getX()) + (d1.getY()-d2.getY())*(d1.getY()-d2.getY()) + (d1.getZ()-d2.getZ())*(d1.getZ()-d2.getZ()));
					if (d2.getDetectionType()== Detection.DetectionTypes.REAL && ed<maxDist)
					{
						if (!isMatching)
						{
							firstMatchingTime = t;
							isMatching = true;
						}
						lastMatchingTime = t;
						distance+=ed;
						numMatchingDetections++;
					}
					else
					{
						// virtual detections are not considered as spurious detections
						if (d2.getDetectionType()== Detection.DetectionTypes.REAL)
							numWrongDetections++;
						numNonMatchedDetections++;
						distance+= maxDist;
					}
				}
				break;
			}
			case DISTANCE_MATCHING:
			{
				boolean matching = false;
				distance = (Math.abs(t0_2 - t0_1) + Math.abs(tend_2 - tend_1));
				for (int t = firstT; t <=endT; t++)
				{
					Detection d1 = ts1.getDetectionAtTime(t);
					Detection d2 = ts2.getDetectionAtTime(t);
					double ed = Math.sqrt((d1.getX()-d2.getX())*(d1.getX()-d2.getX()) + (d1.getY()-d2.getY())*(d1.getY()-d2.getY()) + (d1.getZ()-d2.getZ())*(d1.getZ()-d2.getZ()));
					if (d2.getDetectionType()== Detection.DetectionTypes.REAL && ed<maxDist)
					{
						if (!matching)
						{
							firstMatchingTime = t;
							matching = true;
						}
						lastMatchingTime = t;
						// not penalty if matching
						numMatchingDetections++;
					}
					else
					{
						distance++;
						// virtual detections are not considered as spurious detections
						if (d2.getDetectionType()== Detection.DetectionTypes.REAL)
								numWrongDetections++;
						numNonMatchedDetections++;
					}
				}
				break;
			}
			}
		}
		else
		{
			numMatchingDetections = 0;
			numWrongDetections+= (ts2.getLastDetection().getT() - ts2.getFirstDetection().getT() +1);
			numNonMatchedDetections+= (ts1.getLastDetection().getT() - ts1.getFirstDetection().getT() +1);
			isMatching = false;
			switch (distanceType) {
			case DISTANCE_EUCLIDIAN:
				distance = maxDist*((double)(ts1.getLastDetection().getT() - ts1.getFirstDetection().getT() +1));
				break;
			case DISTANCE_MATCHING:
				distance = ((double)(ts1.getLastDetection().getT() - ts1.getFirstDetection().getT() +1));
				break;
			}
		}
	}
}