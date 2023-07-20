/**
 * Represents a track. It basically consists in a sequence of subsequent detections.
 * 
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 *
 */


import java.util.ArrayList;

public class TrackSegment {
	ArrayList<Detection> detectionList = new ArrayList<Detection>();
	
	/**
	 * Add a detection to the tail of the track.
	 * The time of this detection has to directly follow the last detection,
	 * so that detections are in chronological order with no gap between them.
	 * 
	 * @param d the detection to enqueue to the track
	 * */
	public void addDetection(Detection d)
	{
		if (detectionList.isEmpty())
			detectionList.add(d);
		else
		{
			if (detectionList.get(detectionList.size()-1).getT()+1!=d.getT())
				throw new IllegalArgumentException("Detections need to be added in a chronological order");
			else
				detectionList.add(d);
		}
	}

	/**
	 * @return a copy of the list of detections
	 * */
	public ArrayList<Detection> getDetectionList()
	{
		ArrayList<Detection> list = new ArrayList<Detection>();
		list.addAll(detectionList);
		return list;
	}
	
	/**
	 * @return the last detection of the track
	 * */
	public Detection getLastDetection()
	{
		if (detectionList.isEmpty())
			return null;
		else
			return detectionList.get(detectionList.size()-1);
	}
	
	/**
	 * @return the first detection of the track
	 * */
	public Detection getFirstDetection()
	{
		if (detectionList.isEmpty())
			return null;
		else
			return detectionList.get(0);
	}
	
	/**
	 * @param t a given frame
	 * @return the detection at time t. null the track does not exist at this time
	 * */
	public Detection getDetectionAtTime(int t)
	{
		if (detectionList.isEmpty())
			return null;
		int firstT = detectionList.get(0).getT();
		if (t<firstT)
			return null;
		if (firstT + detectionList.size() -1 < t)
			return null;
		return detectionList.get(t - firstT);
	}
}
