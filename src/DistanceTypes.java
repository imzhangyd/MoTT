/**
 * Types of distances that can be used between detections
 * DISTANCE_EUCLIDIAN corresponds to a gated Euclidian distance
 * DISTANCE_MATCHING corresponds to a binary penalty: 1 if the Euclidian distance between detections is greater than the gate, 0 otherwise
 * 
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 * */
public enum DistanceTypes
{
	DISTANCE_EUCLIDIAN, DISTANCE_MATCHING
}
